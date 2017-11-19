#! /usr/bin/env python

import argparse
from collections import defaultdict
import logging
import os
import pickle
import multiprocessing

from gensim.models import ldamodel, hdpmodel, ldamulticore
from gensim.models.wrappers import ldamallet
from gensim.corpora import Dictionary
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from newspaper import Article
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

from .utils import HnCorpus, parse_date


logger = logging.getLogger(__name__)


class HnLdaModel(object):

    def __init__(self, corpus, workers=1, **model_params):
        self.corpus = corpus
        self.model = None
        self.workers = workers
        self.model_params = model_params
        self.article_topic_matrix = None
        self.row_article_id_mapping = {}
        self.article_id_row_mapping = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def init_from_model(cls, corpus, lda_model):
        hn_lda_model = cls(corpus)
        hn_lda_model.model = lda_model
        hn_lda_model.save_article_topics()
        return hn_lda_model

    @classmethod
    def load(self, filename):
        return pickle.load(open(filename, 'rb'))

    def train(self):
        if self.workers > 1:
            self.model = ldamulticore.LdaMulticore(
                self.corpus,
                **dict(
                    self.model_params,
                    id2word=self.corpus.dictionary,
                    workers=self.workers)
                )
        else:
            self.model = ldamodel.LdaModel(
                self.corpus,
                **dict(
                    self.model_params,
                    id2word=self.corpus.dictionary)
                )
        self.save_article_topics()
        self.save_topic_words()

    def infer_topics(self, article_bows):
        return [self.model.__getitem__(article_bow, eps=0.0) for article_bow in article_bows]

    def save_topic_words(self):
        self.topic_word_matrix = self.model.state.get_lambda()

    def save_article_topics(self):
        article_stream = self.corpus.stream_bow(stream_ids=True)
        chunk_size = 50
        chunk = list(islice(article_stream, chunk_size))
        article_topic_matrix = []
        while chunk:
            article_ids, article_bows = zip(*chunk)
            chunk_topics = self.infer_topics(article_bows)
            for article_id, article_topics in zip(article_ids, chunk_topics):
                topic_ids, topic_probs = zip(*article_topics)
                prob_scores = np.zeros(self.model.num_topics)
                prob_scores[list(topic_ids)] = topic_probs
                article_topic_matrix.append(prob_scores)
                self.row_article_id_mapping[len(article_topic_matrix) - 1] = article_id
                self.article_id_row_mapping[article_id] = len(article_topic_matrix) - 1
                if not (len(article_topic_matrix) % 500):
                    logger.info(
                        'Saved %d topics for article %d id %d',
                        len(article_topics), len(article_topic_matrix), article_id)
            chunk = list(islice(article_stream, chunk_size))
        self.article_topic_matrix = np.array(article_topic_matrix)
        self.init_topic_similarity_matrices()
        self.init_topic_scores()

    def print_topic_row(self, topic_ids, topic_scores=[], top_n=10):
        topic_labels = ["Topic #%d" % topic_id for topic_id in topic_ids]
        row_format = "{:^15}" * (len(topic_ids))
        topics = row_format.format(*topic_labels)
        underscores = row_format.format(*['----------'] * len(topic_ids))
        if topic_scores:
            score_labels = ["Score (%.2f)" % topic_score for topic_score in topic_scores]
            scores = row_format.format(*score_labels)
            lines = [topics, scores, underscores]
        else:
            lines = [topics, underscores]
        def get_topic_words(topic_id, model, top_n):
            return [w for w, _ in model.show_topic(topic_id, top_n)]

        topic_words = {topic_id: get_topic_words(topic_id, self.model, top_n) for topic_id in topic_ids}
        for i in range(top_n):
            row_words = [topic_words[topic_id][i] for topic_id in topic_ids]
            lines.append(row_format.format(*row_words))
        lines.append('\n')
        for line in lines:
            print(line)

    def print_topics_table(self, topic_ids=[], topic_scores=[], topics_per_row=6, top_n=10):
        if not topic_ids:
            topic_ids = np.argsort(-self.topic_scores)
        assert not topic_scores or len(topic_scores) == len(topic_ids)
        for i in range(0, len(topic_ids), topics_per_row):
            if topic_scores:
                self.print_topic_row(topic_ids[i:i+topics_per_row], topic_scores[i:i+topics_per_row], top_n=top_n)
            else:
                self.print_topic_row(topic_ids[i:i+topics_per_row], top_n=top_n)

    def get_topic_articles(self, topic_ids, min_prob=0.1, negative_ids=[], top_n=None):
        if not isinstance(topic_ids, (list, tuple)):
            topic_ids = [topic_ids]
        article_probs = self.article_topic_matrix[:, topic_ids].sum(axis=1) / len(topic_ids)
        trimmed_indices = np.where(article_probs > min_prob)[0]
        sorted_indices = sorted(trimmed_indices, key=lambda index: -article_probs[index])
        if top_n:
            sorted_indices = sorted_indices[:top_n]
        return [(self.row_article_id_mapping[index], article_probs[index]) for index in sorted_indices]

    def show_topic_label(self, topic_id, num_words=10):
        top_words = [word[0] for word in self.model.show_topic(topic_id, num_words)]
        return "%s" % (', '.join(top_words))

    def get_article_topics(self, article_id, min_prob=0.1):
        article_row = self.article_id_row_mapping[article_id]
        topic_probs = self.article_topic_matrix[article_row, :]
        trimmed_idxs = np.where(topic_probs > min_prob)[0]
        sorted_idxs = sorted(trimmed_idxs, key=lambda idx: -topic_probs[idx])
        return [(idx, topic_probs[idx]) for idx in sorted_idxs]

    def get_article_topics_from_url(self, article_url, min_prob=0.1):
        article = self.article_from_url(article_url)
        return self.get_article_topics_from_text(article.text, min_prob)

    def get_article_topics_from_text(self, article_text, min_prob=0.1):
        article_tokens = self.corpus.text_to_tokens(article_text)
        article_bow = self.corpus.dictionary.doc2bow(article_tokens)
        article_topics = self.model[article_bow]
        article_topics = [(topic_id, score) for topic_id, score in article_topics if score > min_prob]
        return sorted(article_topics, key=lambda x: -x[1])

    def get_topic_trend(self, topic_id, min_prob=0.1, window_size=30):
        article_ids, article_probs = zip(*self.get_topic_articles(topic_id, min_prob))
        articles = self.corpus.get_articles(article_ids)
        articles = articles.assign(topic_score=article_probs)
        topic_score_over_time = articles.groupby('created_date')['topic_score'].sum()
        topic_score_over_time = topic_score_over_time.rolling(window=window_size, center=True).mean()
        return topic_score_over_time
        # return (topic_score_over_time.index, topic_score_over_time.values)

    def topic_trend_plot(self, topic_id, min_prob=0.1, window_size=30):
        topic_trend = self.get_topic_trend(topic_id, min_prob, window_size)
        plot_data = [go.Scatter(x=topic_trend.index, y=topic_trend.values, name='Topic #%d' % topic_id)]
        self.print_topics(topic_id)
        return plot_data

    def show_topic_articles(self, topic_ids, negative_ids=[], min_prob=0.1, max_article_length=500, top_n=None):
        if not isinstance(topic_ids, (list, tuple)):
            topic_ids = [topic_ids]
        self.print_topics_table(topic_ids)
        article_ids_and_probs = self.get_topic_articles(topic_ids, min_prob, negative_ids, top_n)
        if not article_ids_and_probs:
            print('No articles found for topic %d' % topic_ids)
            return []
        for article_id, article_score in article_ids_and_probs:
            self.corpus.print_article(article_id, max_article_length, score=article_score)

    def plot_topic_similarities(self, metric='word_doc_sim'):
        similarity_matrix = np.copy(self.get_similarity_matrix(metric))
        np.fill_diagonal(similarity_matrix, 0)
        return plt.matshow(similarity_matrix, cmap=plt.cm.binary)

    def show_article_topics(self, article_id, min_prob=0.1, max_article_length=500):
        self.corpus.print_article(article_id, max_article_length)
        topic_ids_and_probs = self.get_article_topics(article_id, min_prob)
        topic_ids, topic_probs = zip(*topic_ids_and_probs)
        self.print_topics_table(topic_ids, topic_scores=topic_probs)

    def show_article_topics_from_text(self, article_text, min_prob=0.1, max_article_length=500):
        print('Article text:\n %s (...)(trimmed)' % article_text[:max_article_length])
        print('\nMost relevant topics:\n')
        topic_ids_and_probs = self.get_article_topics_from_text(article_text, min_prob)
        topic_ids, topic_probs = zip(*topic_ids_and_probs)
        self.print_topics_table(topic_ids, topic_scores=topic_probs)

    def show_article_topics_from_url(self, article_url, min_prob=0.1, max_article_length=500):
        print('Article: %s' % article_url)
        article = self.article_from_url(article_url)
        self.show_article_topics_from_text(article.text, min_prob, max_article_length)

    @staticmethod
    def article_from_url(article_url):
        article = Article(article_url)
        article.download()
        article.parse()
        return article

    def init_topic_scores(self, min_threshold=0.1):
        topic_vectors_mask = self.article_topic_matrix < min_threshold
        topic_vectors_masked = np.ma.array(self.article_topic_matrix, mask=topic_vectors_mask)
        topic_vector_scores = np.ma.median(topic_vectors_masked, axis=0)
        self.topic_scores = topic_vector_scores

    def init_topic_similarity_matrices(self, jaccard_threshold=0.15):
        # doc-based similarity
        self.topic_doc_similarities = cosine_similarity(self.article_topic_matrix.T, self.article_topic_matrix.T)

        # word-based similarity
        self.topic_word_similarities = cosine_similarity(self.topic_word_matrix, self.topic_word_matrix)

        # jaccard similarity
        topic_vectors = self.article_topic_matrix.T > jaccard_threshold
        topic_vectors_repeated = np.repeat(topic_vectors[:, :, np.newaxis], topic_vectors.shape[0], axis=2).swapaxes(0, 2)
        topic_vectors_intersection = np.logical_and(topic_vectors[:, :, np.newaxis], topic_vectors_repeated).sum(axis=1)
        topic_vectors_union = np.logical_or(topic_vectors[:, :, np.newaxis], topic_vectors_repeated).sum(axis=1)
        topic_jaccard_similarities = topic_vectors_intersection / topic_vectors_union
        self.topic_jaccard_similarities = topic_jaccard_similarities

        # doc-word based similarity
        word_similarities = np.copy(self.topic_word_similarities)
        doc_similarities = np.copy(self.topic_doc_similarities)
        np.fill_diagonal(word_similarities, 0)  # Remove topic similarities with itself
        np.fill_diagonal(doc_similarities, 0)  # Remove topic similarities with itself
        scale_ratio = np.max(doc_similarities) / np.max(word_similarities)
        word_similarities = scale_ratio * word_similarities
        np.fill_diagonal(word_similarities, 1)
        np.fill_diagonal(doc_similarities, 1)
        self.topic_word_doc_similarities = (word_similarities + doc_similarities) / 2.0

    def get_similarity_matrix(self, metric):
        if metric == 'word_doc_sim':
            return self.topic_word_doc_similarities
        if metric == 'word_sim':
            return self.topic_word_similarities
        elif metric == 'doc_sim':
            return self.topic_doc_similarities
        elif metric == 'jaccard':
            return self.topic_jaccard_similarities
        else:
            raise ValueError('Invalid similarity metric')

    def get_similar_topics(self, topic_id, top_n=10, min_similarity=0.0, metric='word_doc_sim'):
        topic_similarities = self.get_similarity_matrix(metric)
        similar_topics = np.argsort(-topic_similarities[topic_id, :])[1:1+top_n]  # Remove index for similarity with self
        return similar_topics, topic_similarities[topic_id][similar_topics]

    def print_topics(self, topic_ids=[]):
        if not isinstance(topic_ids, (list, tuple)):
            topic_ids = [topic_ids]
        if not topic_ids:
            topic_ids = np.argsort(-self.topic_scores)
        for topic_id in topic_ids:
            print('Topic #%d: %s\n' % (topic_id, self.model.print_topic(topic_id)))

    def show_similar_topics(self, topic_id, top_n=10, min_similarity=0.0, metric='word_doc_sim'):
        similar_topics, similarities = self.get_similar_topics(topic_id, top_n, min_similarity, metric)
        self.print_topics_table([topic_id])
        print('Topics similar to topic #%d\n---------------------------\n' % topic_id)
        self.print_topics_table(list(similar_topics), list(similarities))
        # for similar_topic_id, similarity in zip(similar_topics, similarities):
            # print('Topic #%d (%.2f): %s\n' % (similar_topic_id, similarity, self.model.print_topic(similar_topic_id)))

    def get_most_similar_topic_pairs(self, top_n=20, metric='word_doc_sim'):
        topic_similarities = np.copy(self.get_similarity_matrix(metric))
        np.fill_diagonal(topic_similarities, 0)  # Remove topic similarities with itself
        num_topics = topic_similarities.shape[0]
        topic_similarities = topic_similarities.reshape(-1)
        indices = np.argsort(-topic_similarities)
        topic_pairs, topic_scores = [], []
        for idx in indices:
            if len(topic_pairs) == top_n:
                break
            topic_1, topic_2 = idx // num_topics, idx % num_topics
            if topic_2 > topic_1:  # Avoid repeating same topic combination twice
                continue
            topic_pairs.append((topic_1, topic_2))
            topic_scores.append(topic_similarities[idx])
        return topic_pairs, topic_scores

    def show_most_similar_topic_pairs(self, top_n=20, metric='word_doc_sim'):
        topic_pairs, topic_scores = self.get_most_similar_topic_pairs(top_n, metric)
        for topic_pair, topic_score in zip(topic_pairs, topic_scores):
            print(topic_score)
            self.print_topics(topic_pair)


class HnLdaMalletModel(HnLdaModel):
    def __init__(self, mallet_path, corpus, workers=1, **model_params):
        super(HnLdaMalletModel, self).__init__(corpus, workers, **model_params)
        self.mallet_path = mallet_path

    def infer_topics(self, article_bows):
        return self.model[article_bows]

    def save_topic_words(self):
        self.topic_word_matrix = self.model.wordtopics / self.model.wordtopics.sum(axis=1)[:, np.newaxis]

    def train(self):
        self.model = ldamallet.LdaMallet(
            self.mallet_path,
            self.corpus,
            **dict(
                self.model_params,
                id2word=self.corpus.dictionary,
                workers=self.workers)
            )
        self.save_article_topics()
        self.save_topic_words()