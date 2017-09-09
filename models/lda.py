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
from IPython.display import display
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import spacy

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

    def infer_topics(self, article_bows):
        return [self.model.__getitem__(article_bow, eps=0.0) for article_bow in article_bows]

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
                assert topic_ids == tuple(range(self.model.num_topics))
                article_topic_matrix.append(topic_probs)
                self.row_article_id_mapping[len(article_topic_matrix) - 1] = article_id
                self.article_id_row_mapping[article_id] = len(article_topic_matrix) - 1
                if not (len(article_topic_matrix) % 500):
                    logger.info(
                        'Saved %d topics for article %d id %d',
                        len(article_topics), len(article_topic_matrix), article_id)
            chunk = list(islice(article_stream, chunk_size))
        self.article_topic_matrix = np.array(article_topic_matrix)

    def get_topic_articles(self, topic_id, min_prob=0.1):
        article_probs = self.article_topic_matrix[:, topic_id]
        trimmed_idxs = np.where(article_probs > min_prob)[0]
        sorted_idxs = sorted(trimmed_idxs, key=lambda idx: -article_probs[idx])
        return [(self.row_article_id_mapping[idx], article_probs[idx]) for idx in sorted_idxs]

    def get_article_topics(self, article_id, min_prob=0.1):
        article_row = self.article_id_row_mapping[article_id]
        topic_probs = self.article_topic_matrix[article_row, :]
        trimmed_idxs = np.where(topic_probs > min_prob)[0]
        sorted_idxs = sorted(trimmed_idxs, key=lambda idx: -topic_probs[idx])
        return [(idx, topic_probs[idx]) for idx in sorted_idxs]

    def plot_topic(self, topic_id, min_prob=0.1, window_size=30):
        article_ids, article_probs = zip(*self.get_topic_articles(topic_id, min_prob))
        articles = self.corpus.get_articles(article_ids)
        articles['topic_score'] = article_probs
        topic_score_over_time = articles.groupby('created_date')['topic_score'].sum()
        topic_score_over_time = pd.rolling_mean(topic_score_over_time, window=window_size, center=True)
        plot_data = [go.Scatter(x=topic_score_over_time.index, y=topic_score_over_time.values)]
        print(self.model.print_topic(topic_id))
        return py.iplot(plot_data)

    def show_topic_articles(self, topic_id, min_prob=0.1, max_article_length=500):
        print(self.model.print_topic(topic_id))
        article_ids_and_probs = self.get_topic_articles(topic_id, min_prob)
        if not article_ids_and_probs:
            print('No articles found for topic %d' % topic_id)
            return
        article_ids, article_probs = zip(*article_ids_and_probs)
        articles = pd.DataFrame(self.corpus.get_articles(article_ids))
        articles['topic_score'] = article_probs
        for article_id in article_ids:
            self.corpus.print_article(article_id)
        return articles

    def show_article_topics(self, article_id, min_prob=0.1, max_article_length=500):
        self.corpus.print_article(article_id)
        topic_ids_and_probs = self.get_article_topics(article_id, min_prob)
        for topic_id, topic_prob in topic_ids_and_probs:
            print('Topic #%d (%.2f): %s' % (topic_id, topic_prob, self.model.print_topic(topic_id)))


class HnLdaMalletModel(HnLdaModel):
    def __init__(self, mallet_path, corpus, workers=1, **model_params):
        super(HnLdaMalletModel, self).__init__(corpus, workers, model_params)
        self.mallet_path = mallet_path

    def infer_topics(self, article_bows):
        return self.model[article_bows]

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
