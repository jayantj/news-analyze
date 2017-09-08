#! /usr/bin/env python

import argparse
from collections import defaultdict
import logging
import os
import pickle
import multiprocessing

from gensim.models import ldamodel, hdpmodel, ldamulticore
from gensim.corpora import Dictionary
from itertools import islice
from IPython.display import display
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
        self.article_topic_map = defaultdict(list)
        self.topic_article_map = defaultdict(list)
        self.init_from_model = False
        self.article_topic_probs = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def init_from_disk(cls, lda_path, corpus_path):
        lda_model = ldamulticore.LdaMulticore.load(lda_path)
        corpus = pickle.load(open(corpus_path, 'rb'))
        hn_lda_model = cls(corpus)
        hn_lda_model.init_from_model = True
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

    def save_article_topics(self):
        article_stream = self.corpus.stream_bow(stream_ids=True)
        chunk_size = 50
        num_saved = 0
        chunk = list(islice(article_stream, chunk_size))
        while chunk:
            article_ids, article_bows = zip(*chunk)
            chunk_topics = self.model[article_bows]
            for article_id, article_topics in zip(article_ids, chunk_topics):
                num_saved += 1
                for topic_id, topic_prob in article_topics:
                    self.article_topic_map[article_id].append(topic_id)
                    self.topic_article_map[topic_id].append(article_id)
                    self.article_topic_probs[(article_id, topic_id)] = topic_prob
                if not (num_saved % 500):
                    logger.info('Saved %d topics for article %d id %d', len(article_topics), num_saved, article_id)
            chunk = list(islice(article_stream, chunk_size))

    def plot_topic(self, topic_id, window_size=30):
        # for topic_id in topic_ids:
        article_ids = self.topic_article_map[topic_id]
        articles = self.corpus.get_articles(article_ids)
        topic_probs = [self.article_topic_probs[(article_id, topic_id)] for article_id in article_ids]
        articles['topic_prob'] = topic_probs
        topic_importance_over_time = articles.groupby('created_date')['topic_prob'].sum()
        topic_importance_over_time = pd.rolling_mean(topic_importance_over_time, window=window_size, center=True)
        plot_data = [go.Scatter(x=topic_importance_over_time.index, y=topic_importance_over_time.values)]
        print(self.model.print_topic(topic_id))
        return py.iplot(plot_data)

    def show_topic_articles(self, topic_id, max_article_length=500, threshold=0.1):
        article_ids = self.topic_article_map[topic_id]
        article_ids = [article_id for article_id in article_ids if self.article_topic_probs[(article_id, topic_id)] > threshold]
        topic_probs = [self.article_topic_probs[(article_id, topic_id)] for article_id in article_ids]
        articles = pd.DataFrame(self.corpus.get_articles(article_ids))
        articles['topic_prob'] = topic_probs
        print(self.model.print_topic(topic_id))
        display(articles)
        for article_id in article_ids:
            article_metadata = articles[articles.id == article_id]
            title = article_metadata.iloc[0]['title']
            url = article_metadata.iloc[0]['url']
            print('---------------------------------------------------------------------')
            print('Article #%d - %s\n%s\n' % (article_id, url, title))
            print(self.corpus.get_article_text(article_id, max_length=max_article_length))
            print('\n')
