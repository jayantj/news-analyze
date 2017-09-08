#! /usr/bin/env python

import argparse
import logging
import os
import pickle
import multiprocessing

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from dateutil import parser as date_parser
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import spacy


logger = logging.getLogger(__name__)
nlp = spacy.load('en')
DATA_DIR = 'data/'


def parse_date(date_str):
    parsed = date_parser.parse(date_str)
    return parsed.date()


class Article(object):
    def __init__(self, id_, text, metadata={}, min_points=50):
        self.id = id_
        self.text = text
        self.metadata = metadata
        self.min_points = min_points

    def __str__(self):
        return self.text

    @classmethod
    def init_from_df(cls, df, id_, text):
        pd_metadata_dict = df[df.id == id_][['title', 'url', 'num_points', 'author', 'created_at']].to_dict()
        metadata_dict = {}
        for field, field_dict in pd_metadata_dict.items():
            assert len(field_dict) == 1
            metadata_dict[field] = list(field_dict.values())[0]
        return cls(id_, text, metadata_dict)

    def is_valid(self):
        return self.metadata['num_points'] >= self.min_points


class FileStreamingCorpus(object):
    def __init__(self, filename, stream=None):
        self.filename = filename
        self.stream = stream
        self.item_index = {}

    def save(self):
        num_bytes_written = 0
        assert self.item_index == {}
        with open(self.filename, 'wb') as f:
            for item_id, item in self.stream:
                self.item_index[item_id] = num_bytes_written
                pickled_bytes = pickle.dumps((item_id, item))
                f.write(pickled_bytes)
                num_bytes_written += len(pickled_bytes)
        with open(self.filename + '.idx', 'wb') as f:
            pickle.dump(self.item_index, f)

    @classmethod
    def init_from_file(cls, filename):
        return cls(filename)

    @classmethod
    def init_from_stream(cls, stream, filename):
        corpus = cls(filename, stream)
        corpus.save()
        return corpus

    def load_index(self):
        self.item_index = pickle.load(open(self.filename + '.idx', 'rb'))

    def get(self, item_ids):
        if not self.item_index:
            self.load_index()
        items = []
        with open(self.filename, 'rb') as f:
            for item_id in item_ids:
                item_position = self.item_index[item_id]
                f.seek(item_position)
                items.append(pickle.load(f))
        return items

    def __iter__(self):
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


class HnCorpus(object):
    def __init__(self, dirname, encoding='utf8', preprocess=True, metadata={}, cache_path=None):
        self.dirname = dirname
        self.encoding = encoding
        self.preprocess = preprocess
        self.dictionary = None
        if len(metadata):
            metadata['created_date'] = metadata['created_at'].apply(parse_date)
        self.metadata = metadata
        self.cache_path = cache_path
        self.cache = None
        self.phrases = None

    def init_dict(self, reset=False):
        if self.dictionary is not None and not reset:
            return
        dictionary = Dictionary(article_tokens for _, article_tokens in self.stream_articles_tokens())
        dictionary.filter_extremes(no_below=5, no_above=0.6)
        self.dictionary = dictionary

    def init_phrases(self):
        self.phrases = Phrases(article_tokens for _, article_tokens in self.stream_articles_tokens())

    def init_cache(self):
        if self.cache is None:
            if os.path.exists(self.cache_path):
                self.cache = FileStreamingCorpus.init_from_file(self.cache_path)
            else:
                self.cache = FileStreamingCorpus.init_from_stream(self.article_tokens_from_text(), self.cache_path)

    def stream_articles(self, max_count=None):
        for article_id, article_text in self.stream_articles_text(max_count):
            if len(self.metadata):
                article = Article.init_from_df(self.metadata, article_id, article_text)
                if article.is_valid():
                    yield article
                else:
                    continue
            else:
                yield Article(article_id, article_text)

    def stream_articles_text(self, max_count=None):
        count = 0
        for filename in os.listdir(self.dirname):
            article_id = int(os.path.splitext(filename)[0])
            full_filename = os.path.join(self.dirname, filename)
            if not os.path.isfile(full_filename):
                continue
            article_text = open(full_filename, 'rb').read().decode(self.encoding)
            yield article_id, article_text
            count += 1
            if max_count and count >= max_count:
                break

    def stream_articles_tokens(self, max_count=None):
        if self.cache_path:
            article_stream = self.article_tokens_from_disk(max_count)
        else:
            article_stream = self.article_tokens_from_text(max_count)
        if self.phrases is None:
            return article_stream
        else:
            return ((article_id, self.phrases[article_tokens]) for article_id, article_tokens in article_stream)

    def article_tokens_from_text(self, max_count=None):
        for article_id, article_text in self.stream_articles_text(max_count):
            doc = nlp(article_text) 
            article_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            yield article_id, article_tokens

    def article_tokens_from_disk(self, max_count=None):
        self.init_cache()
        return self.cache.__iter__()

    def __getitem__(self, article_ids):
        if not self.cache_path:
            raise ValueError('Cannot fetch random article without cache')
        self.init_cache()
        articles = self.cache.get(article_ids)
        return [self.dictionary.doc2bow(article_tokens) for _, article_tokens in articles]

    def get_articles(self, article_ids):
        if not len(self.metadata):
            raise ValueError('Cannot fetch article objects without metadata')
        idx = pd.Index(self.metadata['id']).get_indexer(article_ids)
        return self.metadata.iloc[idx]

    def get_article_text(self, article_id, max_length=None):
        article_filename = os.path.join(self.dirname, '%d.txt' % article_id)
        with open(article_filename, 'rb') as f:
            if max_length is not None:
                article_text = f.read(max_length).decode(self.encoding)
            else:
                article_text = f.read().decode(self.encoding)
        return article_text

    def __iter__(self):
        return self.stream_bow(stream_ids=False)

    def stream_bow(self, max_count=None, stream_ids=True):
        self.init_dict()
        for article_id, article_tokens in self.stream_articles_tokens(max_count):
            if stream_ids:
                yield article_id, self.dictionary.doc2bow(article_tokens)
            else:
                yield self.dictionary.doc2bow(article_tokens)

    def plot_articles(self, article_ids, window_size=5):
        created_date = self.metadata[self.metadata.id.isin(article_ids)]['created_date']
        created_date_counts = created_date.value_counts().sort_index()
        created_date_counts = pd.rolling_mean(created_date_counts, window=window_size, center=True)
        plot_data = [go.Scatter(x=created_date_counts.index, y=created_date_counts.values)]
        return py.iplot(plot_data)
