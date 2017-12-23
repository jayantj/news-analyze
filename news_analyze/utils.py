#! /usr/bin/env python

import argparse
import logging
import os
import pickle
import multiprocessing

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from dateutil import parser as date_parser
import datetime
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import spacy


logger = logging.getLogger(__name__)
nlp = spacy.load('en')
DATA_DIR = 'data/'


def parse_date(timestamp):
    return timestamp.date()

def parse_month_year(timestamp):
    return datetime.datetime(year=timestamp.year, month=timestamp.month, day=1)

def decode_bytes(bytes_, encoding='utf8', max_char_bytes=4):
    try:
        return bytes_.decode(encoding, 'strict')
    except UnicodeDecodeError:
        # Try skipping last few bytes since they could have been truncated
        last_bytes_to_ignore = 1
        while last_bytes_to_ignore < max_char_bytes:
            try:
                return bytes_[:-last_bytes_to_ignore].decode(encoding, 'strict')
            except UnicodeDecodeError:
                last_bytes_to_ignore += 1
                if last_bytes_to_ignore == max_char_bytes:
                    # Text cannot be parsed using given encoding
                    raise


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

    def get(self, item_id):
        if not self.item_index:
            self.load_index()
        with open(self.filename, 'rb') as f:
            item_position = self.item_index[item_id]
            f.seek(item_position)
            item = pickle.load(f)
        return item

    def __iter__(self):
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


class HnCorpus(object):
    def __init__(self, dirname, encoding='utf8', metadata={}, cache_path=None, min_count=5, max_df=0.6):
        self.dirname = dirname
        self.encoding = encoding
        self.dictionary = None
        if len(metadata):
            metadata['created_at'] = pd.to_datetime(metadata['created_at'])
            metadata['created_date'] = metadata['created_at'].apply(parse_date)
        self.metadata = metadata
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = '%s.tokens.cache' % os.path.basename(dirname)
        self.cache = None
        self.phrases = None
        self.min_count = min_count
        self.max_df = max_df

    def init_dict(self, reset=False):
        if self.dictionary is not None and not reset:
            return
        dictionary = Dictionary(article_tokens for _, article_tokens in self.stream_articles_tokens())
        dictionary.filter_extremes(no_below=self.min_count, no_above=self.max_df)
        self.dictionary = dictionary

    def init_phrases(self):
        self.phrases = Phrases(article_tokens for _, article_tokens in self.stream_articles_tokens())

    def init_cache(self):
        if self.cache is None:
            if os.path.exists(self.cache_path):
                self.cache = FileStreamingCorpus.init_from_file(self.cache_path)
            else:
                self.cache = FileStreamingCorpus.init_from_stream(self.article_tokens_from_text(), self.cache_path)

    def __contains__(self, article_id):
        full_filename = os.path.join(self.dirname, "%d.txt" % article_id)
        return os.path.isfile(full_filename)

    def stream_articles_text(self, max_count=None):
        count = 0
        for filename in os.listdir(self.dirname):
            article_id = int(os.path.splitext(filename)[0])
            full_filename = os.path.join(self.dirname, filename)
            if not os.path.isfile(full_filename):
                continue
            with open(full_filename, 'rb') as f:
                article_text = f.read().decode(self.encoding)
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
            yield article_id, self.text_to_tokens(article_text)

    @staticmethod
    def text_to_tokens(article_text):
        doc = nlp(article_text)
        article_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        return article_tokens

    def article_tokens_from_disk(self, max_count=None):
        self.init_cache()
        for i, (article_id, article_tokens) in enumerate(self.cache, start=1):
            yield article_id, article_tokens
            if max_count is not None and i >= max_count:
                break

    def __getitem__(self, article_id):
        if not self.cache_path:
            raise ValueError('Cannot fetch random article without cache')
        self.init_cache()
        article_id, article_tokens = self.cache.get(article_id)
        return self.dictionary.doc2bow(article_tokens)

    def get_articles(self, article_ids):
        if not len(self.metadata):
            raise ValueError('Cannot fetch article objects without metadata')
        idx = pd.Index(self.metadata['id']).get_indexer(article_ids)
        return self.metadata.iloc[idx]

    def get_article_text(self, article_id, max_length=None):
        article_filename = os.path.join(self.dirname, '%d.txt' % article_id)
        with open(article_filename, 'rb') as f:
            if max_length is not None:
                article_text = decode_bytes(f.read(max_length), self.encoding)
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

    def print_article(self, article_id, max_article_length=500, score=None):
        article_metadata = self.get_articles([article_id]).iloc[0]
        title = article_metadata['title']
        url = article_metadata['url']
        print('---------------------------------------------------------------------')
        print('Article #%d - %s\n%s' % (article_id, url, title))
        if score:
            print('Topic score: %.2f\n' % score)
        article_text = self.get_article_text(article_id, max_length=max_article_length)
        print('Article text:\n %s (...)(trimmed)' % article_text)
        print('\n')


class HnDtmCorpus(HnCorpus):
    def get_time_slices(self):
        article_ids = list(self.stream_article_ids())
        article_metadata = self.get_articles(article_ids)
        created_month_year = article_metadata['created_at'].apply(parse_month_year)
        counts = created_month_year.groupby(created_month_year).count()
        assert counts.sum() == len(article_ids)
        return list(counts.values)

    def stream_article_ids(self, max_count=None):
        if not len(self.metadata):
            raise ValueError('Cannot use HnDtmCorpus without metadata')
        for article_id in self.metadata.sort_values('created_at')['id'].values:
            if article_id in self:
                yield article_id

    def article_tokens_from_text(self, max_count=None):
        for article_id in self.stream_article_ids(max_count):
            article_text = self.get_article_text(article_id)
            doc = nlp(article_text)
            article_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            yield article_id, article_tokens

    def article_tokens_from_disk(self, max_count=None):
        self.init_cache()
        for i, article_id in enumerate(self.stream_article_ids(), start=1):
            _, article_tokens = self.cache.get(article_id)
            yield article_id, article_tokens
            if max_count is not None and i >= max_count:
                break
