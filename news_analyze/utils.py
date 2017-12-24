#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>

"""
Corpuses for fetching article data - text, bag-of-words, and metadata.

"""

import os
import pickle

from gensim.corpora import Dictionary
from gensim.models import Phrases
import datetime
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import spacy


nlp = spacy.load('en')


class ArticleTokenCache(object):
    def __init__(self, file_path=None, stream=None):
        self.file_path = file_path
        self.item_index = {}

        if not os.path.exists(file_path):
            if stream is None:
                raise ValueError('No stream and non-existent file %s specified' % file_path)
            self._serialize_stream(stream)

    def _serialize_stream(self, stream):
        num_bytes_written = 0
        assert self.item_index == {}
        with open(self.file_path, 'wb') as f:
            for item_id, item in stream:
                self.item_index[item_id] = num_bytes_written
                pickled_bytes = pickle.dumps((item_id, item))
                f.write(pickled_bytes)
                num_bytes_written += len(pickled_bytes)
        with open(self.file_path + '.idx', 'wb') as f:
            pickle.dump(self.item_index, f)

    def _load_index(self):
        with open(self.file_path + '.idx', 'rb') as f:
            self.item_index = pickle.load(f)

    def get(self, item_id):
        if not self.item_index:
            self._load_index()
        with open(self.file_path, 'rb') as f:
            item_position = self.item_index[item_id]
            f.seek(item_position)
            item = pickle.load(f)
        return item

    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


class HnCorpus(object):

    def __init__(self, data_dir, metadata_file, encoding='utf8', cache_path=None, min_count=5, max_df=0.6):
        self.data_dir = data_dir
        self.encoding = encoding
        self.dictionary = None
        self.metadata_file = metadata_file
        self._load_metadata()
        if cache_path is not None:
            self.token_cache_path = cache_path
        else:
            data_dirname = data_dir.rstrip(os.sep).split(os.sep)[-1]
            self.token_cache_path = '%s.tokens.cache' % data_dirname
        self.token_cache = None
        self.phrases = None
        self.min_count = min_count
        self.max_df = max_df

    def _load_metadata(self):
        self.metadata = pd.read_csv(self.metadata_file, parse_dates=['created_at'])
        self.metadata['created_date'] = self.metadata['created_at'].apply(lambda t: t.date())

    def init_dict(self, reset=False):
        if self.dictionary is not None and not reset:
            return
        dictionary = Dictionary(article_tokens for _, article_tokens in self._stream_articles_tokens())
        dictionary.filter_extremes(no_below=self.min_count, no_above=self.max_df)
        self.dictionary = dictionary

    def _init_cache(self):
        if self.token_cache is None:
            self.token_cache = ArticleTokenCache(self.token_cache_path, self._article_tokens_from_text())

    def _stream_articles_text(self, max_count=None):
        count = 0
        for filename in os.listdir(self.data_dir):
            article_id = int(os.path.splitext(filename)[0])
            full_filename = os.path.join(self.data_dir, filename)
            if not os.path.isfile(full_filename):
                continue
            with open(full_filename, 'rb') as f:
                article_text = f.read().decode(self.encoding)
            yield article_id, article_text
            count += 1
            if max_count and count >= max_count:
                break

    def _stream_articles_tokens(self, max_count=None):
        if self.token_cache_path:
            article_stream = self._article_tokens_from_cache(max_count)
        else:
            article_stream = self._article_tokens_from_text(max_count)
        if self.phrases is None:
            return article_stream
        else:
            return ((article_id, self.phrases[article_tokens]) for article_id, article_tokens in article_stream)

    def _article_tokens_from_text(self, max_count=None):
        for i, (article_id, article_text) in enumerate(self._stream_articles_text(max_count), start=1):
            yield article_id, self._text_to_tokens(article_text)
            if max_count is not None and i >= max_count:
                break

    @staticmethod
    def _text_to_tokens(article_text):
        doc = nlp(article_text)
        article_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        return article_tokens

    def _article_tokens_from_cache(self, max_count=None):
        self._init_cache()
        for i, (article_id, article_tokens) in enumerate(self.token_cache, start=1):
            yield article_id, article_tokens
            if max_count is not None and i >= max_count:
                break

    def __getitem__(self, article_id):
        if not self.token_cache_path:
            raise ValueError('Cannot fetch random article without cache')
        self._init_cache()
        article_id, article_tokens = self.token_cache.get(article_id)
        return self.dictionary.doc2bow(article_tokens)

    def get_articles_metadata(self, article_ids):
        if not len(self.metadata):
            raise ValueError('Cannot fetch article objects without metadata')
        idx = pd.Index(self.metadata['id']).get_indexer(article_ids)
        return self.metadata.iloc[idx]

    def get_article_text(self, article_id, max_length=None):
        article_filename = os.path.join(self.data_dir, '%d.txt' % article_id)
        with open(article_filename, 'rb') as f:
            article_text = f.read(max_length).decode(self.encoding)
        if max_length is not None:
            article_text = article_text[:max_length]
        return article_text

    def get_bow_from_text(self, article_text):
        return self.dictionary.doc2bow(self._text_to_tokens(article_text))

    def __iter__(self):
        return self.stream_bow(stream_ids=False)

    def stream_bow(self, max_count=None, stream_ids=True):
        self.init_dict()
        for article_id, article_tokens in self._stream_articles_tokens(max_count):
            if stream_ids:
                yield article_id, self.dictionary.doc2bow(article_tokens)
            else:
                yield self.dictionary.doc2bow(article_tokens)

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

