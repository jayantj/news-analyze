#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>

"""
Automated tests for corpuses.

"""

import datetime
import logging
import os
import tempfile
import unittest

from glob import glob
import pandas as pd

from news_analyze.utils import HnCorpus as ArticleCorpus


module_path = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


def datapath(fname):
    return os.path.join(module_path, 'data', fname)


def testfile(file_name):
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), file_name)


class TestArticleCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = datapath('articles')
        cls.metadata_file = datapath('metadata.csv')
        cls.corpus = ArticleCorpus(
            data_dir=cls.data_dir,
            metadata_file=cls.metadata_file,
            cache_path=cls.testfile_cls(),
            min_count=1,
            max_df=1.0
        )
        cls.corpus.init_dict()

    def testfile_inst(self):
        return testfile('article_corpus.test.inst')

    @classmethod
    def testfile_cls(cls):
        return testfile('article_corpus.test.cls')

    def test_stream_bow(self):
        articles_bow = list(self.corpus.stream_bow())
        self.assertEqual(len(articles_bow), 2)

        article_id = articles_bow[0][0]
        self.assertEqual(article_id, 10230628)

        word_of_interest = "internet"
        word_index = self.corpus.dictionary.token2id[word_of_interest]
        word_count = [count for word_id, count in articles_bow[0][1] if word_id == word_index][0]
        self.assertEqual(word_count, 6)

    def test_stream_bow_with_max_count(self):
        articles_bow = list(self.corpus.stream_bow(max_count=1))
        self.assertEqual(len(articles_bow), 1)

        article_id = articles_bow[0][0]
        self.assertEqual(article_id, 10230628)

        word_of_interest = "internet"
        word_index = self.corpus.dictionary.token2id[word_of_interest]
        word_count = [count for word_id, count in articles_bow[0][1] if word_id == word_index][0]
        self.assertEqual(word_count, 6)

    def test_getitem(self):
        word_counts = self.corpus[10230628]
        word_of_interest = "internet"
        word_index = self.corpus.dictionary.token2id[word_of_interest]
        word_count = [count for word_id, count in word_counts if word_id == word_index][0]
        self.assertEqual(word_count, 6)

    def test_cache_for_stream_bow(self):
        cache_path = self.testfile_inst()
        corpus = ArticleCorpus(
            self.data_dir,
            metadata_file=self.metadata_file,
            cache_path=cache_path,
            min_count=1,
            max_df=1.0,
        )
        self.assertTrue(corpus.token_cache is None)
        self.assertFalse(os.path.exists(cache_path))

        corpus.init_dict()
        self.assertFalse(corpus.token_cache is None)
        self.assertTrue(os.path.exists(cache_path))

    def test_get_article_text(self):
        article_text = self.corpus.get_article_text(10230628)
        expected_prefix = "The situation is tricky for American tech companies."
        self.assertTrue(article_text.startswith(expected_prefix))
        self.assertEqual(len(article_text), 5416)

        article_text = self.corpus.get_article_text(10230628, max_length=100)
        self.assertEqual(len(article_text), 100)

    def test_get_articles_metadata(self):
        metadata = self.corpus.get_articles_metadata([10230628, 10230685])
        expected_titles = [
            'China Tries to Extract Pledge of Compliance from U.S. Tech Firms',
            'Tsunami alert as Chile hit by powerful earthquake'
        ]
        expected_dates = [
            datetime.date(2015, 9, 16),
            datetime.date(2015, 9, 15)
        ]
        self.assertEqual(list(metadata['title']), expected_titles)
        self.assertEqual(list(metadata['created_date']), expected_dates)

    def test_get_bow_from_text(self):
        article_text = "Uber, Apple, Google, IBM, Uh-uh"
        tokens = ["uber", "apple", "google", "ibm"]
        expected_token_ids = set(self.corpus.dictionary.token2id[token] for token in tokens)
        bow = self.corpus.get_bow_from_text(article_text)
        actual_token_ids = set(token_id for token_id, _ in bow)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def tearDown(self):
        for test_file in glob(self.testfile_inst() + '*'):
            os.unlink(test_file)

    @classmethod
    def tearDownClass(cls):
        for test_file in glob(cls.testfile_cls() + '*'):
            os.unlink(test_file)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
