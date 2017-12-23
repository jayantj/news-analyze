#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>

"""
Automated tests for corpuses.

"""

import logging
import os
import unittest

from news_analyze.utils import HnCorpus as ArticleCorpus


module_path = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


def datapath(fname):
    return os.path.join(module_path, 'data', fname)


class TestArticleCorpus(unittest.TestCase):

    def test_stream_articles_bow(self):
        data_dir = datapath('articles')
        corpus = ArticleCorpus(data_dir, metadata={}, min_count=1, max_df=1.0)
        articles_bow = list(corpus.stream_bow())
        self.assertEqual(len(articles_bow), 2)

        article_id = articles_bow[0][0]
        self.assertEqual(article_id, 10230628)

        word_of_interest = "internet"
        word_index = corpus.dictionary.token2id[word_of_interest]
        word_count = [count for word_id, count in articles_bow[0][1] if word_id == word_index][0]
        self.assertEqual(word_count, 6)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
