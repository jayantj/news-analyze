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

from news_analyze.utils import ArticleCorpus
from news_analyze.lda import HnLdaModel as ArticleLdaModel, HnLdaMalletModel as ArticleLdaMalletModel


module_path = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


def datapath(fname):
    return os.path.join(module_path, 'data', fname)


def testfile(file_name):
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), file_name)


class TestArticleLdaModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = datapath('articles')
        metadata_file = datapath('metadata.csv')
        corpus = ArticleCorpus(
            data_dir=data_dir,
            metadata_file=metadata_file,
            cache_path=cls.testfile_cls(),
            min_count=1,
            max_df=1.0
        )
        model = ArticleLdaModel(corpus=corpus, num_topics=10)
        model.train()
        cls.corpus = corpus
        cls.model = model

    def test_training_sanity(self):
        self.assertEqual(self.model.topic_word_matrix.shape, (10, len(self.corpus.dictionary)))
        self.assertEqual(self.model.article_topic_matrix.shape, (2, 10))

    def testfile_inst(self):
        return testfile('article_corpus.test.inst')

    @classmethod
    def testfile_cls(cls):
        return testfile('article_corpus.test.cls')

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
