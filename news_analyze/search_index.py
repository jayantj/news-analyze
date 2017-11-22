#! /usr/bin/env python

import argparse
import logging
import os
import pickle

from collections import defaultdict
import spacy
from gensim.corpora import Dictionary, MmCorpus


logger = logging.getLogger(__name__)


class SearchIndex(object):
	def __init__(self, preprocess_tokens=False, preprocessor=None):
		self.documents = {}
		self.term_index = defaultdict(set)
		self.nlp = spacy.load('en')
		self.preprocess_tokens = preprocess_tokens
		self.preprocessor = preprocessor

	def index(self, documents):
		for document in documents:
			self.documents[document.id] = document
			tokens = self.tokenize(document.text)
			for token in tokens:
				self.term_index[token].add(document.id)
			if self.preprocess_tokens:
				preprocessed_tokens = self.preprocess(tokens)
				for token in preprocessed_tokens:
					self.term_index[token].add(document.id)

	def tokenize(self, document_text):
		tokens = self.nlp.tokenizer(document_text)	
		return [token.orth_ for token in tokens]

	def preprocess(self, document_tokens):
		if self.preprocessor:
			return [self.preprocessor(token) for token in document_tokens]
		else:
			return [token.lower() for token in document_tokens]

	def query(self, query_term):
		return self.term_index[query_term]
