import logging
import os
import pickle
import sys

from newspaper import Article

DATA_DIR = 'data/articles'

def download(url):
	article = Article(url)
	article.download()
	return article


if __name__ == "__main__":
	article_id, article_url = sys.argv[1].split()
	article = download(article_url)
	with open(os.path.join(DATA_DIR, article_id + '.pkl'), 'wb') as f:
		pickle.dump(article, f)
	print('Downloaded article %s successfully' % article_id)
	