import multiprocessing
from multiprocessing import Pool
import time

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

def download_article(article):
    id_, url = article
    article = Article(url)
    article.download()
    return (id_, article)

def run_pool(poolsize, input_filename, downloaded_filename):
    id_url_list = []
    downloaded_ids = set(open(downloaded_filename, 'r').readlines())
    with open(input_filename, 'r') as f:
        for line in f:
            values = line.split()
            id_, url = int(values[0]), values[1]
            if id_ in downloaded_ids:
                continue
            id_url_list.append((int(values[0]), values[1]))
    workers = Pool(processes=poolsize)
    for i, (article_id, article) in enumerate(workers.imap_unordered(download_article, id_url_list, chunksize=10), start=1):
        yield article_id, article


if __name__ == "__main__":
    poolsize, input_filename, downloaded_filename = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    print_every = 50
    start_time = time.time()
    last_time = start_time
    for i, (article_id, article) in enumerate(run_pool(poolsize, input_filename, downloaded_filename), start=1):
        with open(os.path.join(DATA_DIR, str(article_id) + '.pkl'), 'wb') as f:
            pickle.dump(article, f)
        with open(downloaded_filename, 'a') as f:
            f.write('%d\n' % article_id)
        if not (i % print_every):
            curr_time = time.time()
            batch_speed = print_every / (curr_time - last_time)
            total_speed = i / (curr_time - start_time)
            print('Downloaded %d articles, batch speed: %.2f articles/s, total sped: %.2f articles/s' % (i, batch_speed, total_speed))
            last_time = time.time()