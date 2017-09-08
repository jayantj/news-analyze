import argparse
import multiprocessing
import time
import logging
import os
import pickle
import sys

import newspaper
from newspaper import Article


logger = logging.getLogger(__name__)

DATA_DIR = 'data/articles'


def datapath(filename):
    return os.path.join(DATA_DIR, filename)

def article_file_iterator(articles_dir, parsed_ids, output_dir, min_article_length):
    for i, filename in enumerate(os.listdir(articles_dir), start=1):
        article_id = os.path.splitext(filename)[0]
        if int(article_id) in parsed_ids:
            continue
        full_filename = os.path.join(articles_dir, filename)
        yield full_filename, output_dir, min_article_length

def parse_articles(articles_dir, output_dir, parsed_ids_file, min_article_length, num_processes=4):
    try:
        already_parsed_ids = set(int(l) for l in open(parsed_ids_file, 'rb'))
    except FileNotFoundError:
        print('File with already parsed ids absent, parsing all articles again')
        already_parsed_ids = set()

    iterator = article_file_iterator(articles_dir, already_parsed_ids, output_dir, min_article_length)
    workers = multiprocessing.Pool(processes=num_processes)
    save_every = 20
    newly_parsed_ids = []
    parsed_count = 0
    start_time = time.time()
    for article_id in workers.imap_unordered(parse_article, iterator):
        if article_id is None:
            continue
        parsed_count += 1
        newly_parsed_ids.append(article_id)
        if len(newly_parsed_ids) == save_every:
            time_taken = time.time() - start_time
            print('Parsed %d articles in %.2f seconds, %.2f articles/sec' % (parsed_count, time_taken, parsed_count/time_taken))
            with open(parsed_ids_file, 'a') as f:
                f.write('\n'.join(newly_parsed_ids) + '\n')
            newly_parsed_ids = []

def parse_article(args):
    filename, output_dir, min_article_length = args
    article = pickle.load(open(filename, 'rb'))
    article.config.fetch_images = False
    article_id = os.path.splitext(os.path.basename(filename))[0]
    try:
        article.parse()
    except newspaper.article.ArticleException as e:
        logger.info('Skipping article %s, error %s', filename, e)
        return
    if len(article.text) < min_article_length:
        return None
    with open(os.path.join(output_dir, article_id + '.txt'), 'w') as f:
        f.write(article.text)
    return article_id


# if __name__ == "__main__":
#     poolsize, input_filename, downloaded_filename = int(sys.argv[1]), sys.argv[2], sys.argv[3]
#     print_every = 50
#     start_time = time.time()
#     last_time = start_time
#     for i, (article_id, article) in enumerate(run_pool(poolsize, input_filename, downloaded_filename), start=1):
#         with open(os.path.join(DATA_DIR, str(article_id) + '.pkl'), 'wb') as f:
#             pickle.dump(article, f)
#         with open(downloaded_filename, 'a') as f:
#             f.write('%d\n' % article_id)
#         if not (i % print_every):
#             curr_time = time.time()
#             batch_speed = print_every / (curr_time - last_time)
#             total_speed = i / (curr_time - start_time)
#             print('Downloaded %d articles, batch speed: %.2f articles/s, total sped: %.2f articles/s' % (i, batch_speed, total_speed))
#             last_time = time.time()

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-p', '--parsed-ids', required=True)
    parser.add_argument('-n', '--num-processes', type=int)
    parser.add_argument('-l', '--min-article-length', type=int)

    args = parser.parse_args()

    args.input_dir = datapath(args.input_dir)
    args.output_dir = datapath(args.output_dir)
    args.parsed_ids = datapath(args.parsed_ids)
    # try:
    parse_articles(args.input_dir, args.output_dir, args.parsed_ids, args.min_article_length, args.num_processes)
    # except Exception as e:
    #     import pdb
    #     print(e)
    #     pdb.set_trace()
