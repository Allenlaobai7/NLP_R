import os, re, sys
import argparse
import logging
from collections import Counter

import pandas as pd
import numpy as np
import base64
import hnswlib
import yaml
import psutil

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

CHUNK_SIZE = 50000

def build_large_index_folder(process, id_fea_folder, all_ids, args):
    for root, dir, path in os.walk(id_fea_folder):
        paths = [os.path.join(id_fea_folder, p) for p in path]
    logging.info('total_id_cnt: {}'.format(len(all_ids)))

    index_vemb = hnswlib.Index(space='cosine', dim=args.dim)
    index_vemb.init_index(max_elements=len(all_ids), ef_construction=200, M=16)  # M=16
    for path in paths:
        for id_fea in pd.read_csv(path, sep="\t", header=None, names=['id', 'emb'], dtype='str', chunksize=CHUNK_SIZE):
            id_fea = id_fea[id_fea['emb']!=''].dropna()
            id_fea['id'] = id_fea['id'].astype(int)
            id_fea = id_fea[id_fea['id'].isin(all_ids)]
            logging.info('new chunk: {0}, {1}'.format(path, process.memory_info().rss))  # in bytes
            embs = [np.frombuffer(base64.b64decode(str(i)[2:-1]), np.float32) for i in id_fea['emb'].tolist()]
            if len(embs[0]) != args.dim:
                raise RuntimeError('wrong input dim! input file: {}; input dim: {}; required dim: {}'.format(path,
                                                                                         len(embs[0]), args.dim))
            index_vemb.add_items(embs, id_fea['id'].tolist())  # id as index
            logging.info('added new ids: {0}, {1}'.format(len(id_fea), process.memory_info().rss))  # in bytes
    index_vemb.save_index(args.model_path)
    logging.info('saved model to {}'.format(args.model_path))

def process(args):
    process = psutil.Process(os.getpid())
    logging.info("{}".format(process.memory_info().rss))  # in bytes

    # load ids
    all_ids = pd.read_csv(args.id_path, usecols=['id'])
    all_ids['id'] = all_ids['id'].astype(int)
    all_ids = all_ids['id'].tolist()

    build_large_index_folder(process, args.id_fea_path, all_ids, args)

# end def

def main():
    parser = argparse.ArgumentParser(description='recall')
    parser.add_argument('id_path', type=str, default='data/{REGION}/1.csv', help='feature file')
    parser.add_argument('id_fea_path', type=str, default='data/{REGION}/.._fea.txt', help='feature file')
    parser.add_argument('model_path', type=str, default='data/{REGION}/...bin', help='output model path')
    parser.add_argument('--dim', type=int, default=5120)
    args = parser.parse_args()
    process(args)

if __name__ == '__main__':
    sys.exit(main())