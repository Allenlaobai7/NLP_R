import os, re, sys
import argparse
import pandas as pd
import Levenshtein as lev
from fasttext import load_model
from sklearn.metrics.pairwise import cosine_similarity

hashtag_sim_model = load_model(args.embedding_path)
r = hashtag_sim_model.get_nearest_neighbors(keyword, k=100)
for i in r:
    if lev.distance(keyword, i[0]) / max(len(keyword), len(i[0])) <= 0.4:
        print(1)
