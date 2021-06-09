#! /usr/bin/env python
from __future__ import division

import sys
import os
import argparse
import logging
import io
import math
import pandas as pd

from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType, FloatType, MapType, IntegerType, StructType, StructField
from pyspark.sql.functions import udf, collect_list, col

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def flatten_list(col):
    return [item for sublist in col for item in sublist]

def process(args):
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .config('hive.exec.dynamic.partition', 'true') \
        .config('hive.exec.dynamic.partition.mode', 'nonstrict') \
        .config('spark.io.compression.codec', 'snappy') \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    DATA_TABLE_NAME = args.input_path
    INSERT_TABLE_NAME = args.output_path

    query = """
SELECT * from {}
""".format(DATA_TABLE_NAME)

    df = spark.sql(query)
    df.show(20)
    logging.info('{} rows in total'.format(df.count()))

    # combine all rows to list of list
    words = [row['words'] for row in df.select('words').collect()]
    words_lists = list(map(lambda x: list(set(x.split(','))), words))
    N = len(words_lists) # sample size

    # creates a Counter for all words
    flattened = [word for item in words_lists for word in item]
    logging.info('{} words in total'.format(len(flattened)))
    logging.info('{} unique words in total'.format(len(list(set(flattened)))))
    word_count = Counter(flattened)

    # remove low_occuring words for calculating co-occurrence:
    def remove_low_occurring_types(type_list):
        return [x for x in type_list if word_count[x] > 9]    # keep only if occurs at least 10 times
    good_words_lists = list(map(remove_low_occurring_types, words_lists))
    good_words_lists = list(
        filter(lambda x: len(x) > 1, good_words_lists))  # select only those with more than 1 word
    logging.info('{} rows of words_lists to process'.format(len(list(good_words_lists))))

    # find co-occurrence for words. P(x,y)
    co_occurence_dict = {}
    for l in good_words_lists:
        for i, type in enumerate(l):
            for j, type2 in enumerate(l[i + 1:]):
                if type > type2:
                    try:  # if alr exist
                        co_occurence_dict[type][type2] += 1
                    except:
                        try:  # if exists counts for other types
                            co_occurence_dict[type][type2] = 1
                        except:  # if does not exist
                            co_occurence_dict[type] = {type2: 1}
                else:
                    try:
                        co_occurence_dict[type2][type] += 1
                    except:
                        try:  # if exists counts for other types
                            co_occurence_dict[type2][type] = 1
                        except:  # if does not exist
                            co_occurence_dict[type2] = {type: 1}

    def calculate_pmi(key, key2, co_occur_count):
        # n(x, y), n(x), n(y)
        n_x = word_count[key]
        n_y = word_count[key2]
        pmi = max(round(math.log((co_occur_count * N) / (n_x * n_y), 2), 3), 0.0)  # (n(x,y)* N) /(n(x)*n(y))
        return pmi

    pmi_res = []
    for key, value in co_occurence_dict.items():
        for key2, co_occur_count in value.items():
            pmi = calculate_pmi(key, key2, co_occur_count)
            if pmi != 0.0:
                pmi_res.append([key, key2, pmi, co_occur_count, word_count[key], word_count[key2]])

    df_pd = pd.DataFrame.from_records(pmi_res,
                                   columns=['word1', 'word2', 'pmi_score', 'co_occur_count', 'count_word1', 'count_word2'])

    Schema1 = StructType([StructField("word1", StringType(), True) \
                              , StructField("word2", StringType(), True) \
                              , StructField("pmi_score", FloatType(), True) \
                             , StructField("co_occur_count", IntegerType(), True)
                             , StructField("count_word1", IntegerType(), True)
                              , StructField("count_word2", IntegerType(), True)])
    df = spark.createDataFrame(df_pd, schema=Schema1)

    df = df.sort("co_occur_count", ascending=False)
    df.show(100)

    df.write.mode('overwrite').insertInto(INSERT_TABLE_NAME)
    logging.info('Saving rows to {}'.format(INSERT_TABLE_NAME))
    spark.stop()

def main():
    parser = argparse.ArgumentParser(description='PMI study.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    process(args)
    return 0

if __name__ == '__main__':
    sys.exit(main())
