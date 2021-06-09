__author__ = 'Allen'

import os
import sys
import json
import re

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from summa import summarizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class PrepareDocuments(object):
    def __init__(self):
        self.final = []  # store documents after removing keywords and \n
        self.output = {}  # store documents of length that are suitable for textrank
        self.index = 0  # index for output_dict
        self.length = []  # store the lengths of the documents
        self.removed_keyword_count = 0  # number of documents that successfully removed keywords\
        self.docs = []

        with open(sys.path[0] + '/sentence_embedding_data/1.json', 'r', encoding='utf8') as f:
            content1 = json.load(f, strict=False)
        for doc in content1:
            self.docs.append(doc)
        with open(sys.path[0] + '/sentence_embedding_data/2.json', 'r', encoding='utf8') as f:
            content2 = json.load(f, strict=False)
        for doc in content2:
            self.docs.append(doc)
        for line in open(sys.path[0] + '/sentence_embedding_data/3.json', 'r', encoding='utf8'):
            self.docs.append(json.loads(line))

        count = 0
        for doc in self.docs:
            self.final.append(self.remove_keywords(doc['content']))
            count += 1
            if count % 1000 == 0:
                print('processed %d number of good documents' % count)

        self.restructure_document(self.final)
        with open(sys.path[0] + '/sentence_embedding_data/documents_for_textrank.json', 'a', encoding='utf8') as f:
            f.write(json.dumps(self.output))
        print('output %d number of documents for textrank' % self.index)  # 136458

        '''for examining the length of the documents and decide a number to cut'''
        # print(stats.describe(self.length))
        # sns.kdeplot(self.length, bw=2, label='content_length')
        # plt.legend()
        # plt.show()

    def remove_keywords(self, content):
        content = re.sub(r'\n+\s*', '', content)  # remove \n and blanks
        try:
            keyword_start = re.search(r'\u2014', content).span()[0]  # start of the keyword
            keyword_end = content[keyword_start:keyword_start + 450].rfind('\u2014') + keyword_start  # last keyword
            content = content[keyword_end:]
            self.removed_keyword_count += 1
        except:
            pass
        self.length.append(len(content))
        return content

    def restructure_document(self, docs):
        for doc in docs:
            if len(doc) > 2000:  # short documents are discarded
                if len(doc) < 10000:
                    self.output[self.index] = doc
                    self.index += 1
                else:
                    parts = [doc[i:i + 10000] for i in range(0, len(doc), 10000)]
                    if len(parts[-1]) < 2000:  # if last document is short, add back
                        last = parts[-2] + parts[-1]
                        parts = parts[0:-2]
                        parts.append(last)
                    for doc in parts:
                        self.output[self.index] = doc
                        self.index += 1
        # end for
    # end def
# end class


class ExtractSentenceFromContent(object):
    def __init__(self):
        self.summarizer = summarizer
        self.output = []        # store all the key sentences for every doc
        # with open(sys.path[0] + '/sentence_embedding_data/documents_for_textrank.json', 'r', encoding='utf8') as f:
        #     data = json.load(f, strict=False)

        for line in open(sys.path[0] + '/sentence_embedding_data/sample.json', 'r', encoding='utf8'):
            data = json.loads(line)

        self.content = list(data.values())
        for content in self.content:
            self.output.append(self.extract_key_sentence(content))

    def extract_key_sentence(self, content):
        return self.summarizer.summarize(content, ratio=0.2, split=True)

    def output_sentences(self):
        return self.output
# end class

class UniversalSentenceEncoderEmbedding(object):
    def __init__(self, sentences):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3", trainable=True)
        self.sentences = sentences


        self.create_embedding()

    def create_embedding(self):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(self.embed(self.sentences))
        for i, message_embedding in enumerate(message_embeddings):
            print(message_embedding)


def main():
    PrepareDocuments()      # convert documents to suitable lengths
    sentences = ExtractSentenceFromContent().output_sentences()     # textrank to extract key sentences
    UniversalSentenceEncoderEmbedding(sentences)        # embedding from key sentences

if __name__ == '__main__':
    main()
