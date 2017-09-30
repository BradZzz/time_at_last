#sudo python -m nltk.downloader all
from __future__ import print_function

import os
import io
import math
import json
import sys
from textblob import TextBlob
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from textblob import TextBlob

def get_files(path):
  for (dirpath, _, filenames) in os.walk(path):
    for filename in filenames:
      yield os.path.join(dirpath, filename)

def getCompiledBooks():
  files = []
  for filename in get_files('./parsed/'):
    if '.json' in filename:
      files += [filename]
  return files

def getBookName(book):
  return os.path.basename(book)

def openFile(fName):
  with open(fName) as dFile:
    data = json.load(dFile)
  return data

def analyzeCoSim(bag):
  sim = []
  for x, matx in enumerate(bag):
    cos_sim = cosine_similarity(matx, bag)
    sim += cos_sim.tolist()
  return sim

def interpretSimilar(names, similarities, idx):
  matrix = similarities[idx]
  sorted = np.argsort(matrix)
  for pos in sorted[::-1]:
    if idx != pos and pos < 6:
      print(str(names[pos]) + ': ' + str(matrix[pos]))


titles = getCompiledBooks()
datum = []
for title in titles:
  data = openFile(title)
  datum += [data]

def createSimilarities(arr):
  tok = sparse.csr_matrix(arr)
  return cosine_similarity(tok)

books = len(datum)
names = map(lambda x: x['name'].encode(encoding='UTF-8',errors='strict'),datum)

# sub_sim = createSimilarities(map(lambda x: x['sub'], datum))
# sub_pol = createSimilarities(map(lambda x: x['pol'], datum))
#
# # vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.05, max_df=0.5, stop_words='english')
# vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.6, stop_words='english')
# #train on all books that aren't the main book
words = map(lambda x: ' '.join(x['words']).encode(encoding='UTF-8',errors='strict'),datum)
parts = 20
size = len(words[0]) / parts
print(size)
for piece in range(0,parts):
  offset = (piece * size)
  print(offset)
  section = words[0][offset:size + offset]
  blob = TextBlob(section)
  print(blob.sentiment)

# X_train = vectorizer.fit_transform(words)
# features = vectorizer.get_feature_names()
# # words in first book
# cosineSim = (X_train * X_train.T).A
# for idx in range(0,books):
#   print()
#   print('<========== Book: ' + str(idx) + ' ================>')
#   print("Book: ", names[idx])
#   d = pd.Series(X_train.getrow(idx).toarray().flatten(), index = features).sort_values(ascending=False)
#   print('Unique words:')
#   print(d[:20])
#   sorted = np.argsort(cosineSim[idx])
#   print('<==== Writing Style ====>')
#   for pos in sorted[::-1]:
#     if idx != pos and pos < 6 and cosineSim[idx][pos] > .08:
#       print(str(names[pos]) + ': ' + str(cosineSim[idx][pos]))
#   print('<======== subjectivity ==============>')
#   interpretSimilar(names, sub_sim, idx)
#   print('<======== polarity ==============>')
#   interpretSimilar(names, sub_pol, idx)
#
#   # print('<======== mood ==============>')
#   # mood = np.add(sub_sim[idx], sub_pol[idx])
#   # sorted = np.argsort(mood)
#   # for pos in sorted[::-1]:
#   #   if idx != pos and pos < 6:
#   #     print(str(names[pos]) + ': ' + str(mood[pos]))
#   print('<============ done ===========>')
#   print()