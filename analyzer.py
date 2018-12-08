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
from nltk.corpus import wordnet


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

def saveData(nFile, data):
  target = open(nFile, 'w')
  target.write(data)
  target.close()
  return target

def createSimilarities(arr):
  tok = sparse.csr_matrix(arr)
  return cosine_similarity(tok)

titles = getCompiledBooks()
datum = []
for title in titles:
  data = openFile(title)
  datum += [data]

books = len(datum)
names = map(lambda x: x['name'].encode(encoding='UTF-8',errors='strict'),datum)

# sub_sim = map(lambda x: -1 if x < .3 else (1 if x > .7 else 0), x['sub'])
# sub_pol = map(lambda x: -1 if x < -.3 else (1 if x > .3 else 0), x['pol'])

sub_sim = createSimilarities(map(lambda x: map(lambda y: -1 if y < .3 else (1 if y > .7 else 0), x['sub']), datum))
sub_pol = createSimilarities(map(lambda x: map(lambda y: -1 if y < -.3 else (1 if y > .3 else 0), x['pol']), datum))

# vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.05, max_df=0.5, stop_words='english')
vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.5, stop_words='english')
#train on all books that aren't the main book

# for dat in datum:
#   if dat['name'] == 'The Divine Comedy by Dante_Dante Alighieri':
#     for word in dat['words']:
#       wrd = word.lower()
#       # print("word: ", wrd)
#       syns = wordnet.synsets(wrd)
#       if len(syns) > 0:
#         for syn in syns:
#           if len(syn.lemmas()) > 1:
#             print('<=============' + wrd + '==============>')
#             for lem in syn.lemmas()[1:]:
#               print("lemma: ", lem.name())

words = map(lambda x: ' '.join(x['words']).encode(encoding='UTF-8',errors='strict'),datum)
X_train = vectorizer.fit_transform(words)
features = vectorizer.get_feature_names()
# words in first book
cosineSim = (X_train * X_train.T).A

definingWords = []
for idx in range(0,books):
  d = pd.Series(X_train.getrow(idx).toarray().flatten(), index = features).sort_values(ascending=False)
  wyrds = {}
  for x in range(0,20):
    wyrds[d.index[x]] = d[x]
  definingWords += [wyrds]

summary = {
  'books' : names,
  'defining_words' : definingWords,
  'word_similarity' : cosineSim.tolist(),
  'polarity' : np.array(sub_pol).tolist(),
  'subjectiveness' : np.array(sub_sim).tolist(),
}

filename = "./analyzed/summary.json"
saveData(filename,json.dumps(summary).encode('utf-8'))









#
#   #instead of this, go through the books, stats and tell the user how objective happy sad it is...
#   subMap = map(lambda x: -1 if x < .3 else (1 if x > .7 else 0), datum[idx]['sub'])
#   polMap = map(lambda x: -1 if x < -.3 else (1 if x > .3 else 0), datum[idx]['pol'])
#   print('<======== subjectivity ==============>')
#   print('Overly Subjective Passages: ', sum(map(lambda x: 1 if x == 1 else 0, subMap)))
#   print('Overly Objective Passages: ', sum(map(lambda x: 1 if x == -1 else 0, subMap)))
#   #interpretSimilar(names, sub_sim, idx)
#   print('<======== polarity ==============>')
#   print('Overly Positive Passages: ', sum(map(lambda x: 1 if x == 1 else 0, polMap)))
#   print('Overly Negative Passages: ', sum(map(lambda x: 1 if x == -1 else 0, polMap)))
#   #interpretSimilar(names, sub_pol, idx)
#
#   # print('<======== mood ==============>')
#   # mood = np.add(sub_sim[idx], sub_pol[idx])
#   # sorted = np.argsort(mood)
#   # for pos in sorted[::-1]:
#   #   if idx != pos and pos < 6:
#   #     print(str(names[pos]) + ': ' + str(mood[pos]))
#   print('<============ done ===========>')
#   print()


# cosineSim = (X_train * X_train.T).A
# print('<==== Similarity Matrix ====>')
# print((X_train * X_train.T).A)


# wordsMinusMain = words[1:]
# print(len(wordsMinusMain))
# print('Training...')
# X_train = vectorizer.fit_transform(wordsMinusMain)
# print("n_samples: %d, n_features: %d" % X_train.shape)
# print()
#
# X_test = vectorizer.transform(words[0])
# print("n_samples: %d, n_features: %d" % X_test.shape)
# print()


#test on the main book. see


#print(analyzeCoSim(subjectiveLines))
#analyzeCoSim(subjectiveLines)
# print('<======= done =========>')
# print(analyzeCoSim(subjectiveLines))
#
# for data in datum:
#   print(data['name'])

# Ways to compare books
# Compare the subjectiveness lines
# Compare the polarity lines
# Compare all the words in one book to all the words in every other book
# Compare all the words in each section of a book to all the words in every other book for intermediate matches
# Tag books by genre and implement learning on the targets

# Eventual tries
# Use a thesaurus (https://wordnet.princeton.edu/) to equate similar words (stage 2)
# Weight verbs more than other POS (stage 2)
# Compare sentence structure similarity throughout the corpus (stage 2)



