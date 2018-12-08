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
import operator

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

def createSimilarities(arr):
  tok = sparse.csr_matrix(arr)
  return cosine_similarity(tok)

def printFromList(title, names, list, scale):
  # if scale:
  #   list = np.array(list)
  #   list = (list-min(list))/(max(list)-min(list))
  sorted = np.argsort(list)[::-1]
  print('<======================= Begin ' + title + ' =====================>')
  for sim in range(0,6):
    if sim != 0 and list[sorted[sim]] > 0:
      print(str(names[sorted[sim]]) + ': ' + str(list[sorted[sim]]))
  print('<======================= End ' + title + ' =====================>')

titles = getCompiledBooks()
datum = []
for title in titles:
  data = openFile(title)
  datum += [data]

summary = openFile("./analyzed/summary.json")
names = map(lambda x: x['name'].encode(encoding='UTF-8',errors='strict'),datum)

for book in datum:
  idx = summary['books'].index(book['name'])
  print('|____________________________________________________________________|')
  print("Book: " + book['name'].encode('ascii', 'ignore'))
  print("Meta: " + str(book['oMeta']))
  word_similarity = summary['word_similarity'][idx]
  printFromList('word similarity', names, word_similarity, False)
  polarity = summary['polarity'][idx]
  printFromList('polarity', names, polarity, True)
  subjectiveness = summary['subjectiveness'][idx]
  printFromList('subjectiveness', names, subjectiveness, True)
  total = 0
  newTags = {}
  print("<== Tags ==>")
  for key, val in book['tags'].items():
    storedKey = key
    if 'NN' in key:
      storedKey = 'NN'
    elif 'VB' in key:
      storedKey = 'VB'
    elif 'PRP' in key:
      storedKey = 'PRP'
    elif 'JJ' in key:
      storedKey = 'JJ'
    elif 'RB' in key:
      storedKey = 'RB'
    elif 'WP' in key:
      storedKey = 'WP'

    if storedKey not in newTags:
      newTags[storedKey] = 0
    newTags[storedKey] += val
    total+= val
  newTags = sorted(newTags.items(), key=operator.itemgetter(1))[::-1]
  for tag in newTags:
    formVal = "{0:.2f}".format((float(tag[1]) / float(total)) * 100)
    print("\t" + tag[0] + ": " + formVal + "%")
  print('|____________________________________________________________________|')



# books = len(datum)
# names = map(lambda x: x['name'].encode(encoding='UTF-8',errors='strict'),datum)
#
# sub_sim = createSimilarities(map(lambda x: x['sub'], datum))
# sub_pol = createSimilarities(map(lambda x: x['pol'], datum))
#
# # vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=0.05, max_df=0.5, stop_words='english')
# vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.5, stop_words='english')
# #train on all books that aren't the main book
# words = map(lambda x: ' '.join(x['words']).encode(encoding='UTF-8',errors='strict'),datum)
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
#   for similar in range(0,6):
#     pos = sorted[::-1][similar]
#     # if idx != pos and pos < 6 and cosineSim[idx][pos] > .075:
#     if idx != pos:
#       print(str(names[pos]) + ': ' + str(cosineSim[idx][pos]))
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



