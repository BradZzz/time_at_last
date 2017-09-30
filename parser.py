#sudo python -m nltk.downloader all
from __future__ import print_function

import os
import io
import math
import json
import sys
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from collections import Counter

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
import numpy as np


def get_files(path):
    for (dirpath, _, filenames) in os.walk(path):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

def getMissingBooks():
    files = []
    for filename in get_files('./books/'):
        if '.txt' in filename:
            files += [filename]
    return files

def getBookName(book):
    return os.path.basename(book)

def getStopWords():
    from sklearn.feature_extraction import text
    my_stops = ['said','just','did','going','wa','let']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_stops)
    return stop_words

def saveData(nFile, data):
    target = open(nFile, 'w')
    target.write(data)
    target.close()
    return target

class analyzeShit:
    def __init__(self, name, blob, sects):
        self.name = name
        self.blob = blob
        self.meta = { 'words': [], 'pol' : [], 'sub' : [], 'tags' : [] }
        #How many times you want to analyze divisions in the data
        self.sects = sects
        self.populateContent()
        self.populateTags()
        self.populateMood()

    def populateTags(self):
        counts = Counter([ x[1] for x in self.blob.tags])
        self.meta['tags'] = counts

    def returnAnalyzedBook(self):
        return {
            'name': self.name.decode('utf-8'),
            'words': self.meta['words'],
            'pol': self.meta['pol'],
            'sub': self.meta['sub'],
            'tags': self.meta['tags'],
        }

    #populate polarity and subjectivity
    def populateMood(self):
        words = self.meta['words']
        parts = self.sects
        size = len(words) / parts
        for piece in range(0,parts):
            offset = (piece * size)
            section = words[offset:size + offset]
            blob = TextBlob(' '.join(section))
            self.meta['sub'] += [blob.sentiment.subjectivity]
            self.meta['pol'] += [blob.sentiment.polarity]

            # print(blob.sentiment)
        # sMax = int(float(len(self.blob.sentences)) / float(self.sects))
        # print('sMax: ', sMax)
        # sCount = 0
        # sParagraph = ""
        # sPolarity = []
        # sSubjectivity = []
        # sHold = []
        # for sentence in blob.sentences:
        #     sSubjectivity += [sentence.sentiment.subjectivity]
        #     sPolarity += [sentence.sentiment.polarity]
        #     sParagraph += '{}'.format(sentence).decode('utf-8').replace('\n','') + ' '
        #     sCount += 1
        #     if sCount == sMax:
        #         sHold += [sParagraph]
        #         s = np.array(sSubjectivity)
        #         p = np.array(sPolarity)
        #         self.meta['sub'] += [np.mean(s)]
        #         self.meta['pol'] += [np.mean(p)]
        #         sCount = 0
        #         sParagraph = ""
        #         sPolarity = []
        #         sSubjectivity = []
        #     if len(self.meta['sub']) == self.sects or len(self.meta['pol']) == self.sects:
        #         break

    #populate content
    #\*\*\* .* OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*
    #\*\*\* END OF THIS PROJECT GUTENBERG EBOOK * \*\*\*
    #re.split("\s+", str1)
    def populateContent(self):
        stoppy_words = getStopWords()
        words = []
        for idx2, sentence in enumerate(self.blob.sentences):
            for x in sentence.words:
                if x.isalpha() and x.lower() not in stoppy_words:
                    words += [x.lemmatize()]
        self.meta['words'] = words

# Marshal books in folder
cache = {}
textArr = []
for book in getMissingBooks():
    with io.open(book, 'r', encoding='utf-8') as f:
        text = f.read()
    title = getBookName(book).replace('.txt','')
    parts = re.split("\*\*\* .* OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    if len(parts) < 2:
        text = parts[0]
    else:
        text = parts[1]
    cache[title] = {
        'path': book,
        'blob': text
    }

print('building...')
for key, value in cache.items():
    blob = TextBlob(value['blob'])
    book = analyzeShit(key, blob, 100).returnAnalyzedBook()
    print('parsed: ' + book['name'])
    filename = "./parsed/" + book['name'] + "(analysis).json"
    saveData(filename,json.dumps(book).encode('utf-8'))

# wordnet and POS

    # sMax = len(blob.sentences) / 100
    #
    # sMood = { 'pol' : [], 'sub' : [] }
    # sCount = 0
    # sParagraph = ""
    # sPolarity = []
    # sSubjectivity = []
    # sHold = []
    # for sentence in blob.sentences:
    #     sSubjectivity += [sentence.sentiment.subjectivity]
    #     sPolarity += [sentence.sentiment.polarity]
    #     sParagraph += '{}'.format(sentence).decode('utf-8').replace('\n','') + ' '
    #     sCount += 1
    #     if sCount == sMax:
    #         sHold += [sParagraph]
    #         s = np.array(sSubjectivity)
    #         p = np.array(sPolarity)
    #         sMood['sub'] += [np.mean(s)]
    #         sMood['pol'] += [np.mean(p)]
    #         sCount = 0
    #         sParagraph = ""
    #         sPolarity = []
    #         sSubjectivity = []
    #     if 'END OF THIS PROJECT GUTENBERG EBOOK' in sentence:
    #         print('done')
    #         break
    #
    # books += [sHold]
    # subjectivity => How subjective / objective the author is being
    # polarity => How positive / neutral / negative an author is being

#     print(plots)
#     plt.subplot(6,2,plots)
#     minmax_scaler = preprocessing.MinMaxScaler()
#
#     sub = minmax_scaler.fit_transform(np.array(sMood['sub']).reshape(-1, 1))
#     pol = minmax_scaler.fit_transform(np.array(sMood['pol']).reshape(-1, 1))
#
#     plt.plot(sub, label="subjectivity")
#     plt.plot(pol, label="polarity")
#     plt.legend(bbox_to_anchor=(1,-.5), loc="lower right", ncol=2)
#     plt.title(key.decode('utf-8'))
#     plt.axis('off')
#     plots += 1
#     # break
#
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
# plt.show()

    # print key
    # # Analyze book blobs
    # bloblist = []
    # for key, value in cache.items():
    #     print key
    #     print value
    #     blob = TextBlob(value['blob'])
    #     bloblist += [blob]
    #     #Go through each word and lemmatize it
    #     #Go through each sentence


# true_k = 4
# vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', use_idf=True)
# X = vectorizer.fit_transform(reduce((lambda x, y: x + y), books))
#
# print("n_samples: %d, n_features: %d" % X.shape)
# print()
#
# km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=True)
# print("Clustering sparse data with %s" % km)
# t0 = time()
# km.fit(X)
# print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Top terms per cluster:")
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i, end='')
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind], end='')
#         print()


# def tf(word, blob):
#     return blob.words.count(word) / len(blob.words)
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
# def idf(word, bloblist):
#     return math.log(float(len(bloblist)) / float((1 + n_containing(word, bloblist))))
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)


# Analyze book blobs
# bloblist = []
# for key, value in cache.items():
#     print key
#     print value
#     blob = TextBlob(value['blob'])
#     bloblist += [blob]
#     #Go through each word and lemmatize it
#     #Go through each sentence

# for i, blob in enumerate(bloblist):
#     print("Top words in document {}".format(i + 1))
#     scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:3]:
#         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
