#sudo python -m nltk.downloader all
from __future__ import print_function

import os
import io
import math
import json
import sys
import requests
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

def getOnlineMeta(search):
    try:
        api_key = ""
        params = {'q' : search, 'key': api_key}
        text = requests.get('https://www.googleapis.com/books/v1/volumes', params=params).text
        data = json.loads(text)
        meta = {}
        cats = ['description','publishedDate','title','subtitle','pageCount','maturityRating', 'infoLink',
                'authors','categories','previewLink','imageLinks','averageRating','ratingsCount','searchInfo']

        for k,v in data["items"][0].items():
            if k == "volumeInfo":
                for k2,v2 in v.items():
                    if k2 in cats:
                        meta[k2] = v2
            if k in cats:
                meta[k] = v

        return meta
    except Exception as e:
        print(e)
        return None

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
    my_stops = ['in','and','i','if','to','as']
    stop_words = list(text.ENGLISH_STOP_WORDS.union(my_stops))
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
        self.populateoMeta()

    def populateoMeta(self):
        trys = [self.name.replace("_"," "), self.name.split("_")[0], self.name.split("_")[1]]
        for tri in trys:
            search =  getOnlineMeta(tri)
            if search:
                self.meta['oMeta'] = search
                return
        self.meta['oMeta'] = None

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
            'oMeta': self.meta['oMeta']
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

    def populateContent(self):
        stoppy_words = getStopWords()
        words = []
        for tag in self.blob.tags:
            if not 'NNP' in tag[1] and not 'VB' in tag[1] and tag[0].isalpha() \
              and tag[0].lower() not in stoppy_words and len(tag[0]) > 2:
                words += [tag[0].lemmatize().lower()]
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