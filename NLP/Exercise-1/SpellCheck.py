# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:02:32 2016

@author: mehdinassih
"""

import nltk

text = "AS THE PALATOFIRM EXPANDS SAND IS SUED BY OTHER OPRIDUYETS THERE MAY BE A FICUS"
lowertext = nltk.word_tokenize(text.lower())

for word in lowertext:
        if not wordnet.synsets(word) and word not in stopwords.words('english'):
            print(word)


