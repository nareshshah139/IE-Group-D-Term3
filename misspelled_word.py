# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:55:59 2016

@author: rahulmehra
"""

from nltk.corpus import brown

text = "AS THE PALATOFIRM EXPANDS SAND IS SUED BY OTHER OPRIDUYETS THERE MAY BE A FICUS"
text = text.split(sep = " ",maxsplit = -1)

text_vocab = set(w.lower() for w in text if w.isalpha())
english_vocab = set(w.lower() for w in brown.words(categories="reviews"))
unusual = text_vocab - english_vocab
print(list(unusual))