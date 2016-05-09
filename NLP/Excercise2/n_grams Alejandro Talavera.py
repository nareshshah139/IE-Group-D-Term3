# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:58:30 2016

@author: AleTalavera
"""

'''

from nltk.util import ngrams
from nltk.util import bigrams
from nltk.util import trigrams

list(ngrams[1,2,3,4,5], 3) #al porner un 3 me devuelver los trigrams

list(bigrams([1,2,3,4,5]))

list(trigrams([1,2,3,4,5]))


from nltk.corpus import brown
from nltk import word_tokenize
from nltk.util import ngrams

sentence = brown.words()
string = str(sentence)
tokens = word_tokenize(string)

n = 3

ngrams = ngrams(tokens.split(), n)

for grams in ngrams:
    print(grams)


from nltk.util import ngrams

sentence = "La casa de Perico es alta y bonita"
n = 4
ngrams = ngrams(sentence.split(), n)
for grams in ngrams:
    print(grams)

'''

from nltk.corpus import brown
from nltk import word_tokenize
from nltk.util import ngrams
from nltk import FreqDist

text = brown.words()
        
ngrams = ngrams(text, 3)
#for grams in ngrams:
    #print(grams)

fdist = FreqDist(text)
print(fdist)
