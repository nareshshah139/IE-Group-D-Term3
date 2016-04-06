# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:53:43 2016

@author: carmennicolasbaixauli
"""

import nltk

#cadena = 'AS THE PALATOFIRM EXPANDS SAND IS SUED BY OTHER OPRIDUYETS THERE MAY BE A FICUS'

cadena=input("Insert a sentence to check spelling mistakes:\n")

english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def spelling_mistakes(text):
    mistakesList=list()
    words=text.split( )
    for w in words: 
        if w.lower() not in english_vocab:
            mistakesList.append(w)
    return mistakesList

listaErrores=spelling_mistakes(cadena)
print("This is the list of words with spelling mistakes: " )
print(listaErrores)