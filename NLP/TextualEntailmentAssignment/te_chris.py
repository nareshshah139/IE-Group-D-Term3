# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:57:50 2016

@author: rahulmehra
"""
import nltk
from nltk.corpus import rte
import pandas as pd
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.util import ngrams


def lemmatize(word):
    lemma = nltk.corpus.wordnet.morphy(word, pos = nltk.corpus.wordnet.VERB)
    if lemma is not None:
        return lemma
    return word


stop = stopwords.words('english')
# parse the XML
print("parsing the XML")
rte_te = rte.pairs(['/Users/cbrandenburg/documents/ie/courses/term3/nlp/textualentailmentdata.xml'])
text_ls = []
hyp_ls = []
val_ls = []
text_tokens_ls = []
h_tokens_ls = []
text_punc_ls=[]
for element in range(len(rte_te)):
    text_ls.append(rte_te[element].text)
    hyp_ls.append(rte_te[element].hyp)
    val_ls.append(rte_te[element].value)
    text_tokens_ls.append(word_tokenize(rte_te[element].text))
    h_tokens_ls.append(word_tokenize(rte_te[element].hyp))

#print(text_ls)
#print(hyp_ls)
#print(val_ls)
#print(text_tokens_ls)
#print(h_tokens_ls)

#put values into a DataFrame
df_text = pd.DataFrame({'text':text_ls})
df_hyp = pd.DataFrame({'hypothesis':hyp_ls})
df_val = pd.DataFrame({'outcome':val_ls})
df_text_tokens = pd.DataFrame({'text_tokens':text_tokens_ls})
df_h_tokens = pd.DataFrame({'hypothesis_tokens':h_tokens_ls})
#df = pd.concat([df_text, df_hyp, df_val, df_text_tokens, df_h_tokens], axis=1) #use this later on to build dataframe from features

#tokens = [i for i in df_text_tokens['text_tokens'][2] if i not in string.punctuation]
#print(tokens)
df_text_tokens['text_tokens_new']=df_text_tokens['text_tokens'].apply(lambda x: [i for i in x
                                                  if i not in string.punctuation])
df_text_tokens['text_tokens_nw']=df_text_tokens['text_tokens_new'].apply(lambda x: [item for item in x if item not in stop])


df_h_tokens['hypothesis_tokens_new']=df_h_tokens['hypothesis_tokens'].apply(lambda x: [i for i in x
                                                  if i not in string.punctuation])

df_h_tokens['hypothesis_nw']=df_h_tokens['hypothesis_tokens_new'].apply(lambda x: [item for item in x if item not in stop])
df_text_tokens['text_tokens_nn'] = df_text_tokens['text_tokens_nw'].apply(lambda x:[lemmatize(item) for item in x])

df_h_tokens['hypothesis_tokens_nn'] = df_h_tokens['hypothesis_nw'].apply(lambda x:[lemmatize(item) for item in x])

df = pd.concat([df_text, df_hyp, df_val, df_text_tokens, df_h_tokens], axis=1)

#ngram functions
def find_unigrams(text):
  return list(zip(*[text[i:] for i in range(1)]))
def find_bigrams(text):
  return list(zip(*[text[i:] for i in range(2)]))
def find_trigrams(text):
  return list(zip(*[text[i:] for i in range(3)]))
def find_quadgrams(text):
  return list(zip(*[text[i:] for i in range(4)]))
def find_figrams(text):
  return list(zip(*[text[i:] for i in range(5)]))

#text ngrams
print('Text ngrams')
df['text_unigrams'] = df['text_tokens_nn'].apply(find_unigrams)
df['text_bigrams'] = df['text_tokens_nn'].apply(find_bigrams)
df['text_trigrams'] = df['text_tokens_nn'].apply(find_trigrams)
df['text_quadgrams'] = df['text_tokens_nn'].apply(find_quadgrams)
df['text_figrams'] = df['text_tokens_nn'].apply(find_figrams)
#print(df['text_bigrams'].head(5))

#hypothesis ngrams
print('Hypothesis ngrams')
df['hyp_unigrams'] = df['hypothesis_tokens_nn'].apply(find_unigrams)
df['hyp_bigrams'] = df['hypothesis_tokens_nn'].apply(find_bigrams)
df['hyp_trigrams'] = df['hypothesis_tokens_nn'].apply(find_trigrams)
df['hyp_quadgrams'] = df['hypothesis_tokens_nn'].apply(find_quadgrams)
df['hyp_figrams'] = df['hypothesis_tokens_nn'].apply(find_figrams)
#print(df['hyp_bigrams'].head(5))
#checks
print(type(df['hyp_unigrams'][1]))
#print(df.head(5))
#check how many text_unigrams match hyp_unigrams and divide by count of hyp_unigrams
for grams in list:
    df['unigram_match'] = df['text_unigrams'].isin(df['hyp_unigrams']).any(1)
#df['unigram_match'] = any(x in df['text_unigrams'] for x in df['hyp_unigrams'])

#df['unigram_match'] = df.iterrows(any(x in df['text_unigrams'] for x in df['hyp_unigrams']))

print(df['unigram_match'])

#print(df['unigram_match'].head(5))



#print(df_text_tokens['text_tokens_nn'].head(5))

#building ngrams

print(df.head(5))
