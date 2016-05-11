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


def lemmatize(word):
    lemma = nltk.corpus.wordnet.morphy(word, pos = nltk.corpus.wordnet.VERB)
    if lemma is not None:
        return lemma
    return word


stop = stopwords.words('english')
# parse the XML
print("parsing the XML")
rte_te = rte.pairs(['/Users/rahulmehra/Downloads/xml_NLP.xml'])
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
#df = pd.concat([df_text, df_hyp, df_val, df_text_tokens, df_h_tokens], axis=1)

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
print(df.head(5))
