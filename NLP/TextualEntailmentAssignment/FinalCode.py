import nltk
from nltk.corpus import rte
import pandas as pd
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils


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

df["unigram_overlap"] = df.apply(lambda x: len(set(x["text_unigrams"]) & set(x["hyp_unigrams"])), axis = 1)
df["bigram_overlap"] = df.apply(lambda x: len(set(x["text_bigrams"]) & set(x["hyp_bigrams"])), axis = 1)
df["trigram_overlap"] = df.apply(lambda x: len(set(x["text_trigrams"]) & set(x["hyp_trigrams"])), axis = 1)
df["quadgram_overlap"] = df.apply(lambda x: len(set(x["text_quadgrams"]) & set(x["hyp_quadgrams"])), axis = 1)
df["figram_overlap"] = df.apply(lambda x: len(set(x["text_figrams"]) & set(x["hyp_figrams"])), axis = 1)

# POS Features

# Named Entity Features


# Tree Features


# Sentiment Features


#Extra-unigram Features
df["hyp_extra"]= df.apply(lambda x: len(set(x["hyp_unigrams"]) - set(x["text_unigrams"])), axis = 1)
df["text_extra"]= df.apply(lambda x: len(set(x["text_unigrams"]) - set(x["hyp_unigrams"])), axis = 1)


# After building features convert to sparkSQL dataframe
dfs = sqlCtx.createDataFrame(df)

#Machine Learning
assembler = VectorAssembler(inputCols =["unigram_overlap","bigram_overlap","trigram_overlap","quadgram_overlap","figram_overlap"],outputCol ="outcome")

transformed = assembler.transform(dfs)

LPs =transformed.select(col("outcome").alias("label"),col("unigram_overlap").alias("features")).map(lambda row: LabeledPoint(row.label,row.features))
rddLPs = dfs.map(lambda row: LabeledPoint(row["outcome"],[row[-5:]]))
(trainingData, testData) =rddLPs.randomSplit([0.7, 0.3])
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))

#Another Model
model = GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo={}, numIterations=10)
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
# Similar performance.
# Test Error = 0.3729508196721312 

#Model for test dataset
#Run the same pre processing and parsing steps
#Building a model
model = GradientBoostedTrees.trainClassifier(rddLPs,categoricalFeaturesInfo={},numIterations=10)
predictions = model.predict(newData.map(lambda x: x.features))
labelsAndPredictions = newData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(newData.count())
print('Test Error = ' + str(testErr))

#Saving the model and the predictions
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
predicitons =predictions.coalesce(1)
predictions.saveAsTextFile("Predictions")

#The only things which change in the final 
