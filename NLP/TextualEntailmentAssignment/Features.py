#After Chris's code
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col


# N Gram Features

df["unigram_overlap"] = df.apply(lambda x: len(set(x["text_unigrams"]) & set(x["hyp_unigrams"])), axis = 1)
df["bigram_overlap"] = df.apply(lambda x: len(set(x["text_bigrams"]) & set(x["hyp_bigrams"])), axis = 1)
df["trigram_overlap"] = df.apply(lambda x: len(set(x["text_trigrams"]) & set(x["hyp_trigrams"])), axis = 1)
df["quadgram_overlap"] = df.apply(lambda x: len(set(x["text_quadgrams"]) & set(x["hyp_quadgrams"])), axis = 1)
df["figram_overlap"] = df.apply(lambda x: len(set(x["text_figrams"]) & set(x["hyp_figrams"])), axis = 1)

# POS Features




# Named Entity Features




# Tree Features


# Sentiment Features




# After building features convert to sparkSQL dataframe
dfs = sqlCtx.createDataFrame(df)

#Machine Learning
assembler = VectorAssembler(inputCols =["unigram_overlap","bigram_overlap","trigram_overlap","quadgram_overlap","figram_overlap"],outputCol ="entailment")

transformed = assembler.transform(dfs)

LPs =transformed.select(col("entailment").alias("label"),col("unigram_overlap").alias("features")).map(lambda row: LabeledPoint(row.label,row.features))
