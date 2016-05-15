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

#Model for test dataset
#Run the same pre processing and parsing steps
#Building a model
model = GradientBoostedTrees.trainClassifier(rddLPs,categoricalFeaturesInfo={},numIterations=10)
predictions = model.predict(newData.map(lambda x: x.features))

#Saving the model and the predictions
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
predicitons =predictions.coalesce(1)
predictions.saveAsTextFile("Users/naresh/Downloads/Predictions")
print('Test Error = ' + str(testErr))

