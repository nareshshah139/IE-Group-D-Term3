#After Chris's code
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
sqlCtx.createDataFrame(df)
