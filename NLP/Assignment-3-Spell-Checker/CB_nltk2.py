from nltk.corpus import brown
from nltk.corpus import stopwords
import nltk
from nltk.metrics import *
import pandas as pd

sentence = input("Give me a sentence to check for spelling mistakes > ")
tokens = nltk.word_tokenize(sentence.lower())
words = set(brown.words())
words_brown = brown.words()
words_brown = [word for word in words_brown if len(word) > 1]
words_brown = [word for word in words_brown if word not in stopwords.words('english')]

#identify incorrectly spelled words within the sentence
incorrect_words = []
def check_spelling():
    for word in tokens:
        #check whether word is in wordnet (this doesnt include stopwords) so also check whether it appears in stopwords, in this case dont show.
        if word not in words and word not in stopwords.words('english'):
            incorrect_words.append(word)
    return(incorrect_words)
check_spelling()
print(incorrect_words)

#compute the edit_distance between incorrectly spelled words and the words within brown corpus
word_dist = []
for item in incorrect_words:
    for word in words:
        if(edit_distance(item,word) < 3):
            word_dist.append({'item':item,'words':word,'distance':edit_distance(item,word)})
word_dist = pd.DataFrame(word_dist)
print(word_dist)

#todo
#1 create dataframe with word and frequency in brown
# Output frequent words
fdist = nltk.FreqDist(words_brown)
word_freq = []
for word, frequency in fdist.most_common(): #change the 100 to include more
    word_freq.append({'words':word, 'frequency':frequency})
word_freq = pd.DataFrame(word_freq)
#print(word_freq.head(5))

#try to work out this join
#print(df.head(5))
#df = word_dist.join(word_freq, on=['word'], how='outer')
#Idea: additioanl feature to ensure we propose the correct word could be to look at the length of the word and give higher importance to words that have the same length in df['item'] and df['word']. Works well for "todya" does not work well for "helo"
result = pd.merge(word_dist,word_freq,on="words")
print(result.head(5))
result = result.sort_values(by ="frequency",ascending = False)

for word in set(result["item"]):
    print(word,result["words"][result["item"]== word].head(5),result["frequency"][result["item"]== word].head(5))




#test helo hello this is how it goes todya
