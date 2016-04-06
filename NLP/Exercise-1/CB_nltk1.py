from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk

sentence = input("Give me a sentence to check for spelling mistakes > ")
tokens = nltk.word_tokenize(sentence.lower())

def check_spelling():
    for word in tokens:
        #check whether word is in wordnet (this doesnt include stopwords) so also check whether it appears in stopwords, in this case dont show.
        if not wordnet.synsets(word) and word not in stopwords.words('english'):
            print(word)

check_spelling()
