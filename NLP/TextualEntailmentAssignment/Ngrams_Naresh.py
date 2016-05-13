import nltk
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
from nltk import word_tokenize
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd

gutenberg_raw = gutenberg.raw("austen-emma.txt")


webtext_words = webtext.words()
print(webtext_words)

nps_chat_raw = nps_chat.raw()


brown_raw = brown.raw()
print(brown_raw)

reuters_words = reuters.words()
print(reuters_words)

inaugral_words = inaugural.words()
print(inaugral_words)

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(gutenberg_raw)



s=set(stopwords.words('english'))
gutenberg_filtered = filter(lambda w: not w in s,tokens)



def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

bigrams = find_ngrams(tokens,2)
trigrams = find_ngrams(tokens,3)
fourgrams = find_ngrams(tokens,4)
fivegrams = find_ngrams(tokens,5)

unigram_counts = Counter(tokens).most_common(10)
print(unigram_counts)
bigram_counts = Counter(list(bigrams)).most_common(10)
print(bigram_counts)
trigram_counts = Counter(list(trigrams)).most_common(10)
print(trigram_counts)
fourgram_counts = Counter(list(fourgrams)).most_common(10)
print(fourgram_counts)
fivegram_counts = Counter(list(fivegrams)).most_common(10)
print(fivegram_counts)


