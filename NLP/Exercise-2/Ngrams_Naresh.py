
# Importing the modules

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

# Pick out the first of these texts — Emma by Jane Austen — and give it a short name, gutenberg_raw
gutenberg_raw = gutenberg.raw("austen-emma.txt")

# Pick out the words from webtext corpus and give it a short name, webtext_words
webtext_words = webtext.words()
print(webtext_words)

# Pick out the text from np_chat corpus and name it as nps_chat_raw
nps_chat_raw = nps_chat.raw()

# Pick out the text from brown corpus and name it as brown_raw
brown_raw = brown.raw()
print(brown_raw)

# Pick out the text from reuters corpus and name it as reuters_words
reuters_words = reuters.words()
print(reuters_words)

# Pick out the text from inaugural corpus and name it as inaugral_raw
inaugral_words = inaugural.words()
print(inaugral_words)

# Creating a variable for tokenizing words
tokenizer = RegexpTokenizer(r'\w+')

# Tokenizing the words in gutenberg corpus and assigning it to a variable named tokens
tokens = tokenizer.tokenize(gutenberg_raw)


# Assigning the stopwords to a variable s
s=set(stopwords.words('english'))

# Removing the stopwords from gutenberg file
gutenberg_filtered = filter(lambda w: not w in s,tokens)


# Defining a function to find ngrams in gutenberg corpus
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# Finding bigrams in gutenberg corpus and assigning it to a variable called bigrams
bigrams = find_ngrams(tokens,2)

# Finding trigrams in gutenberg corpus and assigning it to a variable called trigrams
trigrams = find_ngrams(tokens,3)

# Finding fourgrams in gutenberg corpus and assigning it to a variable called fourgrams
fourgrams = find_ngrams(tokens,4)

# Finding fivegrams in gutenberg corpus and assigning it to a variable called fivegrams
fivegrams = find_ngrams(tokens,5)

# Counting the unigrams in our gutenberg corpus and prinitng it
unigram_counts = Counter(tokens).most_common(10)
print(unigram_counts)

# Counting the bigrams in our gutenberg corpus and prinitng it
bigram_counts = Counter(list(bigrams)).most_common(10)
print(bigram_counts)

# Counting the trigrams in our gutenberg corpus and prinitng it
trigram_counts = Counter(list(trigrams)).most_common(10)
print(trigram_counts)

# Counting the fourgrams in our gutenberg corpus and prinitng it
fourgram_counts = Counter(list(fourgrams)).most_common(10)
print(fourgram_counts)

# Counting the fivegrams in our gutenberg corpus and prinitng it
fivegram_counts = Counter(list(fivegrams)).most_common(10)
print(fivegram_counts)


