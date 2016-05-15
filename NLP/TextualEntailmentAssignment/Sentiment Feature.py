import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


def evaluate_features(feature_select):
    #reading pre-labeled input and splitting into lines
    posSentences = open('/Users/mehdinassih/Desktop/Semester 3/Natural Language Processing/Textual Entailment Assignment/TextualEntailment.xlm', 'r')
    negSentences = open('/Users/mehdinassih/Desktop/Semester 3/Natural Language Processing/Textual Entailment Assignment/TextualEntailment.xlm', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
posFeatures = []
negFeatures = []

#breaks up the sentences into lists of individual words and appends 'pos' or 'neg' after each list
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)
        
print ('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
print ('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
print ('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
print ('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg']))
classifier.show_most_informative_features(10)
    

def make_full_dict(words):
    return dict([(word, True) for word in words])
    

print ('using all words as features')
evaluate_features(make_full_dict)

def create_word_scores():
    #splits sentences into lines
    posSentences = open('/Users/mehdinassih/Desktop/Semester 3/Natural Language Processing/Textual Entailment Assignment/TextualEntailment.xlm', 'r')
    negSentences = open('/Users/mehdinassih/Desktop/Semester 3/Natural Language Processing/Textual Entailment Assignment/TextualEntailment.xlm', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    #creates lists of all positive and negative words
posWords = []
negWords = []
for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i)
        posWords.append(posWord)
for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i)
        negWords.append(negWord)
    
posWords = list(itertools.chain(*posWords))
negWords = list(itertools.chain(*negWords))


word_fd = FreqDist()

cond_word_fd = ConditionalFreqDist()

for word in posWords:
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())
        

pos_word_count = cond_word_fd['pos'].N()
neg_word_count = cond_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
    