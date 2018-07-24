import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm
import pandas as pd
import nltk, nltk.stem.porter
from nltk.tokenize import word_tokenize

print('Objective/ Subjective Text Classification')

# 1) Preprocessing Text
print('Visualization of Data')
input('Press <Enter> to continue')
print('The following is a visualization of a random sports article from the data set: ')

data_dir = 'dataclassified/'

# read objective and subjective articles into arrays (encoded in latin-1)
objective_articles = []
for objective_article in os.listdir(data_dir+'objective_1/'):
  objective_articles.append(open(data_dir+ 'objective_1/' + objective_article, 'r', encoding= 'latin-1').read())

subjective_articles = []
for subjective_article in os.listdir(data_dir + 'subjective_1/'):
	subjective_articles.append(open( data_dir + 'subjective_1/' + subjective_article, 'r', encoding= 'latin-1').read())

# Print the first objective article (visualization of data).

print("\n", objective_articles[1], "\n")

# 1.1) Simplification of articles.

def  preProcess ( article ):
	# Make the entire article lower case
	article = article.lower()

	# Any quoted text gets replaced with the string 'quote'
	article = re.sub(' "[^""]" ', 'quote', article)

    # Any numbers get replaced with the string 'number'
	article = re.sub('[0-9]+','number', article)

    # Anything ending with '!' replaced with 'exmark'
	article = re.sub('[^\s]+[!]+', 'exmark', article)

	 # Anything ending with '?' replaced with 'questmark'
	article = re.sub('[^\s]+[\?]+', 'questmark', article)

    # Strings with "@" in the middle are replaced to 'emailaddr'
	article = re.sub('[^\s]+@[^\s]+', 'emailaddr', article)
    
    # Various monetary signs get replaced with 'monetaryval'
	article = re.sub('[$|€|¥|£]+[^\s]+', 'monetaryval', article)
    
	return article

# 1.2) Tokenization of simplified articles. (Ordered list of Strings and Indices)

def article2TokenList( raw_article ):

	article = preProcess( raw_article )

	# Split the article into individual words (tokens) (split by many delimeters)
	tokens = word_tokenize(article)

	# loop over each word (token) and use a stemmer to shorten it,
	# then check if the word is in the vocab_list.  
	# If true, store what index in the vocab_list the word is

	tokenlist = []

	for token in tokens:

		# Remove any non alphanumeric characters
		token = re.sub('[^a-zA-Z0-9]', '', token)

		# Get rid of empty tokens
		if not len(token): continue

	# Filter words with POS tagging and store them in a list.
	allowed_word_types = ["VBD", "PRP", "PRP$", "JJR", "RBR", "JJS", "RBS"]
	extrafeatures = ["quote", "number", "ex-mark", "quest-mark", "monetaryval"]
	pos = nltk.pos_tag( tokens )
	for w in pos:
		tokenlist.append(w[0])

	return tokenlist

print('Word Dictionary')
input('Press <Enter> to continue')
print('The following regards the simplification of articles through the tokenazation of its stemmed words: *this might take a moment to load*')

# 2) Vocabulary List
# 2.1) Build the dictionary
# Building a dictionary by using the features with most appeareances in the whole corpus. 

# Concanate all articles into one to form a corpus.
# Tokenize each word and count it. 
total_objective_article = [' '.join(objective_articles[:]) ]
total_subjective_article = [' '.join(subjective_articles[:])]
total_corpus = total_objective_article[0]+ total_subjective_article[0]

word_counts = Counter(article2TokenList(total_corpus))

print ('The total number of unique filtered tokens is: ')
print (len(word_counts))
print ('The twenty most common stemmed tokens are:')
print ([ str(x[0]) for x in word_counts.most_common(20) ])

# store tokens into a enumerated matrix. 
common_words = [ str(x[0]) for x in word_counts.most_common(20000) ]
vocab_dict = dict((item, i) for i, item in enumerate(common_words))

# 2.2) Articles into Indices 

# Converting each article into a list of these indices
def article2VocabIndices( raw_article, vocab_dict ):
	tokenlist = article2TokenList( raw_article )
	index_list = [vocab_dict[token] for token in tokenlist if token in vocab_dict ]
	return index_list

# 2.3) Article into feature vector
# detects wheter the filtered features are in an article.
def article2FeatureVector( raw_article, vocab_dict ):
	n = len(vocab_dict)
	result = np.zeros((n,1))
	vocab_indices = article2VocabIndices( raw_article, vocab_dict )
	for idx in vocab_indices:
		result[idx] = 1
	return result
