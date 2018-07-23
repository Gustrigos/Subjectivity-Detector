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

    #Any numbers get replaced with the string 'number'
	article = re.sub('[0-9]+','number', article)
    
    # Anything ending with '!' replaced with 'ex-mark'
	article = re.sub('[!]$', 'ex-mark', article)

	# Anything ending with '?' replaced with 'quest-mark'
	article = re.sub('[?]$', 'quest-mark', article)
    
  # Strings with "@" in the middle are replaced to 'emailaddr'
	article = re.sub('[^\s]+@[^\s]+', 'emailaddr', article)
    
  #Various monetary signs get replaced with 'monetaryval'
	article = re.sub('[$]|[€]|[¥][£]+', 'monetaryval', article)
    
	return article

