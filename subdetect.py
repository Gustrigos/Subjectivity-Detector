# Support Vector Machine for Text Classification

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm
import pandas as pd
import nltk, nltk.stem.porter
from nltk.tokenize import word_tokenize
import collections

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

	# Split the article into individual words (tokens) 
	tokens = word_tokenize(article)

	tokenlist = []

	for token in tokens:

		# Remove any non alphanumeric characters
		token = re.sub('[^a-zA-Z0-9]', '', token)

		# Get rid of empty tokens
		if not len(token): continue

	# Filter words with POS tagging and store them in a list.
	allowed_word_types = ["VBD", "PRP", "PRP$", "JJR", "RBR", "JJS", "RBS", "RB", "JJ", "UH"]
	extrafeatures = ["quote", "number", "exmark", "questmark", "monetaryval"]
	pos = nltk.pos_tag( tokens )
	for w in pos:
		if w[1] in allowed_word_types:
			tokenlist.append(w[0])
		if w[0] in extrafeatures:
			tokenlist.append(w[0])

	# print(tokenlist)
	# print("number of tokenlist: ", len(tokenlist))

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
total_corpus = total_objective_article[0] + total_subjective_article[0]

filtered_corpus = article2TokenList(total_corpus)
word_counts = collections.Counter(filtered_corpus)

# print("filtered_corpus: ", filtered_corpus)
# print("len of filtered: ", len(filtered_corpus))

# print("wordcounts: ", word_counts)
print ('The total number of unique filtered tokens is: ', len(word_counts))
print ('The twenty most common stemmed tokens are:')
print ([ str(x[0]) for x in word_counts.most_common(20) ])

# store tokens into a enumerated matrix. 
common_words = [ str(x[0]) for x in word_counts.most_common(15000) ]
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

# 3) Training SVM for objective/ subjective classification

# 3.1) Training Set, Cross validation set, and test set
# Read in the training set and test set provided.

# Training set X matrix and y vector by 60% of data.
n_objective_train = int(len(objective_articles)*0.6)
n_subjective_train = int(len(subjective_articles)*0.6)

objective_train = [article2FeatureVector(x, vocab_dict)
					for x in objective_articles[:n_objective_train]]
subjective_train = [article2FeatureVector(x, vocab_dict)
					for x in subjective_articles[:n_subjective_train]]

Xtrain = np.concatenate(objective_train + subjective_train, axis=1).T
ytrain = np.concatenate(
	(np.zeros((n_objective_train,1)),
	np.ones((n_subjective_train,1))
	),axis=0)

# Cross Validation set X matrix and y vector by 20% of data.
n_objective_cv = int(len(objective_articles)*0.2)
n_subjective_cv = int(len(subjective_articles)*0.2)

objective_cv = [article2FeatureVector( x, vocab_dict ) 
                 for x in objective_articles[n_objective_train:n_objective_train+n_objective_cv]]
subjective_cv    = [article2FeatureVector( x, vocab_dict ) 
                 for x in subjective_articles [n_subjective_train:n_subjective_train+n_subjective_cv]]

Xcv = np.concatenate(objective_cv + subjective_cv, axis=1).T
ycv = np.concatenate(
	(np.zeros((n_objective_cv,1)),
	np.ones((n_subjective_cv,1))
	),axis=0)

# Test set X matrix and y vector by the remaining data.
n_objective_test = len(objective_articles) - n_objective_train - n_objective_cv
n_subjective_test = len(subjective_articles) - n_subjective_train - n_subjective_cv

objective_test = [article2FeatureVector( x, vocab_dict ) 
                 for x in objective_articles[-n_objective_test:]]
subjective_test    = [article2FeatureVector( x, vocab_dict ) 
                 for x in subjective_articles [-n_subjective_test:]]

Xtest = np.concatenate(objective_test + subjective_test, axis=1).T
ytest = np.concatenate(
	(np.zeros((n_objective_test,1)),
	np.ones((n_subjective_test,1))
	),axis=0)

print("Choosing the Best SVM's parameters")
input('Press <Enter> to continue')
print('The following is a graph visualization of the classification error(%) against C value of both CV set and Training Set: ')

# 3.2) SVM parameters and model
# Cross_validation set will help choose parameter C. 
# Creating several instances of Sklearn linear kernel SVMs each with different CS. 
# Evaluate the performance of each on the CV set.
# choose the instance with the best performance. 

myCs = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0 ]
myErrors = []
myErrors_train = []

for myC in myCs:
    
    # Make an instance of an SVM with C=myC and 'linear' kernel
    linear_svm = svm.SVC(C=myC, kernel='linear')

    # Fit the SVM to our Xtrain matrix, given the labels ytrain
    linear_svm.fit( Xtrain, ytrain.flatten() )
    
    # Determine how well this SVM works by computing the
    # classification error on the cross-validation set
    cv_predictions = linear_svm.predict(Xcv).reshape((ycv.shape[0],1))
    cv_error = 100. * float(sum(cv_predictions != ycv))/ycv.shape[0]
    myErrors.append( cv_error )

    # While we're at it, do the same for the training set error
    train_predictions = linear_svm.predict(Xtrain).reshape((ytrain.shape[0],1))
    train_error = 100. * float(sum(train_predictions != ytrain))/ytrain.shape[0]
    myErrors_train.append( train_error )

# Compare bias and variance with Training set and Cross Validation set respective error.
# The graph should be at its minimum error and avoid either of the sets to start increasing.
plt.figure(figsize=(8,5))
plt.plot(myCs,myErrors,'ro--',label='Cross Validation Set Error')
plt.plot(myCs,myErrors_train,'bo--',label='Training Set Error')
plt.grid(True,'both')
plt.xlabel('$C$ Value',fontsize=16)
plt.ylabel('Classification Error [%]',fontsize=14)
plt.title('Best $C$ Value',fontsize=18)
plt.xscale('log')
myleg = plt.legend()

plt.show()


print(" SVM's Predictive Accuracy for Test Set")
input('Press <Enter> to continue')
print('The following is the Test set accuracy from the SVM: ')

# 4) Testing predicitve accuracy for the Test set. 
best_svm = svm.SVC(C=0.1, kernel='linear')
best_svm.fit( Xtrain, ytrain.flatten() )

test_predictions = best_svm.predict(Xtest).reshape((ytest.shape[0],1))
test_acc = 100. * float(sum(test_predictions == ytest))/ytest.shape[0]
print ('Test set accuracy = %0.2f%%' % test_acc)

