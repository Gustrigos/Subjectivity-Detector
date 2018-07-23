# Subjectivity-Detector
Natural Language Processing project written in Python 3.6.5 to determine whether a body of text is written objectively or subjectively. 

Following research by Rizk and Awad (2018) on subjectivity analysis of sports articles, I decided to integrate a supervised support vector machine model with a linear kernel instead of a syntactic genetic algorithm to process the labeled training examples and the extracted features to classify wheter the given input belongs to sports articles that were written subjectively or objectively by the authors.

Rizk and Awad (2018) discuss that the effectiveness of the classifier is through a frequentist approach based on the presence of specific syntactic features and their frequency of occurrences. Their process of determining features that have a frequent appeareance on subjective and objective articles respectively is mainly based by Parts of Speech (POS) tagging.

They determine the use of certain syntactic attributes as a measure of indication of objectivity and subjectivity in sport articles:

Objectivity:
- Quotations.
- Past tense verbs.
- Third person pronouns.
- Numerical values and dates.

Subjectivity: 
- Imperative verbs.
- Exclamation and question marks.
- First and second person pronouns.
- Present tense verbs with third person pronouns.
- Comparative and superlative adverbs and adjectives.
- Present tense verbs with first and second pronouns.

Program:
The program consists of the following pipeline:
1) Preprocessing text: consists of simplification of articles, text cleaning, and tokenization of simplified corpus.
2) Vocabulary list: concanates all articles, builds a dictionary of the most frequent tokens, and creates a feature vector. 
3) Training the model: splitting dataset into training, cross-validation, and test sets, implementing SVM and choosing best parameters.
4) Testing predictive accuracy: Implementing best parameters for the trained SVM and get the classifier's accuracy.

To run the program:
The following libraries need to be installed and imported: 
numpy: for linear algebra (creating vectors, matrices, and cross-operations between them).
matplotlib: for plotting learning parameters.
scipy.io: for input and output user's interaction.
sklearn: for the Support Vector Machine model.
re: for text preprocessing and regular expressions.
pandas: for loading and organizing the data set. 
nltk: for the tokenization of words.

Acknowledgements:
To Yara Rizk and Mariette Awad (American University of Beirut, IEEE Electrical & Computer Engineering) for the dataset and the research findings. David Kaleko (columbia school of Fu Foundation School of Engineering and Applied Science) for the model preset. Andrew Ng (Stanford University Computer Science Department) for the intuition behind the model.
