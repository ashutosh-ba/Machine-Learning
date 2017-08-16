# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:32:42 2017

@author: Ashutosh
"""


import os
import numpy as np 
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import *
stemmer = PorterStemmer()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
"""
tar = tarfile.open("20news-19997.tar.gz", "r:gz")
for member in tar.getmembers():
     f = tar.extractfile(member)
     if f:
         content = f.read()
         Data = np.loadtxt(content)

output = pd.DataFrame( data={"columnA":Data} )
output.head(10)
tar.getmembers()
print(Data[5])
myfile = open("C:\\Users\\Ashutosh\\Desktop\\CBA\\Machine Learning DMG2\\Assignment\\71620010 Data\\20news-19997.tar\\20news-19997\\20_newsgroups\\sci.crypt\\14147", 'rb').read()
myfile
loadfile = np.loadtxt(myfile)
loadfile
"""

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
    # Stemming
    meaningful_words = [stemmer.stem(w) for w in meaningful_words]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 

path = 'C:\\Users\\Ashutosh\\Desktop\\CBA\\Machine Learning DMG2\\Assignment\\71620010 Data\\20news-19997.tar\\20news-19997\\20_newsgroups'  
clean_reviews = []   
newsgroup =[]


for dirname in os.listdir(path):
    print ("Starting Review %s" %dirname)
    for filename in os.listdir(os.path.join(path,dirname)):
        with open(os.path.join(path,dirname,filename), 'r') as filedata:
            data = filedata.read()
            clean_reviews.append(review_to_words(data))
            newsgroup.append(dirname)
            
len(clean_reviews)
len(newsgroup)

# Creating a Dataframe of News reviews and news groups
data_df = pd.DataFrame(data={"News":clean_reviews, "Group":newsgroup} )

data_df.shape

# random split the data into train:test :: 70:30
train_data=data_df.sample(frac=0.7,random_state=200)
test_data=data_df.drop(train_data.index)

train_data.shape
test_data.shape

test_data.groupby('Group').size()
train_data.groupby('Group').size()
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
features_train = vectorizer.fit_transform(train_data["News"])
# Numpy arrays are easy to work with, so convert the result to an 
# array
features_train = features_train.toarray()

vocab = vectorizer.get_feature_names()
print (vocab)

#Naive bayes
#Initialize a naive bayes  Multinomial classifier with laplacian Smoothing alpha =30
model=MultinomialNB(alpha=30, class_prior=None, fit_prior=True)

# Fit the model  to the training set, using the bag of words as 
# features and the newsgroup labels as the response variable

classify= model.fit(features_train,train_data["Group"])
classify

#predict training set 

predict_train = classify.predict(features_train)

# creating the confusion matrix
confusion_matrix(train_data["Group"], predict_train)

#Find the accuracy score of taining set
accuracy_score(train_data["Group"], predict_train)
# 0.89819974282040294

#Test Set preparation 
features_test = vectorizer.transform(test_data["News"])
features_test = features_test.toarray()

# Use the Naive Bayes to make news group label predictions for test set
predicted = classify.predict(features_test)

# creating the confusion matrix
confusion_matrix(test_data["Group"], predicted)

#Find the accuracy score of test set
accuracy_score(test_data["Group"], predicted)

# 0.86514419069844972

#Finding manual accuracy
count = len(["ok" for idx, label in enumerate(test_data["Group"]) if label == predicted[idx]])
print ( "Accuracy Rate, which is calculated manually is: %f" % (float(count) / len(test_data)))

#Accuracy Rate, which is calculated manually is: 0.865144




