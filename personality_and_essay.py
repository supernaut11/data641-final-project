import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
import nltk
import re
import argparse
from sklearn import svm, tree
from sklearn.svm import SVC
import os
import csv, re
import datetime
import string
from tqdm import tqdm
import codecs
from collections import Counter, defaultdict
import spacy
from spacy.lang.en import English
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import statistics


data = pd.read_csv('./data/wcpr_mypersonality.csv', encoding='cp1252')

array_agg = lambda x: ' '.join(x.astype(str))

group_status = data.groupby(['#AUTHID']).agg({'STATUS': array_agg})
neu_status = data[['#AUTHID','cNEU']].drop_duplicates()

df = pd.merge(group_status, neu_status, on=["#AUTHID"])

essay_data = pd.read_csv('./data/wcpr_essays.csv', encoding='cp1252')


def split_into_lemmas(text): # creates function to lemmatize words
    text = str(text).lower() # converts all words to lowercase
    words = TextBlob(text).words # returns all words
    return [word.lemmatize() for word in words] # lemmatizes every word

def lr_rf_classify(essay = False):
    X_train, X_test, y_train, y_test = train_test_split(df['STATUS'], df['cNEU'], random_state = 42)
    if essay == True:
        X_test = essay_data['TEXT']
        y_test = essay_data['cNEU']
    
    forest = RandomForestClassifier()
    lr_classifier = LogisticRegression()
    #vect = CountVectorizer(analyzer=split_into_lemmas, decode_error='replace', min_df=2, stop_words='english', ngram_range=(1,2), max_features=100000)
    vect = CountVectorizer(stop_words='english', ngram_range=(1,3), max_features=100000)
    
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    forest.fit(X_train_dtm, y_train)
    y_pred_class = forest.predict(X_test_dtm)
    predicted_labels = forest.predict(X_test_dtm)
    print('Test set: ', X_test_dtm.shape)
    print('\n')
    print('Random Forest Accuracy:',accuracy_score(y_test,predicted_labels))
    for label in ['y', 'n']:
        print(' Precision for label {} = {}'.format(label, metrics.precision_score(y_test, predicted_labels, pos_label=label)))
        print(' Recall for label {} = {}'.format(label, metrics.recall_score(y_test, predicted_labels, pos_label=label)))
    #rf_acc.append(accuracy_score(y_test, predicted_labels))
        
    lr_classifier.fit(X_train_dtm, y_train)
    y_pred_class = lr_classifier.predict(X_test_dtm)
    predicted_labels = lr_classifier.predict(X_test_dtm)
    print('Logistic Regression Accuracy:',accuracy_score(y_test,predicted_labels))
    for label in ['y', 'n']:
        print(' Precision for label {} = {}'.format(label, metrics.precision_score(y_test, predicted_labels, pos_label=label)))
        print(' Recall for label {} = {}'.format(label, metrics.recall_score(y_test, predicted_labels, pos_label=label)))
    #lr_acc.append(accuracy_score(y_test, predicted_labels))
    print('\n')


print('Testing on personality data')
lr_rf_classify()

print('Testing on essay data')
lr_rf_classify(essay = True)