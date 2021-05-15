import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
import nltk
import re


data = pd.read_csv('wcpr_mypersonality.csv', encoding='cp1252')

array_agg = lambda x: ' '.join(x.astype(str))

group_status = data.groupby(['#AUTHID']).agg({'STATUS': array_agg})
neu_status = data[['#AUTHID','cNEU']].drop_duplicates()

df = pd.merge(group_status, neu_status, on=["#AUTHID"])

#df.head()


essay_data = pd.read_csv('wcpr_essays.csv', encoding='cp1252')

#essay_data.head()


def lr_rf_classify():
    forest = RandomForestClassifier()
    lr_classifier = LogisticRegression()
    vect = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=100000)
    
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    forest.fit(X_train_dtm, y_train)
    y_pred_class = forest.predict(X_test_dtm)
    predicted_labels = forest.predict(X_test_dtm)
    print('Test set: ', X_test_dtm.shape)
    print('\n')
    print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
    for label in ['y', 'n']:
        print(' Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print(' Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))
        
    lr_classifier.fit(X_train_dtm, y_train)
    y_pred_class = lr_classifier.predict(X_test_dtm)
    predicted_labels = lr_classifier.predict(X_test_dtm)
    print('Logistic Regression Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
    for label in ['y', 'n']:
        print(' Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print(' Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))
    print('\n')

print('Testing on personality data')
X_train, X_test, y_train, y_test = train_test_split(df['STATUS'], df['cNEU'], random_state=42)
lr_rf_classify()


print('Testing on essay data')
X_test = essay_data['TEXT']
y_test = essay_data['cNEU']
lr_rf_classify()

# Testing K Folds
# Adapted from https://www.analyseup.com/python-machine-learning/stratified-kfold.html

skf = StratifiedKFold(n_splits=5)
target = df.loc[:,'cNEU']

model = RandomForestClassifier()

fold_acc = list()

def train_model(train, test, fold_no):
    X_train, X_test, y_train, y_test = train_test_split(df['STATUS'], df['cNEU'])
    #X_test = essay_data['TEXT']
    #y_test = essay_data['cNEU']
    
    vect = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=100000)
    
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    model.fit(X_train_dtm,y_train)
    predictions = model.predict(X_test_dtm)
    print('Fold',str(fold_no),'Accuracy:',accuracy_score(y_test,predictions))
    fold_acc.append(accuracy_score(y_test, predictions))
    
fold_no = 1
for train_index, test_index in skf.split(df, target):
    train = df.loc[train_index,:]
    test = df.loc[test_index,:]
    train_model(train,test,fold_no)
    fold_no += 1

print('\n')
print('Average Accuracy: ',statistics.mean(fold_acc))