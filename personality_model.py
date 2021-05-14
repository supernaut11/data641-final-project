import numpy as np
import argparse
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn import svm, tree
from sklearn.svm import SVC
import os
import csv, re
import string
from tqdm import tqdm
import codecs
import argparse
from collections import Counter, defaultdict
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    #   - All tokens are lowercased
    #   - All tokens starting with a whitespace character have been filtered out
    #   - All handles (tokens starting with @) have been filtered out
    #   - Any underscores have been replaced with + (since we use _ as a special character in bigrams)
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                             if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                             and not token.startswith("@")                       # ignore  handles
                        ]
    return normalized_tokens   

def ngrams(tokens, n):
    # Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]

# Split on whitespace, e.g. "a    b_c  d" returns tokens ['a','b_c','d']
def whitespace_tokenizer(line):
    return line.split()

def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a punctuation character
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    result = [ngram for ngram in ngrams if ngram[0] not in stopwords and ngram[1] not in stopwords]
    return result

def load_stopwords(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)

def split_training_set(data, test_size=0.3, random_seed=42):
    statuses = []
    labels = []
    for key, value in data.items():
        statuses.append(value['STATUS'])
        labels.append(value['cNEU'])
    X_train, X_test, y_train, y_test = train_test_split(statuses, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True, filter_punc=False):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    nlp = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis = nlp(line)
        spacy_tokens = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams = [token for token in normalized_tokens if token not in stopwords and (not filter_punc or token not in string.punctuation)] 

        # Collect string bigram tokens as features
        bigrams = []
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]
        bigrams = ngrams(normalized_tokens, 2)
        if filter_punc:
            bigrams = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'

        feature_string = [] 
        uni_string = ' '.join(unigrams)
        bi_string = ' '.join(bigram_tokens)
        
        feature_string = uni_string + ' ' + bi_string
        # Add this feature string to the output
        all_features.append(feature_string)

    print(" Feature string for first document: '{}'".format(all_features[0]))  
    return all_features

def most_informative_features(vectorizer, classifier, n=20):
    # Adapted from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
    feature_names       = vectorizer.get_feature_names()
    coefs_with_features = sorted(zip(classifier.coef_[0], feature_names))
    top                 = zip(coefs_with_features[:n], coefs_with_features[:-(n + 1):-1])
    for (coef_1, feature_1), (coef_2, feature_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, feature_1, coef_2, feature_2))

def merge_ocean_data(ocean_entries):
    score_fields = ("sEXT","sNEU","sAGR","sCON","sOPN")
    flag_fields = ("cEXT","cNEU","cAGR","cCON","cOPN")
    output = {}

    output["#AUTHID"] = ocean_entries[0]["#AUTHID"]
    output["STATUS"] = ' '.join([entry["STATUS"] for entry in ocean_entries])
    output["CARDINALITY"] = len(ocean_entries)

    for s in score_fields:
        output[s] = ocean_entries[0][s]
        
    for f in flag_fields:
        output[f] = ocean_entries[0][f]
        
    return output

def load_ocean_data(data_handle):
    authid_data = defaultdict(list)
    reader = csv.DictReader(data_handle)
    for row in reader:
        cur_authid = row['#AUTHID']
        authid_data[cur_authid].append(row)
        
    return {k: merge_ocean_data(v) for k, v in authid_data.items()}

def get_train_test(data_dir, use_sklearn_feature_extraction, ngram_size, filter_punc=False):
    # Load personality data from file such that each user is represented by a single record
    with open(os.path.join(data_dir, 'wcpr_mypersonality.csv'), 'r', encoding='cp1252') as data_handle:
        status_mapping = load_ocean_data(data_handle)

    # Read the dataset in and split it into training documents/labels (X) and test documents/labels (y)
    X_train, X_test, y_train, y_test = split_training_set(status_mapping)

    # Load stopwords from file system
    stopwords_file = "./mallet_en_stoplist.txt"
    stop_words = load_stopwords(stopwords_file)

    if use_sklearn_feature_extraction:
        # Use sklearn CountVectorizer's built-in tokenization to get unigrams and bigrams as features
        X_features_train, training_vectorizer = convert_text_into_features(X_train, stop_words, "word", range=(1,ngram_size))
        X_test_documents = X_test
    else:
        # Roll your own feature extraction.
        # Call convert_lines_to_feature_strings() to get your features
        # as a whitespace-separated string that will now represent the document.
        print("Creating feature strings for training data")
        X_train_feature_strings = convert_lines_to_feature_strings(X_train, stop_words, filter_punc=filter_punc)
        print("Creating feature strings for test data")
        X_test_documents   = convert_lines_to_feature_strings(X_test, stop_words, filter_punc=filter_punc)
        
        X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stop_words, whitespace_tokenizer)
    
    # Apply the "vectorizer" created using the training data to the test documents, to create testset feature vectors
    X_test_features =  training_vectorizer.transform(X_test_documents)

    return training_vectorizer, X_features_train, y_train, X_test_features, y_test

# TODO: Placeholder function for kfolds
def do_kfolds():
    # Stratified KFold Cross-Validation, not working yet
    # skf = StratifiedKFold(n_splits=2)
    # statuses = []
    # labels = []
    # for key, value in status_mapping.items():
    #     statuses.append(value['STATUS'])
    #     labels.append(value['cNEU'])
    # print(statuses)
    # print(labels)
    # X_train, X_test, y_train, y_test = skf.split(statuses,labels)
    pass

def log_reg_classify(vectorizer, X_train, y_train, X_test, y_test, num_most_informative=10, plot_metrics=False):
    # Create a logistic regression classifier trained on the featurized training data
    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(X_train, y_train)

    # Show which features have the highest-value logistic regression coefficients
    print("Most informative features")
    most_informative_features(vectorizer, lr_classifier, num_most_informative)

    predicted_labels = lr_classifier.predict(X_test)

    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['y', 'n']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))

    if plot_metrics:
        print("Generating plots")
        metrics.plot_confusion_matrix(lr_classifier, X_test, y_test, normalize='true')
        metrics.plot_roc_curve(lr_classifier, X_test, y_test)
        plt.show()

def random_forest_classify(X_train, y_train, X_test, y_test):
    # Create a Random Forest classifier
    forest = RandomForestClassifier() # defines Random Forest model
    forest.fit(X_train, y_train)
    
    # Classify the test data and see how well you perform
    # For various evaluation scores see https://scikit-learn.org/stable/modules/model_evaluation.html
    print("Classifying test data")
    predicted_labels = forest.predict(X_test)
    
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['y', 'n']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))

def grid_search_classify(X_train, y_train, X_test, y_test):
    # GridSearch to test parameters
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    grid = GridSearchCV(svc, parameters)
    grid.fit(X_train, y_train)
    # print the selected model
    print(grid.best_estimator_)

    # print best parameter after tuning
    print(grid.best_params_)

    # print the best score
    print(grid.best_score_)

    dec_tree = tree.DecisionTreeClassifier()
    rf = RandomForestClassifier()
    pipe = Pipeline(steps=[('dec_tree', dec_tree)])
    criterion = ['gini', 'entropy']
    max_depth = [2,4,6,8,10,12]
    param_grid = {
        'max_depth': [80, 90, 100, 110],
        'max_features': ['auto', 'sqrt', 'log2'],
        #'min_samples_leaf': [3, 4, 5],
        #'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300],
        'criterion' :criterion
    }
    parameters = dict(dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)
    grid2 = GridSearchCV(pipe, parameters)
    grid2.fit(X_train, y_train)
    # print best parameter after tuning
    print(grid2.best_params_)
    # print the best score
    print(grid2.best_score_)

    grid3 = GridSearchCV(rf, param_grid)
    grid3.fit(X_train ,y_train)
    print(grid3.best_params_)
    print(grid3.best_score_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for running this script')
    parser.add_argument('--data-dir', default='.', help='Location of project data resources')
    parser.add_argument('--n-gram', default=2, type=int, help='For n-gram models, the value of n (only works with sklearn)')
    parser.add_argument('--filter-punc', default=False, action='store_true', help='Filter punctuation')
    parser.add_argument('--use_sklearn_features', default=False, action='store_true', help="Use sklearn's feature extraction")
    parser.add_argument('--plot_metrics', default=False, action='store_true', help="Generate figures for evaluation")
    parser.add_argument('--num_most_informative', default=10, action='store', type=int, help="Number of most-informative features to show")
    args = parser.parse_args()
    
    vectorizer, X_train, y_train, X_test, y_test = get_train_test(args.data_dir, args.use_sklearn_features, 
        args.n_gram, args.filter_punc)
    
    log_reg_classify(vectorizer, X_train, y_train, X_test, y_test, args.num_most_informative, args.plot_metrics)
    random_forest_classify(X_train, y_train, X_test, y_test)
    grid_search_classify(X_train, y_train, X_test, y_test)