# Project Option 1
This document outlines what was accomplished for the Final project, Option1, for DATA641. Natural Language Processing (NLP) techniques were used to predict neuroticism based on social media posts and essays.

# Overview
There are two primary entrypoint scripts in the submission, personality_model.py and personality_and_essay.py.

## personality_model.py

NLP module that performs initial data analysis, baseline classifier, and classification improvement tasks.

This module provides the following capabilities:
  - Starting with a typical "raw" dataset
    - Personality dataset is not publicly available (i.e., not included in this repo) and has been de-identified
  - Concatenating relevant social media posts per user to create one or more corpora
  - Tokenizing text
  - Normalizing text
    - Optionally perform stopword list and filter punctuation
  - Performing Log-Likelihood Ratio (LLR) analysis to identify the "most interesting" n-grams in each dataset
  - Using this information to create a baseline classifier
  - Improve classification with exploration of other estimators 

## personality_and_essay.py

NLP module that trains on the personality dataset and evaluates performance of models against the essays dataset.

# Executing analysis

The following subsections provide guidance on executing each of the entrypoint scripts.

## Environment setup

Prior to executing the analysis scripts, make sure that you have the personality and essay datasets downloaded and extracted on a file system accessible to the scripts. If you do not, download the `project_data_and_resources.zip` as provided on the DATA641 Piazza forum and extract the zip file.

You will also need Python3 installed, preferably with an alias `python3` that points to the install.

## Command line options

The personality_model.py script provides a description of its command line options if you execute it using the `--help` flag. Some of these flags are derived from previous DATA641 assignments and will behave as expected. New command line options we have added are:
  * `--data-dir` - Sets directory containing downloaded data. Defaults to current working directory if not set explicitly via this option.
  * `--n-gram` - Sets the cardinality of n-grams in n-gram analysis. This is only used with sklearn tokenization and is thus not part of the analysis in the project write-up.
  * `--llr` - Performs 'top n' LLR results analysis, where n must be greater than 0. LLR is only compatible with non-sklearn (i.e., homebrew) tokenization approach and is part of the analysis in the project write-up.
  * `--filter-punc` - Optionally filter punctuation from text input (filtering not performed by default)
  * `--filter-urls` - Optionally filter URLs from text input (filtering not performed by default)
  * `--baseline` - Runs only the baseline classifier as specified in the write-up
  * `--random-forest` - Runs only the random forest classifier as specified in the write-up
  * `--grid-search` - Runs only the grid search classifiers as specified in the write-up
  * `--optimized-knn` - Runs only the optimized KNN classifier as specified in the write-up

## Suggested command lines

The following sections describe various commands you may consider executing. Some of these commands map to specific sections of the project write-up and are flagged as such.

__NOTE: These command lines assume that your installtion of Python3 is aliased to python3. You may need to replace the first argument of the command line to point to the proper Python if this is not the case.__

### Simple default execution

For a simple run through:

    python3 personality_model.py

This is the default invocation of the personality dataset analysis. The script will output results for a baseline logistic regression classifier, a GridSearch on Random Forest, Decision Tree, and Support Vector Machine estimators, as well as a kNN classifier.

### LLR Analysis

The following command line will execute LLR analysis, providing the top 25 results for each label:

    python3 personality_model.py --llr=25

This will perform data ingest, tokenization, and LLR analysis, providing the top 25 results.

### Baseline Classifier

To obtain the results shown for our baseline classifier in the "Baseline Classifier" section of the write-up, execute:

    python3 personality_model.py --baseline

This will print results to the terminal. Use the `--plot_metrics` flag to generate PNGs of the confusion matrix and ROC curve for the classifier.

### Random Forest Classifier

To obtain the results shown for our random forest classifier in the "Improving Classification - Random Forest Classifier" section of the write-up, execute:

    python3 personality_model.py --random-forest

This will print results to the terminal. Use the `--plot_metrics` flag to generate PNGs of the confusion matrix and ROC curve for the classifier.

### Random Forest, Decision Tree, and SVM Grid Search

To obtain the results shown for our grid search classifiers in the "Improving Classification - Tuning Hyperparameters" section of the write-up, execute:

    python3 personality_model.py --grid-search

This will print results to the terminal.

### KNN Classifier

To obtain the results shown in the "Improving Classification - K-Nearest Neighbors (KNN) Classifier" section of the write-up, execute:

    python3 personality_model.py --optimized-knn

This will print results to the terminal. Use the `--plot_metrics` flag to generate PNGs of the confusion matrix and ROC curve for the classifier.

### Essay classification
