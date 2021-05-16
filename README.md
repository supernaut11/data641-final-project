# Project Option 1
This document outlines what was accomplished for the Final project, Option1, for DATA641. Natural Language Processing (NLP) techniques were used to predict neuroticism based on social media posts and essays.

# Overview
In the main Python Notebook you will be:
  - Starting with a typical "raw" dataset
    - We'll be using a personality dataset, this not a publicly available data set and has been de-identified
  - Concatenating relevant social media posts per user to create one or more corpora
  - Tokenizing text
  - Normalizing text
    - We'll use a stopword list and filter punctuation
  - Extracting potentially useful ngrams
  - Using this information to create a baseline classifier
  - Improve classification with exploration of other estimators 

**The files you will be working with:** 
  - `mypersonality` this will be the main data set the notebook works with 
  - `essays` this will be a new dataset used to test the models
  - `personality_model.py` this is the main Python Notebook that will complete the tasks mentioned above
  - `personality_and_essay.py` this is the supplementary Python notebook that will train on the personality data set and test on the new essays data set

# What you need to do
   For a simple run through, open the `personality_model.py` notebook and run it. The notebook will execute and output results for a baseline logistic regression classifier, a GridSearch on Random Forest, Decision Tree, and Support Vector Machine estimators, as well as a kNN classifier. There are additional flags listed at the bottom of the notebook that will allow you to plot certain metrics as well as decide whether to filter on punctuation or urls'.  
