# data641-final-project
Final project for DATA641. Uses Natural Language Processing (NLP) techniques to predict neuroticism based on social media posts and essays.

# Overview
In the main Python Notebook you will:
  - Starting with a typical "raw" dataset
    - We'll be using a personality dataset, this not a publicly available data set
  - Concatenating relevant social media posts per user to create one or more corpora
  - Tokenizing text
  - Normalizing text
    - Well use case folding and also a stopword list
  - Extracting potentially useful ngrams
  - Using this information to create a baseline classifier
  - Improve classification

**The files you will be working with:** 
  - wcpr_mypersonality.csv this will be the main data set work is done on
  - wcpr_essays.csv this will be a new dataset used to test the models
  - personality_model.py this is the main Python Notebook that will complete the tasks mentioned above
  - personality_and_essay.py this is the supplementary Python notebook that will train on the personality data set and test on the new essays data set

# What you need to do
   For a simple run through, open the personality_model.py notebook and run it. The notebook will execute and output results for a baseline logistic regression classifier, a GridSearch on Random Forest, Decision Tree, and Support Vector Machine estimators, as well as a kNN classifier. There are additional flags listed at the bottom of the notebook that will allow you to plot certain metrics as well as decide whether to filter on punctuation or urls'.  
