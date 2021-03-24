import sys

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    '''
    Loads data and prepares them to serve as models' input
    
    Input:
        database_filepath: string, path to SQLAlchemy database file
    Output:
        X: numpy array, one dimensional array which contains tweet messages
        Y: numpy array, two dimensional array with tweets' classifications/tags
        category_names: list, set of columns' names which represent tags
    '''
    # creating engine and reading df
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster', engine)
    
    # getting X and Y as arrays
    X = df.message.values
    Y = df.drop(['id', 'original', 'genre', 'message'], axis = 1).astype('int').values # all columns except the dropped ones
    
    # putting category names in a list:
    category_names = list(df.columns)[4: ]
        
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes strings into cleaned and lemmatized list of words
    
    Input:
        text: string, text to be tokenized
    Output:
        clean_tokens: list, list of tokenized words
    '''
    # removing special characters:
    text = re.sub('[^a-z0-9]', ' ', text)
    
    # removing stopwords and tokenizing:
    tokens = [w for w in word_tokenize(text) if w not in stopwords.words('english')]
    
    # instantiating lemmatizer, which will be used in the following for loop:
    lemmatizer = WordNetLemmatizer()
    
    # defining empty list, lopping and appending lemmatized tokens:
    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Sets up Machine Learning pipeline, using CountVectorizer, TfidfTransformer and MultiOutputClassifier (with KNeighborsClassifier)
    
    Input: 
        None
    Output:
        pipeline: sklearn model object, which may be used to fit and predict
    '''
    pipeline = Pipeline([
            ('vect',  CountVectorizer(tokenizer = tokenize)), 
            ('tdidf',  TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier()))
             ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model object, returning a classification report
    
    Input:
        model: sklearn model, ML model to be evaluated
        X_test: numpy array, one dimensional array which contains input values for the model
        Y_test: numpy array, two dimensional array which contains tags for the model's evaluation
        category_names: list, list of columns' names which will be evaluated
    Output:
        None
    '''
    # getting y_pred using model:
    Y_pred = model.predict(X_test)
    
    # using transposed versions of arrays in order to iterate over category columns:
    for i, Y_test_i in enumerate(Y_test.T):
        Y_pred_i = Y_pred.T[i]
        print('\nclassification report for class: ', category_names[i])
        print(classification_report(y_true = Y_test_i, y_pred = Y_pred_i))


def save_model(model, model_filepath):
    '''
    Saves the fitted model into a pickle (.pkl) file
    
    Input:
        model: sklearn model, model to be saved into pkl file
        model_filepath: string, path where model will be saved
    Output:
        None
    '''
    joblib.dump(model, model_filepath, compress = True)


def main():
    '''
    By using previously defined functions, reads data from SQLAlchemy database, treats text columns, fits a ML model to it and saves the model into a pickle file
    
    Input:
        None
    Output:
        None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()