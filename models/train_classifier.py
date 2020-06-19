#import libraries
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
     '''
    Load data, transform DataFrame, get X, Y and name of feature columns for score results
    
    INPUT:
    engine - create data base
    df - read a table of engine
    
    OUTPUT:
    X - an array with columns messages from df
    Y - a new dataframe that has the following characteristics:
            1. has all comns except for 'id', 'message', 'original', and 'genre'.
            2. has a column 'related' cleaned from values 2.
    categiries - a list of columns names in Y.    
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("df", engine)  
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    '''
    Clean, normalize, tokenize, lemmatize a text
    
    INPUT:
    text - a string, in this case messages
    
    OUTPUT:
    clean_tokens - an array with a text that has the following characteristics:
            1. has no punctuation marks
            2. splited into sequence of words
            3. cleaned from stopwords
            4. lemmatized
            5. all letters are in low case
    '''
    
    #normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    tokens = word_tokenize(text)
    
    #stop_words
    my_stopwords=stopwords.words('english')
    tokens = [word for word in tokens if word not in my_stopwords]
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
        
    '''
    Create a pipeline and parameters for a grid search model
    
    OUTPUT:
    cv - model that:
            1. defines an improved pipeline 
            2. sets parameters for estimators
            3. defins a grid search model with the pipeline and parameters
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        ))
    ])

    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

model = improved_model()

def evaluate_model():
    y_pred_pd = pd.DataFrame(y_pred, columns = categories)
    for column in categories:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_pd[column]))


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open('classifier.pkl', 'wb'))


def main():
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
        evaluate_model()

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
