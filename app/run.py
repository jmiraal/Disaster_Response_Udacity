import json
import plotly

import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('maxent_ne_chunker')
nltk.download('words')

import re
import time
import numpy as np
from pandas.api.types import CategoricalDtype
from matplotlib import pyplot as plt
import seaborn as sns
from plotly.graph_objs import Bar


from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
import sqlalchemy as db
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

model_performance_df = pd.read_csv('../models/results.csv')
model_performance_df.label = model_performance_df.label.str.replace('_', ' ').str.title()

model_performance_df_0 = model_performance_df[model_performance_df.value == 0]
model_performance_df_1 = model_performance_df[model_performance_df.value == 1]

model_performance_df_0 = model_performance_df_0.sort_values(by = 'f1-score', 
                                                            ascending = False)
                                                        
model_performance_df_1 = model_performance_df_1.sort_values(by = 'f1-score', 
                                                            ascending = False)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    label_counts = df.iloc[:,4:-1].sum().sort_values(ascending = True)
    label_names = list(label_counts.index)

    mean_length = []
    mean = 0
    for label in label_names:
        mean = df[df[label] == 1]['len'].mean()
        mean_length.append(mean)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # original graph number of messages by genre
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # graph number of messages by label
        {
           'data': [
                Bar(
                    x=label_counts,
                    y=[s.replace('_', ' ').title() for s in label_names],
                    orientation= 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Label',
                'height': 800,
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': "Count"
                },
                'margin': {
                     'l': 150,
                     'r': 0
                }
            }
        },
        
        # graph mean length message by label
        {
           'data': [
                Bar(
                    x=mean_length,
                    y=[s.replace('_', ' ').title() for s in label_names],
                    orientation= 'h'
                )
            ],

            'layout': {
                 'height': 800,
                'title': 'Mean Length of Messages by Label',
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': "Mean Length (chars)"
                },
                'margin': {
                     'l': 150,
                     'r': 0
                }

            }
        },
        
        
        # model performance for value 1
        {
           'data': [
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1.precision,
                    offset = 0,
                    width = 0.3,
                    name = 'Precision',
                ),
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1.recall,
                    offset = 0.3,
                    width = 0.3,
                    name = 'Recall'
                ),
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1['f1-score'],
                    offset = 0.6,
                    width = 0.3,
                    name = 'F1-Score'
                )
            ],

            'layout': {
                'title': 'Classification Model Perfomance',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'margin': {
                     'u': 0,
                     'b': 150
                }
            }
        },
        
        
        # model performance for value 0
        {
           'data': [
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0.precision,
                    offset = 0,
                    width = 0.3,
                    name = 'Precision',
                ),
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0.recall,
                    offset = 0.3,
                    width = 0.3,
                    name = 'Recall'
                ),
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0['f1-score'],
                    offset = 0.6,
                    width = 0.3,
                    name = 'F1-Score'
                )
            ],

            'layout': {
                'title': 'Classification Model Perfomance',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'margin': {
                     'u': 0,
                     'b': 150
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()