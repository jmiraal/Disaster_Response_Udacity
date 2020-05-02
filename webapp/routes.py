# import basic librarys

import time
import numpy as np
import pandas as pd
import json
from pandas.api.types import CategoricalDtype

# import for visualizations
import plotly
from plotly.graph_objs import Bar

# import Flask to render web app
from flask import Flask
from flask import render_template, request, jsonify

# import joblib to load the machine learnign model
from sklearn.externals import joblib

# imports to load the data in the SQLite database
import sqlalchemy as db
from sqlalchemy import create_engine

# import custom transformers
import sys
sys.path.insert(1, './')
from transformers.my_transformers import *

from webapp import app
#app = Flask(__name__)


# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# read the model performance data
model_performance_df = pd.read_csv('./models/results.csv')
model_performance_df.label = model_performance_df.label.str.replace('_', ' ').str.title()

# dataframes with the performance when label have the value 1 and value 0
model_performance_df_0 = model_performance_df[model_performance_df.value == 0]
model_performance_df_1 = model_performance_df[model_performance_df.value == 1]

# sort dataframes with model performance in order to visualizations
model_performance_df_0 = model_performance_df_0.sort_values(by = 'f1-score', 
                                                            ascending = False)
                                                        
model_performance_df_1 = model_performance_df_1.sort_values(by = 'f1-score', 
                                                            ascending = False)


# load model
model = joblib.load("./models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    '''
    USAGE 
           index webpage displays cool visuals and receives user input text for model      
    OUTPUT
           graphs rendered with plotly for the webpage master.html         
    '''
    # extract data needed for visuals
    # extract data by genre of the messages
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract number of messabes by label
    label_counts = df.iloc[:,4:-2].sum().sort_values(ascending = True)
    label_names = list(label_counts.index)
    
    # extract number of messages by original language
    language_counts = df.groupby('lang').count()['message']
    language_names = list(language_counts.index)
    
    # extract mean lenght of messages by label
    mean_length = []
    mean = 0
    for label in label_names:
        mean = df[df[label] == 1]['len'].mean()
        mean_length.append(mean)

    
    # create visuals
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
        
        # graph number of messages by language of origin
        {
           'data': [
                Bar(
                    x=language_names,
                    y=language_counts,
                )
            ],

            'layout': {
                'title': 'Original Messages Language Count',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Language",
                    'dtick': 1
                }

            }
        },
        
        
        # model performance for value 1
        {
           'data': [
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1.precision,
                    yaxis='y',
                    offset = 0,
                    width = 0.23,
                    name = 'Precision',
                ),
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1.recall,
                    yaxis='y',
                    offset = 0.23,
                    width = 0.23,
                    name = 'Recall'
                ),
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1['f1-score'],
                    yaxis='y',
                    offset = 0.46,
                    width = 0.23,
                    name = 'F1-Score'
                ),
                Bar(
                    x=model_performance_df_1.label,
                    y=model_performance_df_1.support,
                    yaxis='y2',
                    offset = 0.69,
                    width = 0.23,
                    name = 'Count Messages'
                )
            ],

            'layout': {
                'title': 'Classification Model Perfomance (Value = 1)',
                'yaxis': {
                    'title': "%"
                },
                'yaxis2': {
                'title': 'Nº of Messages', 'overlaying': 'y', 'side': 'right'
                },
                'xaxis': {
                    'title': "Categories"
                },
                'margin': {
                     'u': 0,
                     'b': 150
                },
                'legend': {
		            'orientation': "v",
		            'x': 1.1,
                    'xanchor': 'left',
		            'y': 1,
		            'font': {
                    'size': 9
                    },
		        'borderwidth': 1
                },
            }
        },
        
        
        # model performance for value 0
        {
           'data': [
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0.precision,
                    yaxis='y',
                    offset = 0,
                    width = 0.23,
                    name = 'Precision',
                ),
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0.recall,
                    yaxis='y',
                    offset = 0.23,
                    width = 0.23,
                    name = 'Recall'
                ),
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0['f1-score'],
                    yaxis='y',
                    offset = 0.46,
                    width = 0.23,
                    name = 'F1-Score'
                ),
                Bar(
                    x=model_performance_df_0.label,
                    y=model_performance_df_0.support,
                    yaxis='y2',
                    offset = 0.69,
                    width = 0.23,
                    name = 'Count Messages'
                )
            ],

            'layout': {
                'title': 'Classification Model Perfomance (Value = 0)',
                'yaxis': {
                    'title': "%"
                },
                'yaxis2': {
                'title': 'Nº of Messages', 'overlaying': 'y', 'side': 'right'
                },
                'xaxis': {
                    'title': "Categories"
                },
                'margin': {
                     'u': 0,
                     'b': 150
                },
                'legend': {
		            'orientation': "v",
		            'x': 1.1,
                    'xanchor': 'left',
		            'y': 1,
		            'font': {
                    'size': 9
                    },
		        'borderwidth': 1
                },
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



@app.route('/go')
def go():
    '''
    USAGE 
           web page that handles user query and displays model results      
    OUTPUT
           graphs rendered with plotly for the webpage go.html         
    '''
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
