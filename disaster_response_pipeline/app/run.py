import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Box, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Tokenizes strings into cleaned and lemmatized list of words
    
    Input:
        text: string, text to be tokenized
    Output:
        clean_tokens: list, list of tokenized words
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Extracts data needed for the visuals, creates visuals and returns a plotly object which will be rendered in a HTML file
    
    Input:
        None
    Output:
        render_template object - visualizations in a HTML file
    '''
    
    # extract data needed for visuals
    
    # message count by genre
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending = False)
    genre_names = list(genre_counts.index)
    
    # average tweet size per genre, for tweets shorter than 800 characters
    message_length = df.assign(length = df.message.apply(lambda x: len(x)))[['genre', 'length']]
    message_length = message_length.loc[message_length.length <= 800]
    
    message_length_genres = message_length.genre.tolist()
    message_length_values = message_length.length.tolist()
    
    # tags' count - how often is each tag equal to 1, considering all messages?
    all_tags = df.iloc[:, 4:].sum().sort_values(ascending = False)
    
    all_tags_tag = all_tags.index.tolist()
    all_tags_value = all_tags.values.tolist()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        \
        {
            'data': [
                Box(
                    x=tweet_length_genres,
                    y=tweet_length_values,
                    boxpoints='all'
                   )
                    ],

            'layout': {
                'title': 'Distribution of Message Lenghts, for Messages shorter than 800 characters',
                      }
        },
        \
        {
            'data': [
                Bar(
                    x=all_tags_tag,
                    y=all_tags_value
                   )
                    ],

            'layout': {
                'title': 'Message Tags, Ordered by Appearances Count',
                'yaxis': {
                    'title': "Count"
                         },
                'xaxis': {
                    'title': "Tag"
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
    '''
    Takes user inputs, makes predictions about it using a Machine Learning model, and renders classification results in a HTML file
    
    Input:
        None
    Output:
        render_template object - model's classification results
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()