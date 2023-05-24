# Librairie

import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle


# Charger le modèle pré-entraîné et le vecteur TF-IDF
with open('modele_sentiment.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vecteur_tfidf.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Application Dash
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Img(src='logo.png', style={'width': '200px', 'height': '200px'}),
    html.H1('Analyse de sentiment', style={'text-align': 'center'}),
    dcc.Textarea(id='input-text', placeholder='Entrez votre texte...', style={'width': '100%', 'height': 100}),
    html.Br(),
    html.Button('Analyser', id='analyze-button', n_clicks=0),
    html.Br(),
    dcc.Graph(id='sentiment-graph'),
    html.A('Contactez-moi: joelattouk@gmail.com ou au (+225 07 58 865 982)', href='mailto:joelattouk@gmail.com', style={'text-align': 'center', 'display': 'block', 'margin-top': '20px'})
])

# Inférence statistique
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    proba = model.predict_proba(text_vectorized)[0]
    sentiment_labels = ['Négatif', 'Positif', 'Neutre']
    sentiment_values = [proba[0], proba[1], proba[2]]
    return sentiment_labels, sentiment_values

# Rappel
@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_sentiment_graph(n_clicks, input_text):
    if n_clicks > 0:
        sentiment_labels, sentiment_values = predict_sentiment(input_text)

        max_value = max(sentiment_values) 

        
        colors = ['red' if value == max_value else 'blue' for value in sentiment_values]

        figure = {
            'data': [
                {'x': sentiment_labels, 'y': sentiment_values, 'type': 'bar', 'marker': {'color': colors}}
            ],
            'layout': {
                'title': 'Probabilité de prédiction du sentiment',
                'yaxis': {'title': 'Probabilité'},
                'xaxis': {'title': 'Sentiment'}
            }
        }
        return figure
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)