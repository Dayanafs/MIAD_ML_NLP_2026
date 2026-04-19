from flask import Flask
from flask_restx import Api, Resource
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
api = Api(app, version='1.0', title='Spotify Popularity API',
          description='API para predecir popularidad de canciones')

ns = api.namespace('predict', description='Predicciones')

parser = api.parser()
parser.add_argument('danceability', type=float, required=True, location='args')
parser.add_argument('energy', type=float, required=True, location='args')
parser.add_argument('key', type=int, required=True, location='args')
parser.add_argument('loudness', type=float, required=True, location='args')
parser.add_argument('mode', type=int, required=True, location='args')
parser.add_argument('speechiness', type=float, required=True, location='args')
parser.add_argument('acousticness', type=float, required=True, location='args')
parser.add_argument('instrumentalness', type=float, required=True, location='args')
parser.add_argument('liveness', type=float, required=True, location='args')
parser.add_argument('valence', type=float, required=True, location='args')
parser.add_argument('tempo', type=float, required=True, location='args')
parser.add_argument('duration_ms', type=int, required=True, location='args')
parser.add_argument('time_signature', type=int, required=True, location='args')
parser.add_argument('explicit', type=str, required=True, location='args')
parser.add_argument('track_genre', type=str, required=True, location='args')
parser.add_argument('artist_pop_mean', type=float, required=True, location='args')

# Cargar modelo y encoder
modelo = joblib.load('modelo_spotify.pkl')
le = joblib.load('label_encoder.pkl')

@ns.route('/')
class Prediction(Resource):
    @api.doc(parser=parser)
    def get(self):
        args = parser.parse_args()
        
        # Codificar track_genre
        genre = args['track_genre']
        if genre in le.classes_:
            genre_encoded = int(le.transform([genre])[0])
        else:
            genre_encoded = -1
        
        X = pd.DataFrame([{
            'duration_ms': args['duration_ms'],
            'explicit': 1 if str(args['explicit']).lower() == 'true' else 0,
            'danceability': args['danceability'],
            'energy': args['energy'],
            'key': args['key'],
            'loudness': args['loudness'],
            'mode': args['mode'],
            'speechiness': args['speechiness'],
            'acousticness': args['acousticness'],
            'instrumentalness': args['instrumentalness'],
            'liveness': args['liveness'],
            'valence': args['valence'],
            'tempo': args['tempo'],
            'time_signature': args['time_signature'],
            'track_genre': genre_encoded,
            'artist_pop_mean': args['artist_pop_mean']
        }])
        
        pred = modelo.predict(X)[0]
        pred = float(np.clip(pred, 0, 100))
        
        return {'popularity_predicted': round(pred, 2)}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)