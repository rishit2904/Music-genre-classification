# server.py
from flask import Flask, request, jsonify, send_from_directory,g
from flask_cors import CORS
import os
import numpy as np
import librosa
from keras.models import load_model
from pydub import AudioSegment
import uuid
import sqlite3 # For database
import requests # For calling external music APIs
import io # For handling audio bytes for live mode
import wave # For live mode

app = Flask(__name__, static_folder='.', static_url_path='') # Serve static files from root
CORS(app) # Simplest CORS for now

# --- Configuration ---
MODEL_PATH = 'music_genre_classifier.keras' # Your .keras model
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock'] # Must match training
SAMPLE_RATE = 22050
FIXED_TIME_FRAMES = 200 # Must match training
SAMPLES_PER_SEGMENT = (FIXED_TIME_FRAMES -1) * 512 + 1024 # From your train.py

DATABASE = 'tunetag_stats.db'
LASTFM_API_KEY = "8a9eea9fbc3a5a01e9f75cd209911386" # Get from https://www.last.fm/api/account/create
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# --- Load Model ---
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model '{MODEL_PATH}': {e}")
    model = None # Set model to None if loading fails

# --- Database Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # Access columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    if not os.path.exists(DATABASE):
        with app.app_context():
            db = get_db()
            with app.open_resource('schema.sql', mode='r') as f:
                db.cursor().executescript(f.read())
            db.commit()
        print("✅ Database initialized.")

# --- Feature Extraction (from your train.py, slightly adapted) ---
def extract_features(audio_bytes, sr=SAMPLE_RATE):
    try:
        # Load audio data from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)

        if len(y) < SAMPLES_PER_SEGMENT:
            pad_width = SAMPLES_PER_SEGMENT - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        else:
            y = y[:SAMPLES_PER_SEGMENT]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < FIXED_TIME_FRAMES:
            pad_width = FIXED_TIME_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :FIXED_TIME_FRAMES]

        return mel_db[..., np.newaxis] # Add channel dimension
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

# --- Utility for MP3/other to WAV ---
def convert_to_wav_bytes(file_storage):
    try:
        audio = AudioSegment.from_file(file_storage) # file_storage is a FileStorage object
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2) # Mono, 16-bit
        
        # Export to WAV in memory
        wav_bytes_io = io.BytesIO()
        audio.export(wav_bytes_io, format='wav')
        wav_bytes_io.seek(0) # Rewind to the beginning of the BytesIO object
        return wav_bytes_io.read() # Return bytes
    except Exception as e:
        print(f"❌ Audio conversion error: {e}")
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html') # Serve the main HTML file

@app.route('/predict', methods=['POST'], strict_slashes=False)
def predict_genre_route():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = file.filename
        print(f"📥 Received file for upload: {filename}")

        if filename.lower().endswith(('.mp3', '.ogg', '.flac')): # Add other formats if pydub supports them
            audio_bytes = convert_to_wav_bytes(file)
            if audio_bytes is None:
                return jsonify({'error': 'Audio conversion failed'}), 500
        elif filename.lower().endswith('.wav'):
            audio_bytes = file.read()
        else:
            return jsonify({'error': 'Unsupported file format. Use .wav, .mp3.'}), 400
        
        features = extract_features(audio_bytes, sr=SAMPLE_RATE)
        if features is None:
            return jsonify({'error': 'Could not extract features from audio'}), 500

        input_data = np.expand_dims(features, axis=0)
        predictions = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_genre = GENRES[predicted_index]
        confidence = float(predictions[0][predicted_index]) # Get confidence

        print(f"🎶 Predicted Genre (Upload): {predicted_genre} with confidence {confidence:.2f}")

        # Store prediction for stats
        try:
            db = get_db()
            db.execute('INSERT INTO predictions (genre, source) VALUES (?, ?)',
                       (predicted_genre, 'upload'))
            db.commit()
        except Exception as e:
            print(f"⚠️ DB Error storing prediction: {e}")


        return jsonify({'genre': predicted_genre, 'confidence': confidence})

    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/genre_songs/<genre_name>', methods=['GET'])
def get_genre_songs_route(genre_name):
    if not LASTFM_API_KEY or LASTFM_API_KEY == "YOUR_LASTFM_API_KEY":
        return jsonify({'error': 'Last.fm API key not configured on server.', 'tracks': []}), 503

    params = {
        'method': 'tag.gettoptracks',
        'tag': genre_name,
        'api_key': LASTFM_API_KEY,
        'format': 'json',
        'limit': 10 # Number of tracks
    }
    try:
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        tracks_data = []
        if 'tracks' in data and 'track' in data['tracks']:
            for track_item in data['tracks']['track']:
                image_url = ""
                if 'image' in track_item:
                    # Find largest image or a suitable one
                    for img_info in track_item['image']:
                        if img_info['size'] == 'large': # or extralarge
                            image_url = img_info['#text']
                            break
                        if img_info['size'] == 'medium': # fallback
                             image_url = img_info['#text']


                tracks_data.append({
                    'name': track_item['name'],
                    'artist': track_item['artist']['name'],
                    'url': track_item['url'],
                    'image_url': image_url
                })
        return jsonify({'genre': genre_name, 'tracks': tracks_data})
    except requests.exceptions.RequestException as e:
        print(f"❌ Last.fm API request error: {e}")
        return jsonify({'error': f'Could not fetch songs from Last.fm: {e}', 'tracks': []}), 500
    except Exception as e:
        print(f"❌ Error processing Last.fm data: {e}")
        return jsonify({'error': f'Error processing song data: {e}', 'tracks': []}), 500


@app.route('/stats', methods=['GET'])
def get_stats_route():
    try:
        db = get_db()
        cursor = db.execute('SELECT genre, COUNT(*) as count FROM predictions GROUP BY genre ORDER BY count DESC')
        stats = [{'genre': row['genre'], 'count': row['count']} for row in cursor.fetchall()]
        return jsonify(stats)
    except Exception as e:
        print(f"❌ DB Error fetching stats: {e}")
        return jsonify({'error': f'Could not fetch stats: {e}'}),500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)