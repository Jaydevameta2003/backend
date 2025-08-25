from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from langdetect import detect
import spacy
from textstat import flesch_reading_ease
from nrclex import NRCLex
import tweepy
import time
import nltk

# Ensure required corpora are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('movie_reviews')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Twitter bearer token
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAEch0wEAAAAA0PrR9DhsTme7LkzyBtYZMdWoSM0%3DmOmmROvQYMGNaVPWq5CtQ8AvxShrkNxYsS65N9GDqvwmb3tIrR'  # Replace this with your token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# In-memory cache
cache = {}

@app.route('/')
def home():
    return "✅ Cohere Twitter Local API Running"

@app.route('/user_tweets', methods=['GET'])
def get_user_tweets():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    current_time = time.time()
    if username in cache and current_time - cache[username]['timestamp'] < 900:
        return jsonify({'username': username, 'tweets': cache[username]['tweets']})

    try:
        user = client.get_user(username=username)
        user_id = user.data.id
        tweets = client.get_users_tweets(id=user_id, max_results=10)

        tweets_data = []
        if tweets.data:
            for tweet in tweets.data:
                tweets_data.append(analyze_tweet(tweet.text))

        cache[username] = {'timestamp': current_time, 'tweets': tweets_data}
        return jsonify({'username': username, 'tweets': tweets_data})

    except tweepy.errors.TweepyException as e:
        return jsonify({'error': str(e)}), 500

def analyze_tweet(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    keywords = list(set(blob.noun_phrases))

    doc = nlp(text)
    entities = list(set(ent.text for ent in doc.ents))

    emotion_obj = NRCLex(text)
    emotions = emotion_obj.raw_emotion_scores
    dominant_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

    try:
        language = detect(text)
    except:
        language = "unknown"

    word_count = len(text.split())
    readability = flesch_reading_ease(text)
    toxicity = max(0.0, -1 * polarity)
    summary = ' '.join(text.split()[:20]) + ('...' if word_count > 20 else '')

    return {
        'text': text,
        'summary': summary,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'keywords': keywords,
        'entities': entities,
        'emotion': dominant_emotion,
        'language': language,
        'word_count': word_count,
        'readability_score': readability,
        'toxicity_score': round(toxicity, 2)
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
