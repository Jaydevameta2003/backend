from flask import Flask, request, jsonify
from flask_cors import CORS
import cohere
from textblob import TextBlob
from langdetect import detect
import spacy
from textstat import flesch_reading_ease
from nrclex import NRCLex
from newspaper import Article
import os

app = Flask(__name__)
CORS(app)

# ✅ Use environment variable for API key (never hardcode secrets!)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ Missing COHERE_API_KEY environment variable")
co = cohere.Client(COHERE_API_KEY)

# ✅ Safe spaCy model load
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return "✅ Flask API for News Analysis is running!"

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    data = request.get_json()
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        # 1. Extract article
        article = Article(url)
        article.download()
        article.parse()
        user_text = article.text

        if not user_text.strip():
            return jsonify({'error': 'Failed to extract content from URL'}), 500

        # 2. Summarization with Cohere
        prompt = f"Summarize the following news article: {user_text}"
        summary_response = co.generate(model='command', prompt=prompt, max_tokens=200)
        summary = summary_response.generations[0].text.strip()

        # 3. Sentiment
        blob = TextBlob(summary)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # 4. Keywords (noun phrases)
        keywords = list(set(blob.noun_phrases))

        # 5. Named Entities
        doc = nlp(user_text)
        entities = list(set((ent.text, ent.label_) for ent in doc.ents))

        # 6. Emotion detection
        emotion_obj = NRCLex(user_text)
        emotions = emotion_obj.raw_emotion_scores
        dominant_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

        # 7. Language
        try:
            language = detect(user_text)
        except:
            language = "unknown"

        # 8. Stats
        word_count = len(user_text.split())
        readability_score = flesch_reading_ease(user_text)
        toxicity_score = max(0.0, -1 * polarity)

        return jsonify({
            'title': article.title,
            'source': article.source_url,
            'summary': summary,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keywords': keywords,
            'entities': entities,
            'emotion': dominant_emotion,
            'language': language,
            'word_count': word_count,
            'readability_score': readability_score,
            'toxicity_score': round(toxicity_score, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
