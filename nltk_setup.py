import nltk
from textblob import download_corpora

# Download required corpora
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('omw-1.4')

# For TextBlob
download_corpora()
