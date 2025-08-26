import nltk
from textblob import download_corpora

def download_all():
    # Download default corpora needed by TextBlob
    download_corpora()

    # Explicitly download missing corpora
    nltk.download("punkt")
    nltk.download("punkt_tab")   # NEW: Fixes your error
    nltk.download("averaged_perceptron_tagger")
    nltk.download("brown")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

if __name__ == "__main__":
    download_all()
