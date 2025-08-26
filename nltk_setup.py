import nltk
import textblob.download_corpora as tblob
import os

NLTK_DIR = "/opt/render/nltk_data"

def download_all():
    # Ensure directory exists
    os.makedirs(NLTK_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DIR)

    # Download textblob corpora
    tblob.download_corpora()

    # Explicit NLTK downloads
    for pkg in [
        "punkt", 
        "punkt_tab",  # 👈 THIS is the missing one
        "averaged_perceptron_tagger",
        "brown",
        "wordnet",
        "omw-1.4"
    ]:
        nltk.download(pkg, download_dir=NLTK_DIR)

if __name__ == "__main__":
    download_all()
