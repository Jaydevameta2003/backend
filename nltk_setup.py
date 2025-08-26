import nltk
import textblob.download_corpora as tblob

def download_all():
    # Run textblob's downloader
    tblob.main()

    # Explicitly download required corpora
    nltk.download("punkt")
    nltk.download("punkt_tab")   # NEW: fixes your error
    nltk.download("averaged_perceptron_tagger")
    nltk.download("brown")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

if __name__ == "__main__":
    download_all()
