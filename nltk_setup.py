import nltk

def download_corpora():
    nltk.download("brown")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

if __name__ == "__main__":
    download_corpora()
