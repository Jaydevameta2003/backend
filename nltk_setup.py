import nltk
import textblob.download_corpora as tblob
import os

def download_all():
    # Make sure nltk data path exists
    nltk_data_dir = "/opt/render/nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # Run textblob's downloader
    tblob.main()

    # Explicitly download required corpora
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("punkt_tab", download_dir=nltk_data_dir)
    nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)
    nltk.download("brown", download_dir=nltk_data_dir)
    nltk.download("wordnet", download_dir=nltk_data_dir)
    nltk.download("omw-1.4", download_dir=nltk_data_dir)

if __name__ == "__main__":
    download_all()
