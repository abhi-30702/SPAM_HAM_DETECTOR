import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)
