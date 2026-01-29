import pickle
import os
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "❌ Trained model not found.\n"
        "Please run: python src/train_model.py\n"
        "to generate model/spam_model.pkl"
    )

with open(MODEL_PATH, "rb") as f:
    model, tfidf = pickle.load(f)


def predict_message(message: str) -> int:
    message = clean_text(message)
    vector = tfidf.transform([message])
    return model.predict(vector)[0]
