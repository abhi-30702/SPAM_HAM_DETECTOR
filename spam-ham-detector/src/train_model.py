import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "spam_ham_cleaned.csv")

df = pd.read_csv(DATA_PATH)

print(" Dataset loaded successfully")
print(df.head())
print("\nLabel distribution:\n", df["label"].value_counts())

df["cleaned_text"] = df["message"].apply(clean_text)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["cleaned_text"])
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, tfidf), f)

print(f"\n Model trained and saved at: {MODEL_PATH}")
