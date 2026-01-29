import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

raw_file_path = os.path.join(DATA_DIR, "spam-ham.csv")
clean_file_path = os.path.join(DATA_DIR, "spam_ham_cleaned.csv")

print("Loading dataset from:", raw_file_path)

df = pd.read_csv(
    raw_file_path,
    encoding="latin-1",
    engine="python"
)

print("Original columns:", df.columns)

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df.dropna(inplace=True)
df.drop_duplicates(subset="message", inplace=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(clean_file_path, index=False)

print("✅ Dataset cleaned successfully!")
print("Saved as:", clean_file_path)
print("Final dataset shape:", df.shape)
print(df.head())
