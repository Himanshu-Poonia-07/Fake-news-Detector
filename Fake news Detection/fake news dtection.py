# Fake News Detection Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load Dataset

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake
true["label"] = 1   # Real

# Combine datasets
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# 3. Data Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r"<.*?>+", '', text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub(r"\n", '', text)
    text = re.sub(r"\w*\d\w*", '', text)
    return text

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# 4. Split Data
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Convert Text to Numerical (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Prediction
y_pred = model.predict(X_test_vec)

# 8. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Test with Custom Input
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)

    if result[0] == 0:
        return "Fake News ❌"
    else:
        return "Real News ✅"

# Example
sample = input("\nEnter news text: ")
print("Prediction:", predict_news(sample))