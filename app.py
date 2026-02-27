import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("cyberbullying.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_vec, y_train)

# UI
st.title("Cyberbullying Detection App")

user_input = st.text_input("Enter a comment")

if user_input:
    user_vec = vectorizer.transform([user_input])
    
    prediction = model.predict(user_vec)[0]
    probability = model.predict_proba(user_vec)[0]
    
    confidence = max(probability) * 100

    if prediction == 1:
        st.error(f"⚠ Cyberbullying Detected")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Not Cyberbullying")
        st.write(f"Confidence: {confidence:.2f}%")