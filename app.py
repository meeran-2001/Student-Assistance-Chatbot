# app.py

import nltk
# Ensure required NLTK data is present
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import streamlit as st
import numpy as np
import re
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# Load model and supporting data
model = tf.keras.models.load_model('chatbot_model.h5')
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

lemmatizer = WordNetLemmatizer()

def preprocess_and_predict(sentence):
    # Tokenize & lemmatize
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    # Create bag‑of‑words
    bag = np.array([1 if w in lemmas else 0 for w in words]).reshape(1, -1)
    # Predict and return the class label
    preds = model.predict(bag)
    idx = np.argmax(preds[0])
    return classes[idx]

# Streamlit UI
st.set_page_config(page_title="Student Assistant Chatbot", layout="centered")
st.title("🎓 Student Assistant Chatbot")
st.markdown("Ask me anything about your academics!")

user_input = st.text_input("You:", "")
if user_input:
    response = preprocess_and_predict(user_input)
    st.text_area("Chatbot:", value=response, height=150)
