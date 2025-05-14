import streamlit as st
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# ─── Load model and data ─────────────────────────────────────
model = load_model('chatbot_model.h5')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

lemmatizer = WordNetLemmatizer()

# ─── Preprocessing ────────────────────────────────────────────
def clean_up_sentence(sentence: str) -> list[str]:
    # Regex tokenizer: split on word boundaries, drop punctuation
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    # Lemmatize each token
    return [lemmatizer.lemmatize(tok) for tok in tokens]

def predict_class(sentence: str) -> np.ndarray:
    sentence_words = clean_up_sentence(sentence)
    # Build bag-of-words vector
    bag = [1 if w in sentence_words else 0 for w in vocab]
    return np.array([bag])

def chatbot_response(msg: str) -> str:
    bow_input = predict_class(msg)
    preds = model.predict(bow_input)[0]
    return classes[np.argmax(preds)]

# ─── Streamlit App ────────────────────────────────────────────
st.set_page_config(page_title="Student Assistant Chatbot", page_icon="🤖")
st.title("🎓 Student Assistant Chatbot")
st.markdown("Ask me anything about assignments, exams, syllabus, and more!")

if 'history' not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display chat history
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
