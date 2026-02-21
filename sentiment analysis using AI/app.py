import streamlit as st
import nltk
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download NLTK data (only the first time)
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------
# Text Preprocessing Function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned)

# -------------------------
# Train a Simple Model (or Load if Saved)
# -------------------------
try:
    # Try loading saved model and vectorizer
    model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
except:
    # If not found, train a new one
    sample_data = [
        ("I love this movie, it was fantastic!", "positive"),
        ("The food was terrible and cold", "negative"),
        ("It was okay, not the best but fine", "neutral"),
        ("I really enjoyed the trip!", "positive"),
        ("I hate this product", "negative"),
        ("The service was average", "neutral"),
        ("Amazing performance and great story", "positive"),
        ("Not worth the money", "negative"),
        ("It’s fine, nothing special", "neutral")
    ]
    texts, labels = zip(*sample_data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([clean_text(t) for t in texts])

    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    y = np.array([label_map[l] for l in labels])

    model = MultinomialNB()
    model.fit(X, y)

    # Save for later use
    import os
    os.makedirs('model', exist_ok=True)
    pickle.dump(model, open('model/sentiment_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 AI Sentiment Analysis App")
st.write("Analyze sentiment of a review or tweet — Positive, Negative, or Neutral!")

user_input = st.text_area("Enter your review or tweet:", placeholder="Type something like 'I love this product!'")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ Sentiment: **Positive** 😄")
        elif prediction == -1:
            st.error("❌ Sentiment: **Negative** 😠")
        else:
            st.info("😐 Sentiment: **Neutral**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")
