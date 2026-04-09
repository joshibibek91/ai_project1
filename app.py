import streamlit as st
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- Setup --------------------
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# -------------------- Functions --------------------
def preprocess_news(news: str) -> str:
    news = news.lower()
    news = re.sub("[^a-zA-Z]", " ", news)
    words = news.split()
    
    words = [word for word in words if word not in STOPWORDS]
    words = [STEMMER.stem(word) for word in words]
    
    return " ".join(words)


@st.cache_resource
def load_model():
    with open("LogisticRegression.pickle", "rb") as file:
        return pickle.load(file)


# -------------------- App UI --------------------
st.title("News Classification App")
st.write("Classify news into categories using ML")

model = load_model()

news_input = st.text_area("Enter News Content")

# -------------------- Prediction --------------------
if st.button("Classify News"):
    if news_input.strip():
        cleaned_news = preprocess_news(news_input)
        news_category = model.predict([cleaned_news])[0]

        st.subheader("Predicted News Category:")
        st.write(news_category)
    else:
        st.warning("Please enter news content before classifying.")