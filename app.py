import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------------------------------
# Load model and vectorizer
# -------------------------------------------
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# -------------------------------------------
# Preprocessing Function
# -------------------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove non-alphabetic
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([ps.stem(word) for word in words if word not in stop_words])

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detection App")
st.markdown("Classify a news article as **Fake** or **Real** using Logistic Regression trained on the WELFake dataset.")

# Input text from user
user_input = st.text_area("Enter news article text below:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some news content.")
    else:
        clean_text = preprocess(user_input)
        transformed = vectorizer.transform([clean_text])
        prediction = model.predict(transformed)[0]
        
        # Output result
        st.subheader("Prediction Result:")
        if prediction == 0:
            st.error("Fake News Detected")
        else:
            st.success("Real News Detected")
