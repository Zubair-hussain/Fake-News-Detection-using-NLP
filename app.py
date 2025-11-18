import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download required NLTK data (only once)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# -------------------------------------------
# Load Model and Vectorizer with Error Handling
# -------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = pickle.load(open("logistic_model.pkl", "rb"))
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, vectorizer = load_model_and_vectorizer()

# -------------------------------------------
# Enhanced Preprocessing Function (Matches Training Pipeline)
# -------------------------------------------
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags (common in news/social text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'www\.[^ ]+', ' ', text)
    text = re.sub(r'@[\w]+', ' ', text)
    text = re.sub(r'#[\w]+', ' ', text)
    
    # Keep only alphabets (remove numbers, punctuation, special chars)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords, and stem
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# -------------------------------------------
# Streamlit App UI - Professional Design
# -------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main-header {font-size: 42px !important; color: #1E90FF; text-align: center; font-weight: bold;}
    .subtitle {font-size: 18px; color: #666; text-align: center; margin-bottom: 30px;}
    .footer {margin-top: 50px; text-align: center; color: #888; font-size: 14px;}
    .prediction-box {padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether a news article is <strong>Real</strong> or <strong>Fake</strong> using a trained Logistic Regression model (WELFake Dataset)</p>', unsafe_allow_html=True)

st.info("""
**How it works:**  
This model was trained on over 70,000 news articles using TF-IDF features and Logistic Regression. 
It analyzes linguistic patterns typical of fake vs real news.
""")

# Input area
st.markdown("### Paste the news article below:")
user_input = st.text_area(
    "",
    placeholder="Enter the full news article text here...",
    height=250,
    label_visibility="collapsed"
)

# Predict button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("Detect Fake News", type="primary", use_container_width=True)

if predict_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a news article to analyze.")
    else:
        with st.spinner("Analyzing the text..."):
            # Preprocess
            clean_text = preprocess(user_input)
            
            if len(clean_text.split()) < 5:
                st.warning("⚠️ The text is too short after preprocessing. Please provide a longer article.")
            else:
                # Transform and predict
                vectorized_text = vectorizer.transform([clean_text])
                prediction = model.predict(vectorized_text)[0]
                probability = model.predict_proba(vectorized_text)[0]
                
                confidence = max(probability) * 100
                real_prob = probability[1] * 100
                fake_prob = probability[0] * 100

                # Display Result
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction == 0:
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #ffebee; border: 3px solid #f44336;">
                        ❌ <strong>FAKE NEWS DETECTED</strong><br>
                        <span style="font-size:18px; color:#d32f2f;">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error(f"This article has characteristics commonly found in fake news.")
                else:
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #e8f5e9; border: 3px solid #4caf50;">
                        ✅ <strong>REAL NEWS</strong><br>
                        <span style="font-size:18px; color:#2e7d32;">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success(f"This article appears to be from a credible source.")

                # Show probability breakdown
                st.markdown("#### Confidence Breakdown")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Probability of Being Fake", f"{fake_prob:.1f}%")
                with col_b:
                    st.metric("Probability of Being Real", f"{real_prob:.1f}%")

# Footer
st.markdown("""
<hr style='border-top: 1px solid #eee;'>
<div class='footer'>
    Fake News Detector • Powered by Logistic Regression + TF-IDF • Trained on WELFake Dataset<br>
    Note: This is an AI tool for reference only. Always verify news from multiple trusted sources.
</div>
""", unsafe_allow_html=True)
