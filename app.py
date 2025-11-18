import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
import google.generativeai as genai
import requests

# ==================== YOUR KEYS (Safe for local + Streamlit Cloud) ====================
# For local testing (works when you run `streamlit run app.py`)
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# For Streamlit Cloud — automatically overrides the above
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass  # Keep local keys if no secrets

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ==================== PREPROCESSING ====================
def preprocess(text):
    text = re.sub(r'http[s]?://[^\s]+|www\.[^\s]+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_key_claim(text):
    doc = nlp(text[:1200])
    entities = [e.text for e in doc.ents if e.label_ in ["PERSON", "ORG", "GPE", "EVENT", "DATE"]]
    return " ".join(set(entities[:7])) or "breaking news fact check"

# ==================== SERPAPI VIA DIRECT API (No serpapi package needed) ====================
@st.cache_data(ttl=600)
def check_google_news(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "tbm": "nws",
        "num": 10,
        "api_key": SERPAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted_domains = ["cnn", "bbc", "nytimes", "reuters", "apnews", "theguardian", "npr"]
        credible = sum(1 for item in results if any(dom in item.get("source", "").lower() for dom in trusted_domains))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

# ==================== GEMINI FACT-CHECK ====================
def gemini_verify(text):
    prompt = f"""Fact-check this news article. Respond in this exact format:

VERDICT: REAL / FAKE / SATIRE / MISLEADING
REASON: (max 10 words)

Article:
{text[:5000]}"""

    try:
        response = gemini_model.generate_content(prompt)
        text_out = response.text.strip()
        verdict = "UNKNOWN"
        reason = "Analysis failed"
        for line in text_out.split("\n"):
            if "VERDICT:" in line:
                verdict = line.split("VERDICT:")[1].strip().split()[0]
            if "REASON:" in line:
                reason = line.split("REASON:")[1].strip()
        return {"verdict": verdict.upper(), "reason": reason[:70]}
    except:
        return {"verdict": "ERROR", "reason": "Gemini unavailable"}

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying-glass", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1E40AF;'>TruthLens Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Advanced Fake News Detector • ML + Google News + Gemini AI</p>", unsafe_allow_html=True)
st.markdown("---")

user_text = st.text_area("Paste the full news article below:", height=280, placeholder="Enter news content here...")

if st.button("Detect Fake News", type="primary", use_container_width=True):
    if not user_text.strip():
        st.error("Please enter a news article.")
    else:
        with st.spinner("Analyzing with 3 powerful layers..."):
            # 1. ML Model
            cleaned = preprocess(user_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0].max() * 100

            # 2. Google News Search
            claim = extract_key_claim(user_text)
            news_check = check_google_news(claim)

            # 3. Gemini AI
            gemini_result = gemini_verify(user_text)

            # Final Confidence Score
            score = 0
            if prediction == 1: score += 30
            score += news_check["credible"] * 15
            if "REAL" in gemini_result["verdict"]: score += 25
            elif "FAKE" in gemini_result["verdict"]: score -= 35

            final_confidence = max(5, min(99, score))

        # === DISPLAY RESULT ===
        if final_confidence >= 65:
            st.success(f"REAL NEWS DETECTED | Confidence: {final_confidence:.1f}%")
        elif final_confidence <= 40:
            st.error(f"FAKE / MISLEADING NEWS | Fake Confidence: {100-final_confidence:.1f}%")
        else:
            st.warning(f"UNCERTAIN – Needs Manual Verification | Score: {final_confidence:.1f}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Model", "Real" if prediction == 1 else "Fake", f"{probability:.0f}%")
        with col2:
            st.metric("Trusted Sources Found", news_check["credible"])
        with col3:
            st.metric("Gemini AI", gemini_result["verdict"])

        if gemini_result["reason"]:
            st.info(f"Gemini Reasoning: {gemini_result['reason']}")

        if news_check["results"]:
            st.success("Top Matching Articles from Trusted Sources:")
            for item in news_check["results"][:3]:
                title = item.get("title", "No title")
                link = item.get("link", "#")
                source = item.get("source", "Unknown")
                st.markdown(f"• [{title}]({link}) — _{source}_")

st.markdown("---")
st.caption("TruthLens Pro © 2025 | Powered by Logistic Regression + SerpAPI + Google Gemini | Always verify from original sources.")
