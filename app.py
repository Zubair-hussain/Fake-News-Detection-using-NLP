import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai
import spacy
from spacy.cli import download as spacy_download

# ==================== KEYS (Local + Cloud Safe) ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# Override with Streamlit secrets if deployed
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

# ==================== DOWNLOAD & LOAD SPACY MODEL (FIXED!) ====================
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        with st.spinner(f"Downloading spaCy model {model_name}... (first run only)"):
            spacy_download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model()

# ==================== NLTK & GEMINI SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== LOAD ML MODEL ====================
@st.cache_resource
def load_ml_model():
    try:
        model = pickle.load(open("logistic_model.pkl", "rb"))
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        st.error("Model files not found! Upload logistic_model.pkl and tfidf_vectorizer.pkl")
        st.stop()

model, vectorizer = load_ml_model()

# ==================== PREPROCESSING ====================
def preprocess(text):
    text = re.sub(r'http[s]?://[^\s]+|www\.[^\s]+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_claim(text):
    doc = nlp(text[:1000])
    entities = [e.text for e in doc.ents if e.label_ in ["PERSON", "ORG", "GPE", "EVENT", "DATE"]]
    claim = " ".join(set(entities[:6]))
    return claim if len(claim) > 10 else "breaking news fact check"

# ==================== SERPAPI DIRECT CALL ====================
@st.cache_data(ttl=600)
def google_news_search(query):
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "tbm": "nws", "num": 10, "api_key": SERPAPI_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted = ["cnn", "bbc", "nytimes", "reuters", "apnews", "theguardian", "npr"]
        credible = sum(1 for r in results if any(t in r.get("source", "").lower() for t in trusted))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

# ==================== GEMINI FACT CHECK ====================
def gemini_check(text):
    prompt = f"""Fact-check this news article and answer in this exact format:

VERDICT: REAL / FAKE / SATIRE / MISLEADING
REASON: (max 12 words)

Article:
{text[:4800]}"""
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        verdict = "UNKNOWN"
        reason = "No reason"
        for line in text.split("\n"):
            if "VERDICT:" in line:
                verdict = line.split("VERDICT:")[1].strip().split()[0]
            if "REASON:" in line:
                reason = line.split("REASON:")[1].strip()
        return {"verdict": verdict.upper(), "reason": reason[:80]}
    except:
        return {"verdict": "ERROR", "reason": "Gemini unavailable"}

# ==================== UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="centered")

st.markdown("<h1 style='text-align:center; color:#1E40AF; font-size:50px;'>TruthLens Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:20px; color:#555;'>ML + Google News + Gemini AI • Real-Time Fake News Detection</p>", unsafe_allow_html=True)
st.markdown("---")

article = st.text_area("Paste the full news article here:", height=300)

if st.button("Analyze Now", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Please paste a news article.")
    else:
        with st.spinner("Running 3-layer verification..."):
            # 1. ML
            clean = preprocess(article)
            pred = model.predict(vectorizer.transform([clean]))[0]
            prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100

            # 2. Google News
            claim = extract_claim(article)
            news = google_news_search(claim)

            # 3. Gemini
            gemini = gemini_check(article)

            # Final Score
            score = 0
            if pred == 1: score += 35
            score += news["credible"] * 12
            if "REAL" in gemini["verdict"]: score += 30
            elif "FAKE" in gemini["verdict"]: score -= 40

            confidence = max(10, min(99, score))

        # === RESULT ===
        if confidence >= 70:
            st.balloons()
            st.success(f"REAL NEWS • Confidence: {confidence:.1f}%")
        elif confidence <= 40:
            st.error(f"FAKE OR MISLEADING • Fake Confidence: {100-confidence:.1f}%")
        else:
            st.warning(f"UNCERTAIN • Score: {confidence:.1f}%")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("ML Model", "Real" if pred==1 else "Fake", f"{prob:.0f}%")
        with col2: st.metric("Trusted Sources", news["credible"])
        with col3: st.metric("Gemini AI", gemini["verdict"])

        st.info(f"Gemini: {gemini['reason']}")

        if news["results"]:
            st.success("Top Matching Articles:")
            for r in news["results"][:3]:
                st.markdown(f"• [{r.get('title','No title')}]({r.get('link','#')}) — _{r.get('source','')}_")

st.caption("TruthLens Pro © 2025 • Always verify from original sources")
