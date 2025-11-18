import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai
import PyPDF2
import pandas as pd
from io import BytesIO

# ==================== KEYS ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyBkq0VS1Jz8LCiBG9UzUhxAgUOruVffu2s"

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Safe Gemini setup
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"temperature": 0.1, "max_output_tokens": 20},
        safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in genai.types.HarmCategory]
    )
except:
    gemini_model = None

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== HELPERS ====================
def preprocess(text):
    text = re.sub(r'http\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_keywords(text):
    words = re.findall(r'\b[a-z]{5,20}\b', text.lower())
    filtered = [w for w in words if w not in stop_words][:15]
    return " ".join(filtered) or "breaking news"

def read_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text[:15000]
    except:
        return None

def ml_predict(text):
    clean = preprocess(text)
    pred = model.predict(vectorizer.transform([clean]))[0]
    prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100
    verdict = "REAL" if pred == 1 else "FAKE"
    return {"verdict": verdict, "confidence": round(prob, 1)}

@st.cache_data(ttl=900)
def search_trusted_news(query):
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google", "q": query, "tbm": "nws", "num": 10,
            "api_key": SERPAPI_KEY
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        results = data.get("news_results", [])[:6]
        
        trusted_sources = {"cnn", "bbc", "reuters", "nytimes", "apnews", "theguardian", "npr", "aljazeera", "wsj", "bloomberg"}
        trusted_hits = []
        for item in results:
            source = item.get("source", "").lower()
            if any(trusted in source for trusted in trusted_sources):
                trusted_hits.append({
                    "title": item.get("title", "No title"),
                    "link": item.get("link", "#"),
                    "source": item.get("source", "Unknown"),
                    "date": item.get("date", "No date"),
                    "snippet": item.get("snippet", "")
                })
        return trusted_hits
    except:
        return []

def gemini_verify(text):
    if not gemini_model:
        return {"verdict": "UNKNOWN", "confidence": 0}
    
    prompt = f"""Fact-check this news article. Answer only with one word: REAL or FAKE

Article: {text[:4500]}"""
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().upper()
        if "REAL" in result:
            return {"verdict": "REAL", "confidence": 90}
        else:
            return {"verdict": "FAKE", "confidence": 90}
    except:
        return {"verdict": "UNKNOWN", "confidence": 0}

# ==================== UI - CLEAN & PROFESSIONAL ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="wide")

st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto; padding: 20px;}
    .stButton > button {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white; font-weight: 600; padding: 12px;
        border-radius: 12px; border: none; width: 100%;
    }
    .stButton > button:hover {background: linear-gradient(90deg, #1e3a8a, #2563eb);}
    h1 {text-align: center; color: #1e40af; font-size: 48px; margin-bottom: 0;}
    .subtitle {text-align: center; color: #64748b; font-size: 20px; margin-bottom: 30px;}
    .metric-card {background: #f8fafc; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Header
st.title("TruthLens Pro")
st.markdown('<p class="subtitle">Advanced AI-Powered Fake News Detection System</p>', unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    article = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste any news article here to verify its authenticity..."
    )
with col2:
    st.write("### Or Upload PDF")
    pdf_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

# Process PDF
text_input = article
if pdf_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = read_pdf(pdf_file)
        if pdf_text:
            text_input = pdf_text
            st.success("PDF loaded successfully!")
        else:
            st.error("Could not read PDF")

st.markdown("---")

if st.button("Analyze Truth", type="primary"):
    if not text_input.strip():
        st.error("Please enter text or upload a PDF")
    else:
        with st.spinner("Analyzing with ML Model, Trusted Sources & Gemini AI..."):
            # 1. ML Model
            ml_result = ml_predict(text_input)
            
            # 2. Trusted News Search
            keywords = extract_keywords(text_input)
            trusted_articles = search_trusted_news(keywords)
            
            # 3. Gemini AI
            gemini_result = gemini_verify(text_input)
            
            # Final Scoring
            ml_score = ml_result["confidence"] if ml_result["verdict"] == "REAL" else -ml_result["confidence"]
            news_score = len(trusted_articles) * 30
            gemini_score = gemini_result["confidence"] if gemini_result["verdict"] == "REAL" else -gemini_result["confidence"] if gemini_result["verdict"] == "FAKE" else 0
            
            total_score = ml_score * 0.4 + news_score + gemini_score * 0.6
            final_confidence = min(99, max(1, round((total_score + 300) / 6, 1)))  # Normalize to 0-99
            final_verdict = "REAL" if final_confidence >= 60 else "FAKE"
            
            # Save to history
            st.session_state.history.append({
                "Verdict": final_verdict,
                "ML Model": f"{ml_result['verdict']} ({ml_result['confidence']:.0f}%)",
                "Trusted Sources": len(trusted_articles),
                "Gemini AI": gemini_result["verdict"],
                "Confidence": f"{final_confidence:.1f}%"
            })

        # === FINAL RESULT ===
        st.markdown("---")
        if final_verdict == "REAL":
            st.success(f"### VERDICT: REAL NEWS")
            st.markdown(f"**Confidence: {final_confidence:.1f}%** – This article is likely authentic")
            st.balloons()
        else:
            st.error(f"### VERDICT: FAKE / MISLEADING")
            st.markdown(f"**Risk Level: {final_confidence:.1f}%** – High chance this is false")

        # Detailed Metrics
        st.markdown("### Detailed Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Model", ml_result["verdict"], f"{ml_result['confidence']:.0f}%")
        with col2:
            st.metric("Trusted Sources Found", len(trusted_articles))
        with col3:
            st.metric("Gemini AI Verdict", gemini_result["verdict"])

        # Show Trusted Sources
        if trusted_articles:
            st.markdown("### Verified Sources Reporting This")
            for i, article in enumerate(trusted_articles[:5], 1):
                with st.expander(f"{i}. {article['title'][:100]}"):
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Link:** [{article['link']}]({article['link']})")
                    if article['snippet']:
                        st.write(f"**Preview:** {article['snippet'][:200]}...")
        else:
            st.warning("No trusted news outlets are reporting this story")

# History Table
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Recent Analysis History")
    df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("© 2025 TruthLens Pro • Powered by Machine Learning • Gemini AI • Real-time News Verification")
