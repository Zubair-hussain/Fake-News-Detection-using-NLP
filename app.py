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
            "engine": "google",
            "q": query,  # Clean query without site: OR
            "tbm": "nws",
            "num": 15,  # More results
            "gl": "us",  # Country for better news
            "hl": "en",  # Language
            "api_key": SERPAPI_KEY
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()  # Raise if 4xx/5xx (e.g., invalid key)
        data = response.json()
        
        # Optional debug (uncomment if needed)
        # with st.expander("Debug: Raw SerpAPI Response"):
        #     st.json(data)
        
        results = data.get("news_results", [])
        trusted_sources = {"cnn", "bbc", "reuters", "nytimes", "apnews", "the guardian", "npr", "al jazeera", "wsj", "bloomberg"}
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
    except requests.exceptions.HTTPError as e:
        st.error(f"SERPAPI HTTP Error: {e} – Check key/params")
        return []
    except requests.exceptions.Timeout:
        st.error("SERPAPI Timeout – Try again")
        return []
    except Exception as e:
        st.error(f"SERPAPI Unexpected Error: {e}")
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
st.set_page_config(page_title="TruthLens Pro", page_icon="🔍", layout="wide")

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
st.title("🔍 TruthLens Pro")
st.markdown('<p class="subtitle">AI-Powered Fake News Detection System</p>', unsafe_allow_html=True)

# Input Section
st.subheader("📝 Enter News Article")
article = st.text_area("Paste article text here:", height=200, placeholder="Enter the news article you want to verify...")

st.subheader("📄 Or Upload PDF")
pdf_file = st.file_uploader("Upload PDF document", type="pdf")

# Process PDF
text_input = article
if pdf_file:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(pdf_file)
        if pdf_text:
            text_input = pdf_text
            st.success("✅ PDF loaded successfully")
        else:
            st.error("❌ Failed to read PDF")

st.markdown("---")

if st.button("🚀 Analyze Now", type="primary"):
    if not text_input.strip():
        st.error("⚠️ Please enter text or upload a PDF")
    else:
        with st.spinner("🔍 Analyzing article..."):
            # 1. ML Model
            ml_result = ml_predict(text_input)
            
            # 2. Trusted News Search
            keywords = extract_keywords(text_input)
            trusted_articles = search_trusted_news(keywords)
            
            # 3. Gemini AI
            gemini_result = gemini_verify(text_input)
            
            # Final Scoring (accurate logic)
            ml_score = ml_result["confidence"] if ml_result["verdict"] == "REAL" else (100 - ml_result["confidence"])
            news_score = len(trusted_articles) * 20
            gemini_score = gemini_result["confidence"] if gemini_result["verdict"] == "REAL" else (100 - gemini_result["confidence"]) if gemini_result["verdict"] == "FAKE" else 50
            
            total_score = (ml_score * 0.4) + news_score + (gemini_score * 0.3)
            final_confidence = min(99, max(1, round(total_score / 2, 1)))  # Balanced to 0-99
            final_verdict = "REAL" if final_confidence >= 60 else "FAKE"
            
            # Save to history
            st.session_state.history.append({
                "Verdict": final_verdict,
                "ML Model": f"{ml_result['verdict']} ({ml_result['confidence']:.0f}%)",
                "Trusted Sources": len(trusted_articles),
                "Gemini AI": gemini_result["verdict"],
                "Confidence": f"{final_confidence:.1f}%"
            })

        st.markdown("---")
       
        # FINAL VERDICT
        if final_verdict == "REAL":
            st.success(f"### ✅ VERDICT: REAL NEWS ({final_confidence:.1f}% Confidence)")
            st.info("This article appears to be legitimate based on our analysis.")
            st.balloons()
        else:
            st.error(f"### ⚠️ VERDICT: FAKE NEWS ({final_confidence:.1f}% Risk)")
            st.warning("This article appears to be misleading or false.")
        st.markdown("---")
       
        # DETAILED METRICS
        st.subheader("📊 Detailed Analysis")
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.metric("🤖 ML Model", ml_result["verdict"], f"{ml_result['confidence']:.0f}%")
        with col2:
            st.metric("🌐 Trusted Sources", len(trusted_articles))
        with col3:
            st.metric("✨ Gemini AI", gemini_result["verdict"])

        # SOURCE ARTICLES WITH URLS
        if trusted_articles:
            st.markdown("---")
            st.subheader("📰 Trusted News Sources")
            st.info(f"Found {len(trusted_articles)} articles from verified outlets")
           
            for idx, article in enumerate(trusted_articles, 1):
                title = article.get('title', 'No title')
                link = article.get('link', '#')
                source = article.get('source', 'Unknown')
                date = article.get('date', 'No date')
               
                with st.expander(f"{idx}. {title[:80]}...", expanded=False):
                    st.markdown(f"**Source:** {source}")
                    st.markdown(f"**Date:** {date}")
                    st.markdown(f"**URL:** [{link}]({link})")
                    if 'snippet' in article:
                        st.markdown(f"**Preview:** {article['snippet']}")
        else:
            st.warning("⚠️ No trusted sources found for this article")

# HISTORY TABLE
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Analysis History")
    df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df, use_container_width=True, hide_index=True)

# FOOTER
st.markdown("---")
st.caption("© 2025 TruthLens Pro | Powered by Machine Learning, Gemini AI & Real-Time News Verification")
