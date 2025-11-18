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

# KEYS
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyBkq0VS1Jz8LCiBG9UzUhxAgUOruVffu2s"

# SETUP
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash",
    generation_config={"temperature": 0, "max_output_tokens": 20},
    safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in genai.types.HarmCategory])

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer
model, vectorizer = load_model()
if "history" not in st.session_state: st.session_state.history = []

# HELPERS
def preprocess(t): 
    return " ".join([ps.stem(w) for w in re.sub(r'[^a-z\s]', ' ', re.sub(r'http\S+', ' ', t.lower())).split() if w not in stop_words and len(w)>2])

def keywords(t): 
    return " ".join([w for w in re.findall(r'\b[a-z]{5,20}\b', t.lower()) if w not in stop_words][:12]) or "news"

def read_pdf(f): 
    try: 
        r = PyPDF2.PdfReader(BytesIO(f.read()))
        return "".join(p.extract_text() or "" for p in r.pages)[:15000]
    except: 
        return None

def ml_check(t):
    p = preprocess(t)
    pred = model.predict(vectorizer.transform([p]))[0]
    prob = model.predict_proba(vectorizer.transform([p]))[0].max() * 100
    return {"verdict": "REAL" if pred==1 else "FAKE", "conf": round(prob,1)}

@st.cache_data(ttl=900)
def news_check(q):
    try:
        r = requests.get("https://serpapi.com/search.json", params={"engine":"google","q":q,"tbm":"nws","num":10,"api_key":SERPAPI_KEY}, timeout=15).json()
        hits = [x for x in r.get("news_results", [])[:6] if any(d in x.get("source","").lower() for d in ["cnn","bbc","reuters","nytimes","apnews","guardian","npr","aljazeera","wsj"])]
        return {"articles": hits, "count": len(hits)}
    except: 
        return {"articles": [], "count": 0}

def gemini_check(t):
    try:
        resp = gemini_model.generate_content(f"Real or fake news? Answer only REAL or FAKE\n\n{' '.join(t.split()[:500])}")
        v = "REAL" if "REAL" in resp.text.strip().upper() else "FAKE"
        return {"verdict": v, "conf": 90}
    except: 
        return {"verdict": "FAKE", "conf": 0}

# PAGE CONFIG
st.set_page_config(page_title="TruthLens Pro", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    .stButton > button {width: 100%; background: #1f77b4; color: white; font-weight: 600; padding: 0.75rem;}
    .stButton > button:hover {background: #1557a0;}
    h1 {text-align: center; color: #1f77b4; margin-bottom: 0.5rem;}
    .subtitle {text-align: center; color: #666; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🔍 TruthLens Pro")
st.markdown('<p class="subtitle">AI-Powered Fake News Detection System</p>', unsafe_allow_html=True)

# INPUT SECTION
st.subheader("📝 Enter News Article")
article = st.text_area("Paste article text here:", height=200, placeholder="Enter the news article you want to verify...")

st.subheader("📄 Or Upload PDF")
pdf = st.file_uploader("Upload PDF document", type="pdf")

text = article
if pdf:
    with st.spinner("Reading PDF..."):
        txt = read_pdf(pdf)
        if txt: 
            text = txt
            st.success("✅ PDF loaded successfully")
        else:
            st.error("❌ Failed to read PDF")

# ANALYZE BUTTON
st.markdown("---")
if st.button("🚀 Analyze Now", type="primary"):
    if not text.strip():
        st.error("⚠️ Please enter text or upload a PDF")
    else:
        with st.spinner("🔍 Analyzing article..."):
            # Run all checks
            ml = ml_check(text)
            news = news_check(keywords(text))
            gem = gemini_check(text)

            # Calculate final score
            score = ml["conf"]*0.9 + news["count"]*25 + (gem["conf"]*1.2 if gem["conf"]>0 else 0)
            final_conf = min(99, round(score,1))
            verdict = "REAL" if final_conf >= 60 else "FAKE"

            # Save to history
            st.session_state.history.append({
                "Verdict": verdict,
                "ML Model": f"{ml['verdict']} ({ml['conf']:.0f}%)",
                "Sources": news["count"],
                "Gemini": gem["verdict"],
                "Confidence": f"{final_conf:.1f}%"
            })

        st.markdown("---")
        
        # FINAL VERDICT
        if verdict == "REAL":
            st.success(f"### ✅ VERDICT: REAL NEWS ({final_conf:.1f}% Confidence)")
            st.info("This article appears to be legitimate based on our analysis.")
            st.balloons()
        else:
            st.error(f"### ⚠️ VERDICT: FAKE NEWS ({final_conf:.1f}% Risk)")
            st.warning("This article appears to be misleading or false.")

        st.markdown("---")
        
        # DETAILED METRICS
        st.subheader("📊 Detailed Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🤖 ML Model", ml["verdict"], f"{ml['conf']:.0f}%")
        with col2:
            st.metric("🌐 Trusted Sources", news["count"])
        with col3:
            st.metric("✨ Gemini AI", gem["verdict"])

        # SOURCE ARTICLES WITH URLS
        if news["articles"]:
            st.markdown("---")
            st.subheader("📰 Trusted News Sources")
            st.info(f"Found {len(news['articles'])} articles from verified outlets")
            
            for idx, article in enumerate(news["articles"], 1):
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
