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

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"temperature": 0, "max_output_tokens": 30},
    safety_settings=[{"category": cat, "threshold": "BLOCK_NONE"} for cat in genai.types.HarmCategory]
)

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ==================== HELPERS ====================
def preprocess(t): return " ".join([ps.stem(w) for w in re.sub(r'[^a-z\s]', ' ', re.sub(r'http\S+|www\.\S+', ' ', t.lower())).split() if w not in stop_words and len(w)>2])
def extract_keywords(t): return " ".join([w for w in re.findall(r'\b[a-z]{5,20}\b', t.lower()) if w not in stop_words][:12]) or "news"

def read_pdf(f):
    try:
        reader = PyPDF2.PdfReader(BytesIO(f.read()))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        lines = [l.strip() for l in text.split('\n') if 20 < len(l.strip()) < 120]
        key = lines[:8]
        return text[:15000], key
    except: return None, []

def ml_check(t):
    p = preprocess(t)
    pred = model.predict(vectorizer.transform([p]))[0]
    prob = model.predict_proba(vectorizer.transform([p]))[0].max() * 100
    return {"verdict": "REAL" if pred==1 else "FAKE", "confidence": round(prob,1)}

@st.cache_data(ttl=900)
def news_check(q):
    try:
        r = requests.get("https://serpapi.com/search.json", params={"engine":"google","q":q,"tbm":"nws","num":10,"api_key":SERPAPI_KEY}, timeout=15).json()
        results = r.get("news_results", [])[:6]
        trusted = {"cnn","bbc","reuters","nytimes","apnews","theguardian","npr","aljazeera","wsj"}
        hits = [x for x in results if any(t in x.get("source","").lower() for t in trusted)]
        return {"articles": hits, "count": len(hits)}
    except: return {"articles": [], "count": 0}

def gemini_check(t):
    try:
        resp = gemini_model.generate_content(f"Real or fake news? Answer only: REAL or FAKE\n\n{' '.join(t.split()[:500])}")
        ans = resp.text.strip().upper()
        v = "REAL" if "REAL" in ans else "FAKE"
        return {"verdict": v, "confidence": 90}
    except:
        return {"verdict": "FAKE", "confidence": 0}

# ==================== ULTRA PREMIUM UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="lock", layout="centered")

st.markdown("""
<style>
    .title {
        font-size: 56px !important; font-weight: 900 !important; text-align: center !important;
        background: linear-gradient(135deg, #000814 0%, #001d3d 30%, #ffc300 100%);
        -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
        letter-spacing: 1px; text-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    .subtitle {text-align:center; color:#94a3b8; font-size:18px; margin:10px 0 25px;}
    .tiny-box textarea {height:110px !important; font-size:15px !important; padding:12px !important;}
    .result-real {color:#10b981; font-size:50px; font-weight:bold; text-align:center;}
    .result-fake {color:#ef4444; font-size:50px; font-weight:bold; text-align:center;}
    .card {background:#0f172a; color:white; padding:20px; border-radius:16px; box-shadow:0 8px 32px rgba(0,0,0,0.6); margin:15px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Elite Verification Engine • Instant Truth Detection</p>', unsafe_allow_html=True)

# TINY INPUT BOX + PDF
col1, col2 = st.columns([3,1])
with col1:
    article = st.text_area("Paste news or article:", height=110, placeholder="Type or paste text here...", key="tinybox", label_visibility="collapsed")
with col2:
    pdf = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")

text_to_check = article
if pdf:
    with st.spinner("Reading PDF..."):
        txt, keys = read_pdf(pdf)
        if txt:
            text_to_check = txt
            with st.expander("Extracted Key Points", expanded=True):
                for k in keys: st.write("• " + k)

if st.button("Analyze Truth", type="primary", use_container_width=True):
    if not text_to_check.strip():
        st.error("Enter text or upload PDF")
    else:
        with st.spinner("Verifying..."):
            ml = ml_check(text_to_check)
            news = news_check(extract_keywords(text_to_check))
            gem = gemini_check(text_to_check)

            score = ml["confidence"]*0.9 + news["count"]*25 + (gem["confidence"]*1.2 if gem["confidence"]>0 else 0)
            final_conf = min(99, round(score, 1))
            verdict = "REAL" if final_conf >= 60 else "FAKE"

            st.session_state.history.append({
                "Verdict": verdict,
                "ML": f"{ml['verdict']} {ml['confidence']:.0f}%",
                "Sources": news["count"],
                "Gemini": gem["verdict"],
                "Score": f"{final_conf:.1f}%"
            })

        # RESULT
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        if verdict == "REAL":
            st.markdown('<p class="result-real">REAL NEWS</p>', unsafe_allow_html=True)
            st.success(f"**{final_conf:.1f}% Confidence** — Verified Authentic")
            st.balloons()
        else:
            st.markdown('<p class="result-fake">FAKE NEWS</p>', unsafe_allow_html=True)
            st.error(f"**{final_conf:.1f}% Risk** — Likely False")

        c1,c2,c3 = st.columns(3)
        c1.metric("ML Model", ml["verdict"], f"{ml['confidence']:.0f}%")
        c2.metric("Trusted Sources", news["count"])
        c3.metric("Gemini AI", gem["verdict"])

        if news["articles"]:
            st.success(f"{news['count']} trusted outlets confirm:")
            for a in news["articles"]:
                st.markdown(f"• [{a.get('title','')[:90]}]({a.get('link','#')})")

        st.markdown('</div>', unsafe_allow_html=True)

# LIVE TABLE
if st.session_state.history:
    st.markdown("### Recent Verifications")
    df = pd.DataFrame(st.session_state.history[-12:])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("TruthLens Pro © 2025 | Premium • Instant • Unbreakable")
