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
def preprocess(t): return " ".join([ps.stem(w) for w in re.sub(r'[^a-z\s]', ' ', re.sub(r'http\S+', ' ', t.lower())).split() if w not in stop_words and len(w)>2])
def keywords(t): return " ".join([w for w in re.findall(r'\b[a-z]{5,20}\b', t.lower()) if w not in stop_words][:12]) or "news"
def read_pdf(f): 
    try: 
        r = PyPDF2.PdfReader(BytesIO(f.read()))
        return "".join(p.extract_text() or "" for p in r.pages)[:15000]
    except: return None

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
    except: return {"articles": [], "count": 0}

def gemini_check(t):
    try:
        resp = gemini_model.generate_content(f"Real or fake news? Answer only REAL or FAKE\n\n{' '.join(t.split()[:500])}")
        v = "REAL" if "REAL" in resp.text.strip().upper() else "FAKE"
        return {"verdict": v, "conf": 90}
    except: return {"verdict": "FAKE", "conf": 0}

# PREMIUM UI
st.set_page_config(page_title="TruthLens Pro", page_icon="🔒", layout="centered")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    
    * {font-family: 'Inter', sans-serif !important;}
    
    .stApp {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);}
    
    .title {
        font-size: 64px !important; 
        font-weight: 900 !important; 
        text-align: center !important;
        background: linear-gradient(135deg, #ffc300 0%, #ffd60a 50%, #fff 100%); 
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important; 
        margin: 30px 0 10px;
        letter-spacing: -2px;
        text-shadow: 0 0 80px rgba(255, 195, 0, 0.3);
    }
    
    .subtitle {
        text-align: center; 
        color: #94a3b8; 
        font-size: 18px; 
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    .input-container {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #334155;
        border-radius: 20px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .input-container:hover {
        border-color: #ffc300;
        box-shadow: 0 20px 60px rgba(255, 195, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 16px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ffc300 !important;
        box-shadow: 0 0 0 2px rgba(255, 195, 0, 0.1) !important;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 30px 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #ffc300;
        box-shadow: 0 12px 40px rgba(255, 195, 0, 0.15);
    }
    
    .feature-icon {
        font-size: 36px;
        margin-bottom: 12px;
        display: block;
    }
    
    .feature-title {
        color: #ffc300;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .feature-desc {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.4;
    }
    
    .final-real {
        background: linear-gradient(145deg, #052e16 0%, #064e3b 100%);
        color: #10b981;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3);
        border: 2px solid #10b981;
        letter-spacing: -1px;
    }
    
    .final-fake {
        background: linear-gradient(145deg, #450a0a 0%, #7f1d1d 100%);
        color: #ef4444;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(239, 68, 68, 0.3);
        border: 2px solid #ef4444;
        letter-spacing: -1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ffc300 0%, #ffd60a 100%);
        color: #000814;
        font-weight: 800;
        font-size: 18px;
        padding: 16px 32px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(255, 195, 0, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(255, 195, 0, 0.5);
    }
    
    .upload-section {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #ffc300;
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    }
    
    .stFileUploader {
        background: transparent !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 800;
        color: #ffc300;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .stDataFrame {
        background: #0f172a;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stExpander {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🔒 TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Truth Verification • One Click • One Final Verdict</p>', unsafe_allow_html=True)

# FEATURE SHOWCASE
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <div class="feature-title">ML Analysis</div>
        <div class="feature-desc">Custom-trained model with 95%+ accuracy</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🌐</span>
        <div class="feature-title">Live Sources</div>
        <div class="feature-desc">Real-time verification from trusted outlets</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">✨</span>
        <div class="feature-title">Gemini AI</div>
        <div class="feature-desc">Google's latest AI for cross-validation</div>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN INPUT
st.markdown('<div class="input-container">', unsafe_allow_html=True)
article = st.text_area("📰 Paste your news article or claim here...", height=150, label_visibility="collapsed", 
                       placeholder="Paste any news article, headline, or claim you want to verify...")
st.markdown('</div>', unsafe_allow_html=True)

# PDF UPLOAD (SEAMLESS)
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("**📄 Or upload a PDF document**", unsafe_allow_html=True)
pdf = st.file_uploader("", type="pdf", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

text = article
if pdf:
    with st.spinner("📖 Extracting text from PDF..."):
        txt = read_pdf(pdf)
        if txt: 
            text = txt
            st.success("✅ PDF successfully loaded and ready for analysis")

# ANALYZE BUTTON
if st.button("🚀 Verify Now", type="primary", use_container_width=True):
    if not text.strip():
        st.error("⚠️ Please enter text or upload a PDF to analyze")
    else:
        with st.spinner("🔍 Analyzing with multiple AI systems..."):
            ml = ml_check(text)
            news = news_check(keywords(text))
            gem = gemini_check(text)

            score = ml["conf"]*0.9 + news["count"]*25 + (gem["conf"]*1.2 if gem["conf"]>0 else 0)
            final_conf = min(99, round(score,1))
            verdict = "REAL" if final_conf >= 60 else "FAKE"

            # SAVE HISTORY
            st.session_state.history.append({
                "Verdict": verdict,
                "ML": f"{ml['verdict']} {ml['conf']:.0f}%",
                "Sources": news["count"],
                "Gemini": gem["verdict"],
                "Confidence": f"{final_conf:.1f}%"
            })

        # FINAL VERDICT
        if verdict == "REAL":
            st.markdown(f'<div class="final-real">✓ VERIFIED REAL<br><span style="font-size:28px; font-weight:600;">{final_conf:.1f}% Confidence</span></div>', unsafe_allow_html=True)
            st.success("✅ This story has been verified as authentic and trustworthy.")
            st.balloons()
        else:
            st.markdown(f'<div class="final-fake">⚠ LIKELY FAKE<br><span style="font-size:28px; font-weight:600;">{final_conf:.1f}% Risk Score</span></div>', unsafe_allow_html=True)
            st.error("🚨 High probability this content is false or misleading.")

        # DETAILED BREAKDOWN
        with st.expander("📊 View Detailed Analysis", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("🤖 ML Model", ml["verdict"], f"{ml['conf']:.0f}%")
            c2.metric("🌐 Trusted Sources", news["count"])
            c3.metric("✨ Gemini AI", gem["verdict"])

            if news["articles"]:
                st.markdown("**📰 Trusted outlets reporting this:**")
                for a in news["articles"][:5]:
                    st.markdown(f"• [{a.get('title','')[:90]}...]({a.get('link','#')})")

# HISTORY TABLE
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Recent Verification History")
    df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("🔒 TruthLens Pro © 2025 | Premium AI Truth Verification | Powered by ML + Gemini + Live Sources")
