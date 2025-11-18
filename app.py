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
st.set_page_config(page_title="TruthLens Pro", page_icon="lock", layout="centered")
st.markdown("""
<style>
    .title {font-size:60px !important; font-weight:900 !important; text-align:center !important;
            background:linear-gradient(135deg,#000814,#001d3d,#ffc300); -webkit-background-clip:text !important;
            -webkit-text-fill-color:transparent !important; margin:20px 0 10px;}
    .subtitle {text-align:center; color:#94a3b8; font-size:18px; margin-bottom:25px;}
    .chat-box {background:#0f172a; border:2px solid #1e293b; border-radius:16px; padding:16px; box-shadow:0 4px 20px rgba(0,0,0,0.5);}
    .chat-box textarea {background:transparent !important; color:white !important; border:none !important;}
    .drop-zone {border:2px dashed #475569; border-radius:12px; padding:20px; text-align:center; background:#1e293b; color:#94a3b8; font-size:15px;}
    .drop-zone:hover {border-color:#ffc300; color:#ffc300;}
    .final-real {background:#052e16; color:#10b981; padding:25px; border-radius:16px; text-align:center; font-size:48px; font-weight:bold; margin:20px 0; box-shadow:0 8px 32px rgba(0,0,0,0.6);}
    .final-fake {background:#450a0a; color:#ef4444; padding:25px; border-radius:16px; text-align:center; font-size:48px; font-weight:bold; margin:20px 0; box-shadow:0 8px 32px rgba(0,0,0,0.6);}
    .card {background:#0f172a; color:white; padding:20px; border-radius:16px; margin:15px 0; box-shadow:0 8px 32px rgba(0,0,0,0.6);}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">One Click • One Final Verdict • No Confusion</p>', unsafe_allow_html=True)

# TWIN INPUT
col1, col2 = st.columns([4,1])
with col1:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    article = st.text_area("Paste news or ask anything...", height=110, label_visibility="collapsed", placeholder="Enter article or upload PDF →")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="drop-zone">', unsafe_allow_html=True)
    pdf = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

text = article
if pdf:
    with st.spinner("Reading PDF..."):
        txt = read_pdf(pdf)
        if txt: text = txt; st.success("PDF loaded")

if st.button("Get Final Verdict", type="primary", use_container_width=True):
    if not text.strip():
        st.error("Please enter text or upload PDF")
    else:
        with st.spinner("Final analysis in progress..."):
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

        # ONE SINGLE FINAL OUTPUT
        if verdict == "REAL":
            st.markdown(f'<div class="final-real">REAL NEWS<br><span style="font-size:28px;">{final_conf:.1f}% Confidence</span></div>', unsafe_allow_html=True)
            st.success("This story is verified and authentic.")
            st.balloons()
        else:
            st.markdown(f'<div class="final-fake">FAKE NEWS<br><span style="font-size:28px;">{final_conf:.1f}% Risk</span></div>', unsafe_allow_html=True)
            st.error("High probability this is false or misleading.")

        # Details (collapsible)
        with st.expander("View Detailed Breakdown", expanded=False):
            c1,c2,c3 = st.columns(3)
            c1.metric("Your ML Model", ml["verdict"], f"{ml['conf']:.0f}%")
            c2.metric("Trusted Sources Found", news["count"])
            c3.metric("Gemini AI", gem["verdict"])

            if news["articles"]:
                st.write("**Trusted outlets reporting this:**")
                for a in news["articles"][:5]:
                    st.markdown(f"• [{a.get('title','')[:90]}]({a.get('link','#')})")

# LIVE HISTORY
if st.session_state.history:
    st.markdown("### Recent Verdicts")
    df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("TruthLens Pro © 2025 | One Final Verdict • Premium • Instant")
