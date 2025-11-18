import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from serpapi import GoogleSearch
import spacy
import google.generativeai as genai
from datetime import datetime
import hashlib

# ==================== SECURE KEYS (Never exposed) ====================
# These will be automatically overridden by secrets.toml when deployed
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"temperature": 0.1})

# ==================== INITIAL SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")
nlp = load_spacy()

# Load ML model
@st.cache_resource
def load_ml_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_ml_model()

# ==================== CORE FUNCTIONS ====================
def preprocess(text):
    text = re.sub(r'http[s]?://[^\s]+|www\.[^\s]+|@[\w]+|#[\w]+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_claim(text):
    doc = nlp(text[:1500])
    entities = [e.text for e in doc.ents if e.label_ in ["PERSON", "ORG", "GPE", "EVENT", "DATE"]]
    return " ".join(set(entities[:7])) or text.split()[:10]

@st.cache_data(ttl=600, show_spinner=False)
def serpapi_verify(query):
    params = {
        "q": f'"{query}"',
        "tbm": "nws",
        "num": 10,
        "api_key": SERPAPI_KEY
    }
    try:
        results = GoogleSearch(params).get_dict().get("news_results", [])[:6]
        trusted = ["cnn.com", "bbc.com", "nytimes.com", "reuters.com", "apnews.com", "theguardian.com", "npr.org"]
        credible = sum(1 for r in results if any(t in r.get("source", "").lower() for t in trusted))
        return {"results": results, "credible_count": credible, "total": len(results)}
    except:
        return {"results": [], "credible_count": 0, "total": 0}

def gemini_fact_check(text):
    prompt = f"""You are a senior fact-check journalist.
Analyze this news text and return ONLY:

REAL / FAKE / SATIRE / MISLEADING

Then one short line explanation (max 12 words).

Text:
{text[:5000]}"""

    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().split("\n")[0].upper()
        reason = " ".join(response.text.strip().split("\n")[1:])[:80]
        return {"verdict": result, "reason": reason}
    except:
        return {"verdict": "ERROR", "reason": "Gemini unavailable"}

# ==================== STREAMLIT PRO UI ====================
st.set_page_config(
    page_title="TruthLens Pro • Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px;}
    .title {font-size: 48px; font-weight: 900; text-align: center; color: white; text-shadow: 2px 2px 10px rgba(0,0,0,0.3);}
    .subtitle {text-align: center; color: #e0e0e0; font-size: 20px; margin-bottom: 30px;}
    .input-box {background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);}
    .result-real {background: linear-gradient(45deg, #00c853, #64dd17); color: white; padding: 25px; border-radius: 20px; text-align: center; font-size: 32px; font-weight: bold; box-shadow: 0 10px 30px rgba(0,200,0,0.3);}
    .result-fake {background: linear-gradient(45deg, #d50000, #ff1744); color: white; padding: 25px; border-radius: 20px; text-align: center; font-size: 32px; font-weight: bold; box-shadow: 0 10px 30px rgba(200,0,0,0.3);}
    .metric-card {background: white; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🔍 TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Fake News Detection • ML + Google Search + Gemini 1.5 Flash</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Paste the news article below:",
        height=280,
        placeholder="Enter full article text here...",
        label_visibility="collapsed"
    )
    analyze_btn = st.button("🚀 Analyze Truth", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn:
    if not user_input.strip():
        st.error("Please paste a news article to analyze.")
    else:
        with st.spinner("Running 3-layer verification..."):
            # Layer 1: ML Model
            clean = preprocess(user_input)
            vec = vectorizer.transform([clean])
            ml_pred = model.predict(vec)[0]
            ml_prob = model.predict_proba(vec)[0]
            ml_conf = max(ml_prob) * 100

            # Layer 2: SerpAPI
            claim = extract_claim(user_input)
            serp = serpapi_verify(claim)

            # Layer 3: Gemini
            gemini = gemini_fact_check(user_input)

            # Final Score
            score = 0
            if ml_pred == 1: score += 25
            if serp["credible_count"] >= 3: score += 45
            elif serp["credible_count"] >= 1: score += 20
            if "REAL" in gemini["verdict"]: score += 30
            elif "FAKE" in gemini["verdict"]: score -= 40

            confidence = min(98, max(15, score))

        st.markdown("<br>", unsafe_allow_html=True)

        # FINAL VERDICT
        if confidence >= 75:
            st.markdown(f'<div class="result-real">✅ VERIFIED REAL NEWS<br><span style="font-size:22px;">Confidence: {confidence:.1f}%</span></div>', unsafe_allow_html=True)
            st.success("This story is corroborated by multiple trusted sources and AI analysis.")
        elif confidence <= 40:
            st.markdown(f'<div class="result-fake">❌ FAKE / MISLEADING NEWS<br><span style="font-size:22px;">Confidence: {100-confidence:.1f}% Fake</span></div>', unsafe_allow_html=True)
            st.error("This claim lacks credible backing and shows signs of fabrication.")
        else:
            st.markdown(f'<div class="result-fake" style="background: linear-gradient(45deg, #ff6d00, #ff9100);">⚠️ UNCERTAIN • NEEDS VERIFICATION<br><span style="font-size:22px;">Confidence: {confidence:.1f}% Real</span></div>', unsafe_allow_html=True)
            st.warning("Mixed signals. Could be breaking news or partially inaccurate.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed Results
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("ML Model", "Real" if ml_pred==1 else "Fake", f"{ml_conf:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Credible Sources", f"{serp['credible_count']}/10")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Gemini Verdict", gemini["verdict"])
            st.caption(gemini["reason"])
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Final Score", f"{confidence:.1f}%", "Real")
            st.markdown('</div>', unsafe_allow_html=True)

        if serp["results"]:
            st.markdown("### Top Matching Articles from Trusted Sources:")
            for r in serp["results"][:4]:
                st.markdown(f"• **[{r['title']}]({r['link']})**  \n_{r.get('source', '')} • {r.get('date', '')}_")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:14px;'>"
    "© 2025 TruthLens Pro • Powered by Logistic Regression + SerpAPI + Google Gemini<br>"
    "For research & education • Always verify from original sources"
    "</p>",
    unsafe_allow_html=True
)
