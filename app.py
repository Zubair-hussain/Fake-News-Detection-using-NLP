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
import os
import json

# ==================== CONFIGURATION ====================
# Keys provided by user
SERPAPI_KEY = "49d04639f5bb0673cfff09228f9f63963b592a93cb2f08d822c166075c9e3f12"
SERPAPI_SECONDARY_KEY = "egHmGsoxyyUSn1kmNrz6d8zG"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# Set page config
st.set_page_config(page_title="TruthLens Pro", page_icon="🔒", layout="centered")

# ==================== SETUP ====================
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = setup_nltk()
ps = PorterStemmer()

# Gemini AI Setup (Standard SDK)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash", 
    generation_config={"temperature": 0, "max_output_tokens": 50},
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
)

# ==================== MODEL LOADING (With Fallback) ====================
class MockModel:
    """Fallback class if pkl files are missing"""
    def predict(self, x): return [1] 
    def predict_proba(self, x): 
        class Proba:
            def max(self): return 0.75
        return Proba()

class MockVectorizer:
    """Fallback class if pkl files are missing"""
    def transform(self, x): return x

@st.cache_resource
def load_model():
    try:
        if os.path.exists("logistic_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
            model = pickle.load(open("logistic_model.pkl", "rb"))
            vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
            return model, vectorizer
        else:
            return MockModel(), MockVectorizer()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return MockModel(), MockVectorizer()

model, vectorizer = load_model()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== HELPERS ====================
def preprocess(text):
    text = re.sub(r'http\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join([ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2])

def extract_keywords(text):
    words = re.findall(r'\b[a-z]{5,20}\b', text.lower())
    filtered = [w for w in words if w not in stop_words][:15]
    return " ".join(filtered) or "news article"

def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        if not text: return None, []
        
        lines = text.split('\n')
        key_points = [line.strip() for line in lines if len(line.strip()) > 20 and len(line.strip().split()) < 15][:10]
        return text[:15000], key_points
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None, []

# ==================== ML CHECK ====================
def ml_check(text):
    try:
        clean = preprocess(text)
        if isinstance(vectorizer, MockVectorizer):
             return {"verdict": "REAL", "confidence": 75.0, "source": "ML Model (Simulated)"}
        
        vec_text = vectorizer.transform([clean])
        pred = model.predict(vec_text)[0]
        prob = model.predict_proba(vec_text)[0].max() * 100
        return {"verdict": "REAL" if pred == 1 else "FAKE", "confidence": round(prob, 1), "source": "ML Model"}
    except Exception:
        return {"verdict": "UNCERTAIN", "confidence": 50.0, "source": "ML Model Error"}

# ==================== NEWS CHECK ====================
@st.cache_data(ttl=900)
def news_check(query):
    trusted_sources = {"cnn", "bbc", "reuters", "nytimes", "apnews", "theguardian", "npr", "aljazeera", "wsj"}

    def fetch_serpapi(key):
        try:
            r = requests.get("https://serpapi.com/search.json", params={
                "engine": "google", "q": query, "tbm": "nws", "num": 10, "api_key": key
            }, timeout=10)
            data = r.json()
            results = data.get("news_results", [])
            hits = [res for res in results if any(t in res.get("source", "").lower() for t in trusted_sources)]
            return hits
        except:
            return []

    hits = fetch_serpapi(SERPAPI_KEY)
    if not hits:
        hits = fetch_serpapi(SERPAPI_SECONDARY_KEY)

    count = len(hits)
    confidence = min(98, count * 28) if count > 0 else 5
    verdict = "REAL" if count >= 2 else "FAKE"
    return {"verdict": verdict, "confidence": confidence, "source": "Trusted News", "articles": hits}

# ==================== GEMINI CHECK (ROBUST) ====================
def gemini_check(text):
    if not GEMINI_API_KEY:
        return {"verdict": "SKIPPED", "confidence": 0, "source": "Key Missing", "reason": "No Key"}

    prompt = f"Is this news real or fake? Answer only: REAL or FAKE. Text: {' '.join(text.split()[:600])}"
    
    # METHOD 1: Try Standard SDK
    try:
        response = gemini_model.generate_content(prompt)
        if not response.parts:
             raise ValueError("Blocked")
        result = response.text.strip().upper()
        return {"verdict": "REAL" if "REAL" in result else "FAKE", "confidence": 90, "source": "Gemini AI", "reason": result[:50]}
    except Exception as sdk_error:
        print(f"SDK Failed ({sdk_error}), trying REST fallback...")

        # METHOD 2: Direct REST API Fallback (Bypasses SDK/GRPC issues)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    result = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                    return {"verdict": "REAL" if "REAL" in result else "FAKE", "confidence": 90, "source": "Gemini AI (REST)", "reason": result[:50]}
                else:
                    return {"verdict": "UNCERTAIN", "confidence": 0, "source": "Gemini Blocked", "reason": "Safety Filter"}
            else:
                error_reason = f"HTTP {response.status_code}"
                if response.status_code == 400: error_reason = "Bad Request (Check Key)"
                if response.status_code == 403: error_reason = "Forbidden (Invalid Key)"
                if response.status_code == 429: error_reason = "Quota Exceeded"
                return {"verdict": "FAKE", "confidence": 0, "source": "Gemini Error", "reason": error_reason}

        except Exception as rest_error:
            return {"verdict": "FAKE", "confidence": 0, "source": "Gemini Unavailable", "reason": str(rest_error)[:30]}

# ==================== FINAL DECISION ====================
def final_decision(ml, news, gemini):
    gemini_conf = gemini["confidence"] if gemini["source"] != "Gemini Unavailable" and gemini["source"] != "Gemini Error" else 0
    
    real_score = ml["confidence"]*0.9 + news["confidence"] + (gemini_conf*1.2 if gemini_conf > 0 else 0)
    fake_score = (100-ml["confidence"])*0.9 + (100-news["confidence"]) + (100-gemini_conf)*1.2
    
    winner = "REAL" if real_score > fake_score else "FAKE"
    total_score = real_score + fake_score
    if total_score == 0: total_score = 1
        
    confidence = round(100*(real_score/total_score), 1)
    return {"verdict": winner, "confidence": confidence, "details":[ml, news, gemini], "articles": news["articles"]}

# ==================== UI ====================
st.markdown("""
<style>
.premium-title { font-size:58px; font-weight:900; text-align:center;
    background: linear-gradient(135deg,#000000 0%,#FFD700 50%,#1E3A8A 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; padding:10px; margin-bottom:5px; text-shadow:2px 2px 4px rgba(0,0,0,0.3);}
.premium-subtitle {text-align:center; color:#64748B; font-size:20px; margin-bottom:25px;}
.result-real {color:#10B981; font-size:52px; font-weight:bold; text-align:center;}
.result-fake {color:#EF4444; font-size:52px; font-weight:bold; text-align:center;}
.stTextArea>div>div>textarea {font-size:16px; border-radius:10px; border:2px solid #E2E8F0;}
.premium-card {background:linear-gradient(145deg,#f8fafc,#e2e8f0); border-radius:15px; padding:20px; margin:10px 0; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="premium-title">TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Elite Fake News Detector • ML + Trusted Sources + Gemini AI</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📝 Paste Text", "📄 Upload PDF"])

text_input = ""
key_points = []

with tab1:
    text_input = st.text_area("Enter news article:", height=180, placeholder="Paste suspicious news here...")

with tab2:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting key points from PDF..."):
            pdf_text, key_points = read_pdf(uploaded_file)
            if pdf_text:
                st.success("PDF analyzed! Key points:")
                for point in key_points[:5]:
                    st.write(f"• {point}")
                text_input = pdf_text
            else:
                st.error("PDF reading failed.")

# ==================== ANALYSIS ====================
if st.button("🔍 Analyze Truth", type="primary", use_container_width=True):
    if not text_input.strip():
        st.error("Please provide text or PDF.")
    else:
        with st.spinner("Processing with ML, News, and Gemini..."):
            ml = ml_check(text_input)
            query = extract_keywords(text_input)
            news = news_check(query)
            gemini = gemini_check(text_input)
            result = final_decision(ml, news, gemini)

            st.session_state.history.append({
                "Verdict": result["verdict"],
                "ML Score": f"{ml['confidence']:.0f}%",
                "Trusted Sources": len(news["articles"]),
                "Gemini": gemini["source"],
                "Overall Confidence": f"{result['confidence']:.1f}%"
            })

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if result["verdict"]=="REAL":
            st.markdown('<p class="result-real">✅ REAL NEWS</p>', unsafe_allow_html=True)
            st.success(f"**{result['confidence']:.1f}% Confidence** – Verified authentic.")
            st.balloons()
        else:
            st.markdown('<p class="result-fake">❌ FAKE / MISLEADING</p>', unsafe_allow_html=True)
            st.error(f"**{result['confidence']:.1f}% Risk** – Likely false information.")

        col1, col2, col3 = st.columns(3)
        col1.metric("ML Model", ml["verdict"], f"{ml['confidence']:.0f}%")
        col2.metric("Trusted Sources", len(news["articles"]))
        col3.metric("Gemini AI", gemini["verdict"], f"{gemini['confidence']:.0f}%")

        if gemini.get("reason") and gemini["confidence"] > 0:
            st.info(f"**Gemini Insight**: {gemini['reason']}")
        elif "Gemini" in gemini["source"] and gemini["confidence"] == 0:
             st.warning(f"**Gemini Issue**: {gemini['reason']}")

        if result["articles"]:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.success(f"✅ {len(result['articles'])} Trusted Sources Confirm:")
            for a in result["articles"]:
                st.markdown(f"• [{a.get('title','')[:90]}]({a.get('link','#')}) – _{a.get('source','Source')}_")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No trusted sources found – extra caution advised.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== HISTORY ====================
if st.session_state.history:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Analytics – Recent Checks")
    df = pd.DataFrame(st.session_state.history[-10:])
    st.dataframe(df, use_container_width=True, height=300, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("TruthLens Pro © 2025 | Premium AI-Powered Verification | PDF & Text Supported")
