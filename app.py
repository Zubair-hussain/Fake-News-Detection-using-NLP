import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import PyPDF2
import pandas as pd
from io import BytesIO
import os
import json
import random

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

# ==================== MODEL LOADING (Neutral Fallback) ====================
class MockModel:
    """Fallback class: Returns Neutral/Uncertain instead of Fake/Real bias"""
    def predict(self, x): return [0] # 0 = Fake/Neutral default
    def predict_proba(self, x): 
        class Proba:
            def max(self): return 0.50 # 50% confidence (Neutral)
        return Proba()

class MockVectorizer:
    def transform(self, x): return x

@st.cache_resource
def load_model():
    try:
        if os.path.exists("logistic_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
            model = pickle.load(open("logistic_model.pkl", "rb"))
            vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
            return model, vectorizer
        else:
            # Defaults to neutral mock if files missing
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
    # Logic Update: Prioritize Capitalized words (Named Entities) if possible, 
    # otherwise fall back to long words.
    
    # 1. Try to find proper nouns (simple heuristic: capitalized words in middle of sentences)
    # (Note: simple heuristic since we don't have full NLP pipeline loaded)
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    if len(entities) > 3:
        # Use unique entities, prioritize top 5
        return " ".join(list(set(entities))[:8])
    
    # 2. Fallback to original long-word logic
    words = re.findall(r'\b[a-z]{5,20}\b', text.lower())
    filtered = [w for w in words if w not in stop_words][:10]
    return " ".join(filtered) or "news article"

def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        if not text: return None, []
        lines = text.split('\n')
        key_points = [line.strip() for line in lines if len(line.strip()) > 20][:10]
        return text[:15000], key_points
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None, []

# ==================== ML CHECK (Unbiased) ====================
def ml_check(text):
    try:
        clean = preprocess(text)
        
        # Logic Update: Identify if we are using Mock or Real
        if isinstance(vectorizer, MockVectorizer):
             # Return Neutral 50% if no model is present
             return {"verdict": "UNCERTAIN", "confidence": 50.0, "source": "ML (Not Loaded)"}
        
        vec_text = vectorizer.transform([clean])
        pred = model.predict(vec_text)[0]
        prob = model.predict_proba(vec_text)[0].max() * 100
        
        verdict = "REAL" if pred == 1 else "FAKE"
        return {"verdict": verdict, "confidence": round(prob, 1), "source": "ML Model"}
    except Exception:
        return {"verdict": "UNCERTAIN", "confidence": 50.0, "source": "ML Error"}

# ==================== NEWS CHECK (Grounding) ====================
@st.cache_data(ttl=900)
def news_check(query):
    # Logic Update: Expanded source list for less bias
    trusted_sources = {
        "cnn", "bbc", "reuters", "nytimes", "apnews", "theguardian", "npr", 
        "aljazeera", "wsj", "bloomberg", "forbes", "usatoday", "dw", "france24", 
        "snopes", "politifact", "factcheck", "washingtonpost", "abc", "nbc", "cbs"
    }

    def fetch_serpapi(key):
        try:
            r = requests.get("https://serpapi.com/search.json", params={
                "engine": "google", "q": query, "tbm": "nws", "num": 10, "api_key": key
            }, timeout=6)
            data = r.json()
            results = data.get("news_results", [])
            # Check if source name contains trusted string OR if trusted string is in the link
            hits = []
            for res in results:
                src = res.get("source", "").lower()
                link = res.get("link", "").lower()
                if any(t in src for t in trusted_sources) or any(t in link for t in trusted_sources):
                    hits.append(res)
            return hits
        except:
            return []

    hits = fetch_serpapi(SERPAPI_KEY)
    if not hits:
        hits = fetch_serpapi(SERPAPI_SECONDARY_KEY)

    count = len(hits)
    
    # Logic Update: Confidence Calculation
    # 0 hits = 10% confidence (Could be real but obscure)
    # 1 hit = 40% confidence
    # 2+ hits = High confidence
    if count == 0:
        confidence = 10
        verdict = "UNCERTAIN"
    elif count == 1:
        confidence = 45
        verdict = "LIKELY REAL"
    else:
        confidence = min(95, 40 + (count * 20))
        verdict = "REAL"

    return {"verdict": verdict, "confidence": confidence, "source": "Trusted News", "articles": hits}

# ==================== GEMINI CHECK (Nuanced Prompt) ====================
def gemini_check(text):
    if not GEMINI_API_KEY:
        return {"score": 0, "confidence": 0, "source": "Key Missing", "reason": "No Key"}

    # Logic Update: Changed prompt to ask for a SCORE (0-100) instead of Binary Real/Fake
    prompt = (
        f"Analyze the following news text for factual accuracy and credibility. "
        f"Provide a credibility score from 0 (Completely Fake/Fabricated) to 100 (Highly Credible/Verified). "
        f"Provide a short reasoning (max 1 sentence). "
        f"Return ONLY this format: Score: <number> | Reason: <text>\n\n"
        f"Text: {' '.join(text.split()[:800])}"
    )
    
    models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]
    
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=payload, headers=headers, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    # Parse the Score
                    # Looking for "Score: 85" pattern
                    score_match = re.search(r"Score:\s*(\d+)", raw_text, re.IGNORECASE)
                    reason_match = re.search(r"Reason:\s*(.*)", raw_text, re.IGNORECASE)
                    
                    score = int(score_match.group(1)) if score_match else 50
                    reason = reason_match.group(1) if reason_match else raw_text[:100]
                    
                    return {
                        "score": score,  # 0 to 100
                        "confidence": 90, # High confidence in the AI's analysis capability
                        "source": f"Gemini ({model})", 
                        "reason": reason
                    }
        except:
            continue 

    # Fallback heuristic
    return {"score": 50, "confidence": 0, "source": "AI Offline", "reason": "Could not connect to AI."}

# ==================== FINAL DECISION (Weighted Logic) ====================
def final_decision(ml, news, gemini):
    # 1. Normalize inputs to 0-100 scale representing "Probability of being REAL"
    
    # ML Score
    ml_prob = ml["confidence"] if ml["verdict"] == "REAL" else (100 - ml["confidence"])
    
    # News Score (Already scaled 0-100 in news_check)
    news_prob = news["confidence"]
    
    # Gemini Score (Already 0-100)
    gemini_prob = gemini.get("score", 50)
    
    # 2. Apply Weights (Crucial for fixing bias)
    # News (Evidence) > Gemini (Reasoning) > ML (Pattern)
    w_news = 0.50
    w_gemini = 0.30
    w_ml = 0.20
    
    final_score = (news_prob * w_news) + (gemini_prob * w_gemini) + (ml_prob * w_ml)
    
    # 3. Determine Verdict
    if final_score >= 75:
        verdict = "REAL"
        risk_label = "Verified Authentic"
    elif final_score >= 55:
        verdict = "LIKELY REAL"
        risk_label = "Likely Authentic"
    elif final_score >= 45:
        verdict = "UNCERTAIN"
        risk_label = "Unverified / Mixed"
    else:
        verdict = "FAKE"
        risk_label = "Likely Fabricated"
        
    return {
        "verdict": verdict, 
        "final_score": final_score, 
        "risk_label": risk_label,
        "details": {"ML": ml_prob, "News": news_prob, "AI": gemini_prob},
        "articles": news["articles"]
    }

# ==================== UI ====================
st.markdown("""
<style>
.premium-title { font-size:58px; font-weight:900; text-align:center;
    background: linear-gradient(135deg,#000000 0%,#FFD700 50%,#1E3A8A 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; padding:10px; margin-bottom:5px; text-shadow:2px 2px 4px rgba(0,0,0,0.3);}
.premium-subtitle {text-align:center; color:#64748B; font-size:20px; margin-bottom:25px;}
.result-box {padding:20px; border-radius:15px; text-align:center; margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
.res-real {background-color:#d1fae5; color:#065f46; border:2px solid #10b981;}
.res-fake {background-color:#fee2e2; color:#991b1b; border:2px solid #ef4444;}
.res-uncertain {background-color:#fef3c7; color:#92400e; border:2px solid #f59e0b;}
.stTextArea>div>div>textarea {font-size:16px; border-radius:10px; border:2px solid #E2E8F0;}
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
                st.success("PDF analyzed!")
                text_input = pdf_text
            else:
                st.error("PDF reading failed.")

# ==================== ANALYSIS ====================
if st.button("🔍 Analyze Truth", type="primary", use_container_width=True):
    if not text_input.strip():
        st.error("Please provide text or PDF.")
    else:
        with st.spinner("Triangulating sources (ML + Search + AI)..."):
            # 1. Run Checks
            ml = ml_check(text_input)
            query = extract_keywords(text_input)
            news = news_check(query)
            gemini = gemini_check(text_input)
            
            # 2. Final Logic
            result = final_decision(ml, news, gemini)
            
            # 3. Save History
            st.session_state.history.append({
                "Verdict": result["verdict"],
                "Score": f"{result['final_score']:.1f}%",
                "Sources": len(news["articles"]),
                "AI Reason": gemini["reason"][:50]+"..."
            })

        # 4. Display Result
        res_class = "res-real" if result["final_score"] > 60 else ("res-uncertain" if result["final_score"] > 40 else "res-fake")
        
        st.markdown(f"""
        <div class="result-box {res_class}">
            <h1 style="margin:0;">{result['verdict']}</h1>
            <h3 style="margin:5px 0;">{result['risk_label']}</h3>
            <p style="font-size:24px; font-weight:bold;">Truth Score: {result['final_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Trusted Sources", f"{len(news['articles'])} Fnd", delta=f"{result['details']['News']:.0f} Score")
        col2.metric("AI Analysis", f"{result['details']['AI']:.0f}/100", help="Gemini's assessment of credibility")
        col3.metric("Pattern Match", f"{result['details']['ML']:.0f}/100", help="ML Stylometric analysis")

        if gemini.get("reason") and "AI Offline" not in gemini["source"]:
            st.info(f"**Gemini Insight**: {gemini['reason']}")
        
        if result["articles"]:
            st.markdown("### 📰 Corroborating Sources")
            for a in result["articles"]:
                st.markdown(f"• [{a.get('title', 'Link')}]({a.get('link', '#')}) - *{a.get('source', 'Unknown')}*")
        elif result["final_score"] < 40:
            st.warning("⚠️ No mainstream sources could verify this story. Proceed with caution.")

# ==================== HISTORY ====================
if st.session_state.history:
    st.divider()
    st.markdown("### 📊 Recent Checks")
    st.dataframe(pd.DataFrame(st.session_state.history[-5:]), use_container_width=True, hide_index=True)

st.caption("TruthLens Pro © 2025 | Logic: 50% Evidence + 30% AI Reasoning + 20% Pattern Match")
