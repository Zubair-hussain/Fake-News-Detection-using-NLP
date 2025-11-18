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
GEMINI_API_KEY = "AIzaSyBkq0VS1Jz8LCiBG9UzUhxAgUOruVffu2s"  # Your new key

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Gemini setup with safe config
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"temperature": 0, "max_output_tokens": 50},
    safety_settings=[{"category": cat, "threshold": "BLOCK_NONE"} for cat in genai.types.HarmCategory]
)

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Session state for analytics
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
    return " ".join(filtered) or "news article"

def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        # Extract key points (headings-like)
        lines = text.split('\n')
        key_points = [line.strip() for line in lines if len(line.strip()) > 20 and len(line.strip().split()) < 15][:10]
        return text[:15000], key_points
    except:
        return None, []

# ==================== 1. ML CHECK ====================
def ml_check(text):
    clean = preprocess(text)
    pred = model.predict(vectorizer.transform([clean]))[0]
    prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100
    verdict = "REAL" if pred == 1 else "FAKE"
    return {"verdict": verdict, "confidence": round(prob, 1), "source": "ML Model"}

# ==================== 2. NEWS CHECK ====================
@st.cache_data(ttl=900)
def news_check(query):
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": query, "tbm": "nws", "num": 10, "api_key": SERPAPI_KEY
        }, timeout=15)
        data = r.json()
        results = data.get("news_results", [])[:6]
        trusted = {"cnn", "bbc", "reuters", "nytimes", "apnews", "theguardian", "npr", "aljazeera", "wsj"}
        hits = [r for r in results if any(t in r.get("source", "").lower() for t in trusted)]
        count = len(hits)
        verdict = "REAL" if count >= 2 else "FAKE"
        return {"verdict": verdict, "confidence": min(98, count * 28), "source": "Trusted News", "articles": hits}
    except:
        return {"verdict": "FAKE", "confidence": 5, "source": "Search Failed", "articles": []}

# ==================== 3. GEMINI CHECK ====================
def gemini_check(text):
    short_text = " ".join(text.split()[:600])  # Short for speed/no limits
    prompt = f"Is this news real or fake? Answer only: REAL or FAKE\n\n{short_text}"
    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().upper()
        verdict = "REAL" if "REAL" in result else "FAKE"
        return {"verdict": verdict, "confidence": 90, "source": "Gemini AI", "reason": result[:50]}
    except:
        return {"verdict": "FAKE", "confidence": 0, "source": "Gemini Unavailable", "reason": "Skipped"}

# ==================== FINAL VOTING ====================
def final_decision(ml, news, gemini):
    real_score = ml["confidence"] * 0.9 + news["confidence"] + (gemini["confidence"] * 1.2 if gemini["confidence"] > 0 else 0)
    fake_score = (100 - ml["confidence"]) * 0.9 + (100 - news["confidence"]) + (100 - gemini["confidence"]) * 1.2
    winner = "REAL" if real_score > fake_score else "FAKE"
    confidence = round(100 * (real_score / (real_score + fake_score + 1)), 1)
    return {"verdict": winner, "confidence": confidence, "details": [ml, news, gemini], "articles": news["articles"]}

# ==================== PREMIUM UI ====================
st.markdown("""
<style>
    .premium-title {
        font-size: 58px !important;
        font-weight: 900 !important;
        text-align: center !important;
        background: linear-gradient(135deg, #000000 0%, #FFD700 50%, #1E3A8A 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding: 10px !important;
        margin-bottom: 5px !important;
        font-family: 'Arial Black', sans-serif !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    }
    .premium-subtitle {
        text-align: center !important;
        color: #64748B !important;
        font-size: 20px !important;
        margin-bottom: 25px !important;
        font-family: 'Georgia', serif !important;
    }
    .result-real {color: #10B981 !important; font-size: 52px !important; font-weight: bold !important; text-align: center !important;}
    .result-fake {color: #EF4444 !important; font-size: 52px !important; font-weight: bold !important; text-align: center !important;}
    .stTextArea > div > div > textarea {font-size: 16px !important; border-radius: 10px !important; border: 2px solid #E2E8F0 !important;}
    .premium-card {background: linear-gradient(145deg, #f8fafc, #e2e8f0); border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="premium-title">TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Elite Fake News Detector • ML + Trusted Sources + Gemini AI</p>', unsafe_allow_html=True)

# Input Tabs (Compact)
tab1, tab2 = st.tabs(["📝 Paste Text", "📄 Upload PDF"])

text_input = ""
key_points = []
with tab1:
    text_input = st.text_area("Enter news article:", height=180, placeholder="Paste suspicious news here for instant analysis...")

with tab2:
    uploaded_file = st.file_uploader("Upload PDF for extraction", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting key points from PDF..."):
            pdf_text, key_points = read_pdf(uploaded_file)
            if pdf_text:
                st.success("PDF analyzed! Key points extracted:")
                for point in key_points[:5]:
                    st.write(f"• {point}")
                text_input = pdf_text
            else:
                st.error("PDF reading failed.")

if st.button("🔍 Analyze Truth", type="primary", use_container_width=True):
    if not text_input.strip():
        st.error("Please provide text or PDF.")
    else:
        with st.spinner("Processing with 3 layers..."):
            ml = ml_check(text_input)
            query = extract_keywords(text_input)
            news = news_check(query)
            gemini = gemini_check(text_input)
            result = final_decision(ml, news, gemini)

            # Save to history
            st.session_state.history.append({
                "Verdict": result["verdict"],
                "ML Score": f"{ml['confidence']:.0f}% ({ml['verdict']})",
                "Trusted Sources": len(news["articles"]),
                "Gemini": gemini["source"],
                "Overall Confidence": f"{result['confidence']:.1f}%"
            })

        # Result Card
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if result["verdict"] == "REAL":
            st.markdown('<p class="result-real">✅ REAL NEWS</p>', unsafe_allow_html=True)
            st.success(f"**{result['confidence']:.1f}% Confidence** – Verified authentic.")
            st.balloons()
        else:
            st.markdown('<p class="result-fake">❌ FAKE / MISLEADING</p>', unsafe_allow_html=True)
            st.error(f"**{result['confidence']:.1f}% Risk** – Likely false information.")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("ML Model", ml["verdict"], f"{ml['confidence']:.0f}%")
        with col2: st.metric("Trusted Sources", len(news["articles"]))
        with col3: st.metric("Gemini AI", gemini["verdict"], f"{gemini['confidence']:.0f}%")

        if gemini.get("reason") and gemini["confidence"] > 0:
            st.info(f"**Gemini Insight**: {gemini['reason']}")

        if result["articles"]:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.success(f"✅ {len(result['articles'])} Trusted Sources Confirm:")
            for a in result["articles"]:
                st.markdown(f"• [{a.get('title','')[:90]}]({a.get('link','#')}) – _{a.get('source','Source')}_")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No trusted sources found – extra caution advised.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== LIVE ANALYTICS ====================
if st.session_state.history:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Analytics – Recent Checks")
    df = pd.DataFrame(st.session_state.history[-10:])  # Last 10
    st.dataframe(df, use_container_width=True, height=300, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("TruthLens Pro © 2025 | Premium AI-Powered Verification | PDF & Text Supported")
