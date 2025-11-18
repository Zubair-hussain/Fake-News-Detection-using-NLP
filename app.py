import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import PyPDF2
from io import BytesIO
import pandas as pd

# ==================== CONFIG ====================
st.set_page_config(page_title="TruthLens Pro",  # Changed from "Fake News"
                   page_icon="lock", 
                   layout="centered")

# ==================== KEYS ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== TEXT PREPROCESSING ====================
def preprocess(text):
    text = re.sub(r'http\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_keywords(text):
    words = re.findall(r'\b[a-z]{5,20}\b', text.lower())
    filtered = [w for w in words if w not in stop_words][:15]
    return " ".join(filtered) or "news article"

# ==================== PDF READER ====================
def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text[:15000]  # Limit for speed
    except:
        return None

# ==================== TRUSTED NEWS SEARCH ====================
@st.cache_data(ttl=900)
def search_trusted_news(query):
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "tbm": "nws",
            "num": 10,
            "api_key": SERPAPI_KEY
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        results = data.get("news_results", [])[:6]
        trusted_domains = {"cnn.com", "bbc.com", "reuters.com", "nytimes.com", "apnews.com", 
                          "theguardian.com", "npr.org", "aljazeera.com", "wsj.com"}
        trusted = [r for r in results if any(d in r.get("link", "") for d in trusted_domains)]
        return trusted, len(trusted)
    except:
        return [], 0

# ==================== PREMIUM UI ====================
st.markdown("""
<style>
    .main-title {
        font-size: 52px;
        font-weight: 800;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 8px;
        font-family: 'Georgia', serif;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-real {color: #059669; font-size: 48px; font-weight: bold; text-align: center;}
    .result-fake {color: #dc2626; font-size: 48px; font-weight: bold; text-align: center;}
    .stTextArea > div > div > textarea {font-size: 16px !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">TruthLens Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Fake News Detection • ML + Global Trusted Sources</p>', unsafe_allow_html=True)

# Input: Text OR PDF
tab1, tab2 = st.tabs(["Paste Text", "Upload PDF"])

text_input = ""
with tab1:
    text_input = st.text_area("Enter news article:", height=180, placeholder="Paste any news text here...")

with tab2:
    uploaded_file = st.file_uploader("Or upload a PDF document", type="pdf")
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            pdf_text = read_pdf(uploaded_file)
            if pdf_text:
                st.success("PDF loaded successfully!")
                text_input = pdf_text
                st.text_area("Extracted Text (preview):", pdf_text[:1000] + "...", height=120, disabled=True)
            else:
                st.error("Could not read PDF.")

if st.button("Analyze Now", type="primary", use_container_width=True):
    if not text_input.strip():
        st.error("Please enter text or upload a PDF.")
    else:
        with st.spinner("Analyzing content..."):
            # 1. ML Prediction
            clean = preprocess(text_input)
            pred = model.predict(vectorizer.transform([clean]))[0]
            prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100
            ml_verdict = "Real" if pred == 1 else "Fake"

            # 2. Trusted Sources
            query = extract_keywords(text_input)
            sources, count = search_trusted_news(query)

            # Final Score
            ml_weight = prob * 0.9
            news_weight = count * 25
            total = ml_weight + news_weight
            final_confidence = min(99, total)
            is_real = final_confidence >= 60

            # Save to history
            st.session_state.history.append({
                "Verdict": "REAL" if is_real else "FAKE",
                "ML": f"{ml_verdict} ({prob:.0f}%)",
                "Trusted Sources": count,
                "Confidence": f"{final_confidence:.1f}%"
            })

        # === RESULT ===
        if is_real:
            st.markdown('<p class="result-real">REAL NEWS</p>', unsafe_allow_html=True)
            st.success(f"**{final_confidence:.1f}% Confidence** — This appears to be authentic.")
            st.balloons()
        else:
            st.markdown('<p class="result-fake">FAKE / MISLEADING</p>', unsafe_allow_html=True)
            st.error(f"**{final_confidence:.1f}% Risk** — High chance of being false.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your ML Model", ml_verdict, f"{prob:.0f}%")
        with col2:
            st.metric("Trusted Sources Found", count)

        if sources:
            st.success(f"Found {count} trusted outlets reporting similar content:")
            for s in sources:
                st.markdown(f"• [{s['title'][:90]}]({s['link']})")
        else:
            st.warning("No major trusted news outlet is covering this story.")

# ==================== LIVE ANALYTICS TABLE ====================
if st.session_state.history:
    st.markdown("### Recent Checks")
    df = pd.DataFrame(st.session_state.history[-10:])  # Last 10
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("TruthLens Pro © 2025 | Professional Fake Counterintelligence • No AI Limits • PDF Supported")
