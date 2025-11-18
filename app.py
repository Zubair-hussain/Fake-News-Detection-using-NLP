import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai

# ==================== KEYS ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyBkq0VS1Jz8LCiBG9UzUhxAgUOruVffu2s"

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ==================== PREPROCESS & CLAIM ====================
def preprocess(text):
    text = re.sub(r'http\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_claim(text):
    words = [w for w in re.findall(r'\b[a-z]{5,20}\b', text.lower()) if w not in stop_words][:12]
    return " ".join(words) or "breaking news"

# ==================== 1. ML MODEL ====================
def ml_check(text):
    clean = preprocess(text)
    pred = model.predict(vectorizer.transform([clean]))[0]
    prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100
    verdict = "REAL" if pred == 1 else "FAKE"
    return {"verdict": verdict, "confidence": prob, "source": "Your ML Model"}

# ==================== 2. TRUSTED SOURCES (SERPAPI) ====================
@st.cache_data(ttl=600)
def news_check(query):
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": query, "tbm": "nws", "num": 12, "api_key": SERPAPI_KEY
        }, timeout=15)
        data = r.json()
        results = data.get("news_results", [])[:8]
        trusted = {"cnn", "bbc", "reuters", "nytimes", "apnews", "theguardian", "npr", "aljazeera", "wsj"}
        trusted_hits = [r for r in results if any(t in r.get("source", "").lower() for t in trusted)]
        count = len(trusted_hits)
        verdict = "REAL" if count >= 2 else "FAKE"
        return {
            "verdict": verdict,
            "confidence": min(95, count * 25),
            "source": "Trusted News Outlets",
            "articles": trusted_hits
        }
    except:
        return {"verdict": "FAKE", "confidence": 10, "source": "Search Failed", "articles": []}

# ==================== 3. GEMINI AI ====================
def gemini_check(text):
    prompt = f"""Fact-check this article. Answer only:

VERDICT: REAL or FAKE
REASON: one short sentence

Article:
{text[:4500]}"""
    try:
        resp = gemini_model.generate_content(prompt)
        output = resp.text.strip().upper()
        verdict = "REAL" if "REAL" in output else "FAKE"
        reason_start = output.find("REASON:")
        reason = output[reason_start+7:].strip() if reason_start != -1 else "No reason"
        return {"verdict": verdict, "confidence": 90, "source": "Gemini AI", "reason": reason[:80]}
    except:
        return {"verdict": "FAKE", "confidence": 5, "source": "Gemini Error", "reason": "AI unavailable"}

# ==================== SMART VOTING ALGORITHM ====================
def final_verdict(ml, news, gemini):
    votes = {"REAL": 0, "FAKE": 0}
    details = []

    # ML Vote
    votes[ml["verdict"]] += ml["confidence"] * 0.8
    details.append(ml)

    # News Vote
    votes[news["verdict"]] += news["confidence"]
    details.append(news)

    # Gemini Vote
    votes[gemini["verdict"]] += gemini["confidence"] * 1.2  # Gemini has highest weight
    details.append(gemini)

    winner = "REAL" if votes["REAL"] > votes["FAKE"] else "FAKE"
    confidence = votes[winner] / (votes["REAL"] + votes["FAKE"]) * 100

    return {
        "verdict": winner,
        "confidence": round(confidence, 1),
        "details": details,
        "trusted_articles": news["articles"],
        "gemini_reason": gemini.get("reason", "")
    }

# ==================== UI – BEAUTIFUL & PROFESSIONAL ====================
st.set_page_config(page_title="Fake News Detection", page_icon="warning", layout="centered")

st.markdown("""
<style>
    .title {
        font-size: 68px !important;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff0000, #000000, #ff0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .real { color: #00ff00; font-size: 60px; font-weight: bold; text-align: center; }
    .fake { color: #ff0000; font-size: 60px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Fake News Detection</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Truth Checker • 3 Independent Systems • Final Verdict")

article = st.text_area("Paste the suspicious news article below:", height=320, placeholder="Enter any news text here...")

if st.button("Detect Fake News", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Please paste an article!")
    else:
        with st.spinner("Running 3 independent checks..."):
            claim = extract_claim(article)
            ml_result = ml_check(article)
            news_result = news_check(claim)
            gemini_result = gemini_check(article)
            final = final_verdict(ml_result, news_result, gemini_result)

        # FINAL VERDICT
        if final["verdict"] == "REAL":
            st.markdown(f'<p class="real">REAL NEWS</p>', unsafe_allow_html=True)
            st.success(f"**Confidence: {final['confidence']}%** – This story is verified.")
            st.balloons()
        else:
            st.markdown(f'<p class="fake">FAKE NEWS</p>', unsafe_allow_html=True)
            st.error(f"**Fake Likelihood: {final['confidence']}%** – High chance this is false.")

        # Show trusted sources
        if final["trusted_articles"]:
            st.success(f"Found {len(final['trusted_articles'])} trusted outlets reporting this:")
            for r in final["trusted_articles"]:
                title = r.get("title", "No title")[:110]
                link = r.get("link", "#")
                source = r.get("source", "News")
                st.markdown(f"• [{title}]({link}) — _{source}_")
        else:
            st.warning("No major trusted news outlet is reporting this story.")

        # Show breakdown
        with st.expander("See how the AI decided (3 votes)"):
            for d in final["details"]:
                st.write(f"**{d['source']}** → **{d['verdict']}** (confidence: {d.get('confidence', 0):.0f}%)")
            if final.get("gemini_reason"):
                st.info(f"Gemini AI: {final['gemini_reason']}")

st.caption("Fake News Detection © 2025 | Powered by ML + Google News + Gemini 1.5 Flash")
