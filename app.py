import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai

# KEYS (hardcoded)
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# NLTK setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load your ML model
@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer
model, vectorizer = load_model()

# Simple preprocessing
def preprocess(text):
    text = re.sub(r'http\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Simple claim extraction (no spaCy)
def extract_claim(text):
    words = [w for w in text.lower().split()[:70] if len(w) > 4 and w not in stop_words][:10]
    return " ".join(words) or "news fact check"

# Google News search
@st.cache_data(ttl=600)
def search_news(query):
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": query, "tbm": "nws", "num": 8, "api_key": SERPAPI_KEY
        }, timeout=15)
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted = {"cnn","bbc","reuters","nytimes","apnews","theguardian","npr"}
        credible = sum(1 for r in results if any(t in r.get("source","").lower() for t in trusted))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

# Gemini fact-check
def gemini_check(text):
    prompt = f"Fact-check this. Answer exactly:\nVERDICT: REAL or FAKE or SATIRE or MISLEADING\nREASON: max 10 words\n\nText: {text[:4800]}"
    try:
        resp = gemini_model.generate_content(prompt)
        output = resp.text.strip()
        verdict = "UNKNOWN"
        reason = ""
        for line in output.split("\n"):
            if "VERDICT:" in line: verdict = line.split("VERDICT:")[-1].strip().split()[0].upper()
            if "REASON:" in line: reason = line.split("REASON:")[-1].strip()
        return {"verdict": verdict, "reason": reason[:70]}
    except:
        return {"verdict": "ERROR", "reason": "Gemini busy"}

# UI
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="centered")
st.title("magnifying glass TruthLens Pro")
st.markdown("### Fake News Detector – No spaCy, No Errors")

article = st.text_area("Paste the article here:", height=300)

if st.button("Check Truth", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Paste some text!")
    else:
        with st.spinner("Analyzing..."):
            clean = preprocess(article)
            pred = model.predict(vectorizer.transform([clean]))[0]
            prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100
            claim = extract_claim(article)
            news = search_news(claim)
            gemini = gemini_check(article)

            score = 50
            if pred == 1: score += 25
            score += news["credible"] * 10
            if gemini["verdict"] == "REAL": score += 30
            elif gemini["verdict"] in ["FAKE","SATIRE","MISLEADING"]: score -= 40
            confidence = max(10, min(99, score))

        if confidence >= 70:
            st.success(f"REAL NEWS – {confidence}%")
            st.balloons()
        elif confidence <= 40:
            st.error(f"LIKELY FAKE – {100-confidence}% fake")
        else:
            st.warning(f"UNCERTAIN – {confidence}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("ML Model", "Real" if pred==1 else "Fake", f"{prob:.0f}%")
        c2.metric("Trusted Sources", news["credible"])
        c3.metric("Gemini", gemini["verdict"])
        if gemini["reason"]: st.info(gemini["reason"])
        if news["results"]:
            st.success("Trusted articles:")
            for r in news["results"]:
                st.markdown(f"• [{r.get('title','')[:80]}]({r.get('link','#')}) — _{r.get('source','')}_")

st.caption("TruthLens Pro – 100% working on Streamlit Cloud")
