import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai
import spacy
import subprocess
import sys

# ==================== KEYS ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# Use secrets on Streamlit Cloud
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

# ==================== FIX SPACY MODEL (WORKS 100%) ====================
@st.cache_resource
def get_spacy_model():
    try:
     return spacy.load("en_core_web_sm")
 except:
     with st.spinner("First time setup: Downloading small English model (~12MB)..."):
         subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"], check=True)
     return spacy.load("en_core_web_sm")

nlp = get_spacy_model()

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== LOAD YOUR MODEL ====================
@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ==================== HELPERS ====================
def preprocess(text):
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join([ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2])
    return text

def extract_claim(text):
    try:
        doc = nlp(text[:1000])
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON","ORG","GPE","EVENT","DATE"]]
        return " ".join(set(entities[:6])) or "news fact check"
    except:
        return "news fact check"

@st.cache_data(ttl=600)
def search_news(query):
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google", "q": query, "tbm": "nws", "num": 8, "api_key": SERPAPI_KEY
        }, timeout=10)
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted = ["cnn","bbc","nytimes","reuters","apnews","guardian","npr"]
        credible = sum(1 for x in results if any(t in x.get("source","").lower() for t in trusted))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

def gemini_check(text):
    prompt = f"Fact-check this article. Answer exactly:\nVERDICT: REAL or FAKE or SATIRE or MISLEADING\nREASON: (max 10 words)\n\nText: {text[:4800]}"
    try:
        resp = gemini_model.generate_content(prompt)
        t = resp.text.strip()
        verdict = "UNKNOWN"
        reason = ""
        for line in t.split("\n"):
            if "VERDICT:" in line: verdict = line.split("VERDICT:")[1].strip().split()[0]
            if "REASON:" in line: reason = line.split("REASON:")[1].strip()
        return {"verdict": verdict.upper(), "reason": reason[:70]}
    except:
        return {"verdict": "ERROR", "reason": "Gemini down"}

# ==================== UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="centered")

st.title("TruthLens Pro")
st.markdown("### Fake News Detector — ML + Google News + Gemini AI")

article = st.text_area("Paste the news article below:", height=300)

if st.button("Analyze Truth", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Enter some text!")
    else:
        with st.spinner("Analyzing..."):
            clean = preprocess(article)
            pred = model.predict(vectorizer.transform([clean]))[0]
            prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100

            claim = extract_claim(article)
            news = search_news(claim)
            gemini = gemini_check(article)

            score = 0
            if pred == 1: score += 30
            score += news["credible"] * 15
            if gemini["verdict"] == "REAL": score += 30
            elif gemini["verdict"] in ["FAKE","SATIRE","MISLEADING"]: score -= 40

            confidence = max(5, min(99, score))

        # RESULT
        if confidence >= 70:
            st.success(f"REAL NEWS  | Confidence {confidence:.1f}%")
            st.balloons()
        elif confidence <= 35:
            st.error(f"FAKE / MISLEADING  | Fake confidence {100-confidence:.1f}%")
        else:
            st.warning(f"UNCERTAIN  | Score {confidence:.1f}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("ML Model", "Real" if pred==1 else "Fake", f"{prob:.0f}%")
        c2.metric("Trusted Sources", news["credible"])
        c3.metric("Gemini", gemini["verdict"])

        if gemini["reason"]: st.info("Gemini: " + gemini["reason"])

        if news["results"]:
            st.success("Top trusted articles found:")
            for r in news["results"]:
                st.markdown(f"• [{r.get('title','Untitled')}]({r.get('link','#')}) — _{r.get('source','')}_")

st.caption("TruthLens Pro © 2025 | ML + SerpAPI + Gemini | Always double-check sources")
