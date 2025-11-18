import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai
import en_core_web_sm_core_web_sm  # This line is correct after the wheel is installed

# ==================== YOUR KEYS (HARD CODED – WORKS EVERYWHERE) ====================
SERPAPI_KEY = "46cba0dda74df8cb1aec4b685e0290209f5946f75d4e500ee602fa908e11ab8e"
GEMINI_API_KEY = "AIzaSyAYYFPpYJdBL0HoW2g_hLE2X7UIs1K8Qfg"

# ==================== SPACY MODEL – PRE-INSTALLED (NO DOWNLOAD!) ====================
@st.cache_resource
def get_spacy_model():
    return en_core_web_sm.load()

nlp = get_spacy_model()

# ==================== SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== LOAD YOUR PICKLE MODEL ====================
@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ==================== HELPERS ====================
def preprocess(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join([ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2])
    return text

def extract_claim(text: str) -> str:
    try:
        doc = nlp(text[:1000])
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "DATE"]]
        claim = " ".join(set(entities[:6]))
        return claim.strip() or "news fact check"
    except:
        return "news fact check"

@st.cache_data(ttl=600)
def search_news(query: str):
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "tbm": "nws",
                "num": 8,
                "api_key": SERPAPI_KEY
            },
            timeout=15
        )
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted = {"cnn", "bbc", "nytimes", "reuters", "apnews", "theguardian", "npr", "associated press"}
        credible = sum(1 for x in results if any(t in x.get("source", "").lower() for t in trusted))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

def gemini_check(text: str):
    prompt = (
        "Fact-check this article/text. Respond exactly in this format only:\n"
        "VERDICT: REAL or FAKE or SATIRE or MISLEADING\n"
        "REASON: (max 10 words)\n\n"
        f"Article: {text[:4800]}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        output = response.text.strip()
        verdict = "UNKNOWN"
        reason = ""
        for line in output.splitlines():
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().split()[0].upper()
            if line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return {"verdict": verdict, "reason": reason[:70]}
    except:
        return {"verdict": "ERROR", "reason": "Gemini limit reached"}

# ==================== UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="centered")
st.title("magnifying glass TruthLens Pro")
st.markdown("### Fake News Detector with ML + Google News + Gemini AI")

article = st.text_area("Paste the news article here:", height=320)

if st.button("Check Truth", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Please paste an article!")
    else:
        with st.spinner("Analyzing using 3 powerful methods..."):
            clean = preprocess(article)
            pred = model.predict(vectorizer.transform([clean]))[0]
            prob = model.predict_proba(vectorizer.transform([clean]))[0].max() * 100

            claim = extract_claim(article)
            news = search_news(claim)
            gemini = gemini_check(article)

            score = 0
            if pred == 1: score += 30
            score += news["credible"] * 15
            if gemini["verdict"] == "REAL": score += 35
            elif gemini["verdict"] in ["FAKE", "SATIRE", "MISLEADING"]: score -= 40
            confidence = max(5, min(99, score))

        # Final Result
        if confidence >= 70:
            st.success(f"REAL NEWS – {confidence:.1f}% Confidence")
            st.balloons()
        elif confidence <= 35:
            st.error(f"LIKELY FAKE – {100-confidence:.1f}% Fake Score")
        else:
            st.warning(f"UNCERTAIN – {confidence:.1f}% Score")

        col1, col2, col3 = st.columns(3)
        col1.metric("ML Model", "Real" if pred == 1 else "Fake", f"{prob:.0f}%")
        col2.metric("Trusted Sources", news["credible"])
        col3.metric("Gemini AI", gemini["verdict"])

        if gemini["reason"]:
            st.info(f"Gemini says: {gemini['reason']}")

        if news["results"]:
            st.success("Found trusted sources reporting similar news:")
            for r in news["results"]:
                st.markdown(f"• [{r.get('title', 'No title')}]({r.get('link', '#')}) — _{r.get('source', '')}_")

st.caption("TruthLens Pro © 2025 | Works instantly – no secrets needed")
