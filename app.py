import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import google.generativeai as genai
import en_core_web_sm  # This works because it's pre-installed via requirements.txt

# ==================== KEYS ====================
# Use secrets on Streamlit Cloud (recommended)
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback for local testing only (never commit real keys!)
    SERPAPI_KEY = "YOUR_SERPAPI_KEY_HERE"
    GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"

# ==================== SPACY MODEL (Pre-installed) ====================
@st.cache_resource
def get_spacy_model():
    return en_core_web_sm.load()

nlp = get_spacy_model()

# ==================== NLTK SETUP ====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ==================== GEMINI SETUP ====================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== LOAD ML MODEL ====================
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
                "num": 10,
                "api_key": SERPAPI_KEY
            },
            timeout=12
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("news_results", [])[:5]
        trusted = {"cnn", "bbc", "nytimes", "reuters", "apnews", "theguardian", "npr", "associated press"}
        credible = sum(1 for res in results if any(t in res.get("source", "").lower() for t in trusted))
        return {"results": results, "credible": credible}
    except:
        return {"results": [], "credible": 0}

def gemini_check(text: str):
    prompt = (
        "Fact-check this article/claim. Respond exactly in this format:\n"
        "VERDICT: REAL or FAKE or SATIRE or MISLEADING\n"
        "REASON: (max 10 words)\n\n"
        f"Text: {text[:4800]}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        output = response.text.strip()
        verdict = "UNKNOWN"
        reason = ""
        for line in output.split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip().split()[0].upper()
            if line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        return {"verdict": verdict, "reason": reason[:70]}
    except Exception as e:
        return {"verdict": "ERROR", "reason": "Gemini unavailable"}

# ==================== UI ====================
st.set_page_config(page_title="TruthLens Pro", page_icon="magnifying glass", layout="centered")

st.title("magnifying glass TruthLens Pro")
st.markdown("### Advanced Fake News Detector — ML + Google News + Gemini AI")

article = st.text_area("Paste the full news article here:", height=300, placeholder="Enter news text to analyze...")

if st.button("Analyze Truth", type="primary", use_container_width=True):
    if not article.strip():
        st.error("Please paste an article to analyze!")
    else:
        with st.spinner("Analyzing with ML model, Google News, and Gemini AI..."):
            clean_text = preprocess(article)
            prediction = model.predict(vectorizer.transform([clean_text]))[0]
            probability = model.predict_proba(vectorizer.transform([clean_text]))[0].max() * 100

            claim = extract_claim(article)
            news_results = search_news(claim)
            gemini_result = gemini_check(article)

            # Scoring system
            score = 0
            if prediction == 1:  # Real news
                score += 30
            score += news_results["credible"] * 14
            if gemini_result["verdict"] == "REAL":
                score += 35
            elif gemini_result["verdict"] in ["FAKE", "SATIRE", "MISLEADING"]:
                score -= 45

            confidence = max(5, min(99, score))

        # Final Verdict
        if confidence >= 70:
            st.success(f"REAL NEWS | Confidence: {confidence:.1f}%")
            st.balloons()
        elif confidence <= 35:
            st.error(f"FAKE OR MISLEADING | Fake Likelihood: {100-confidence:.1f}%")
        else:
            st.warning(f"UNCERTAIN | Confidence Score: {confidence:.1f}%")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("ML Model Prediction", "Real" if prediction == 1 else "Fake", f"{probability:.0f}%")
        col2.metric("Trusted Sources Found", news_results["credible"], help="CNN, BBC, Reuters, NYT, etc.")
        col3.metric("Gemini AI Verdict", gemini_result["verdict"])

        if gemini_result["reason"] and gemini_result["reason"] != "Gemini unavailable":
            st.info(f"Gemini Reasoning: {gemini_result['reason']}")

        if news_results["results"]:
            st.success("Related trusted articles found:")
            for item in news_results["results"]:
                title = item.get("title", "No title")
                link = item.get("link", "#")
                source = item.get("source", "Unknown source")
                st.markdown(f"• [{title}]({link}) — _{source}_")

st.caption("TruthLens Pro © 2025 | Powered by Logistic Regression + SerpAPI + Gemini 1.5 Flash | Always verify with multiple sources.")
