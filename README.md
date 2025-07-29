#  Fake News Detection using NLP & Machine Learning

##  Overview
This project builds a machine learning system to classify news articles as **Fake** or **Real** using Natural Language Processing techniques and classical ML algorithms.

---

##  Goal
To detect and classify fake news using the **WELFake Dataset**, and deploy a lightweight, interactive **Streamlit app** for real-time prediction.

---

##  Dataset
- **Source**: [Kaggle - WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Contents**:
  - `text` (news content)
  - `label`: `0 = Fake`, `1 = Real`
  - ~72,000 labeled news samples

---

##  Process

1. **Data Preprocessing**
   - Lowercasing
   - Punctuation removal
   - Stopword removal
   - Stemming (Porter Stemmer via NLTK)

2. **Feature Extraction**
   - TF-IDF Vectorization (Top 1000 features for speed)

3. **Model Training**
   - Naive Bayes
   - Logistic Regression
   - Linear SVM (best performing)

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix visualization
   - Final comparison chart

5. **Deployment**
   - Live Streamlit UI added using saved `.pkl` model and vectorizer

---

##  Results

| Model                | Accuracy   |
|---------------------|------------|
| Naive Bayes          | 93.71%     |
| Logistic Regression  | 98.74%     |
| Linear SVM (Best)    | **99.31%** ✅ |

---

##  Tools & Libraries Used

- Python
- NLTK (stopwords, stemming)
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
- TF-IDF Vectorizer
- KaggleHub (dataset import)

---

##  Learning Outcomes

- Hands-on NLP preprocessing pipeline
- Comparison of ML classifiers on real-world data
- Deployment of ML model using Streamlit
- End-to-end workflow from dataset to interactive UI

---

##  Deliverables

-  Full training notebook: `Fake_News_Classification_With_UI.ipynb`
-  Trained model: `logistic_model.pkl`
-  TF-IDF Vectorizer: `tfidf_vectorizer.pkl`
-  Streamlit App: `app.py`
-  Complete GitHub structure
-  `requirements.txt` for deployment

---

##  Future Improvements

- Add deep learning models (e.g., BERT, LSTM)
- Improve UI styling (multi-tab layout, article scraping)
- Enable real-time classification via News API

---

##  Run the Project

> **Launch the Notebook in Colab**  
[📎 Open in Google Colab](https://colab.research.google.com/drive/1XXdJ86nstgqwbYVO4X1ScLBaiUfpOjNN?usp=sharing)

> **Try the Web App (Streamlit)**  
 Deploy using Streamlit Cloud by uploading this repo.

---

##  License

This project is shared for educational and demonstration purposes only. Dataset belongs to original Kaggle source.

---

## Tags

`#MachineLearning` `#FakeNewsDetection` `#NLP` `#TFIDF` `#Streamlit` `#ScikitLearn` `#TextClassification` `#WELFake`

