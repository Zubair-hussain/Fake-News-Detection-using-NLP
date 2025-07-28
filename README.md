#  Fake News Detection using NLP & Machine Learning

##  Overview
This project aims to develop a machine learning model that accurately classifies news articles as *Fake* or *Real* using Natural Language Processing techniques.

##  Goal
To detect and classify fake news from a dataset of ~40,000 labeled news articles.

---

##  Dataset
- Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Contains:
  - `text`, `title`, `subject`, `date`
  - Labels: `0 = Fake`, `1 = Real`

---

##  Process
1. **Data Cleaning & Preprocessing**
   - Lowercasing, punctuation removal, stopword removal
   - Lemmatization using NLTK
2. **TF-IDF Vectorization**
   - Transformed clean text into 1000-dimensional TF-IDF features
3. **Model Training**
   - Trained 3 classical ML models:
     - Naive Bayes
     - Logistic Regression
     - Linear SVM
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix

---

##  Results

| Model               | Accuracy         |
|--------------------|------------------|
| Naive Bayes        | 93.71%           |
| Logistic Regression| 98.74%           |
| Linear SVM         | 99.31%  Best   |

---

##  Tools Used
- Python
- NLTK, Scikit-learn
- Matplotlib, Seaborn
- TF-IDF Vectorizer
- KaggleHub for dataset access

---

##  Learning Outcomes
- Practical experience with NLP pipelines
- Training & evaluating classification models
- Data preprocessing & vectorization
- Cross-validation of real-world datasets

---

##  Deliverables
- Jupyter Notebook (.ipynb)
- README.md (this file)
- Final Accuracy & Evaluation
- Optional: Streamlit front-end (not implemented)

---

##  Future Enhancements
- Add BERT-based deep learning model
- Deploy a Streamlit-based fake news detection app
- Enable live news API classification

##  Run the Project

>  **Open in Google Colab**:  
[Click here to launch the notebook](https://colab.research.google.com/drive/1XXdJ86nstgqwbYVO4X1ScLBaiUfpOjNN?usp=sharing)

---

##  License
This project is shared for educational and research purposes.

