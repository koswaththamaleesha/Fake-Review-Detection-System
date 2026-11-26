# AI - Based Fake & Deceptive Online Review Detecting System 
An intelligent system that detects fake and deceptive online reviews using Natural Language Processing (NLP) and Machine Learning techniques. Users can paste a review and instantly get a prediction (Genuine / Deceptive) along with a confidence score and key indicators that influenced the decision.

## 🔍 Project Overview

Online reviews heavily influence customer decisions. However, the rise of fake and deceptive reviews (paid promotions, competitor attacks, AI-generated text) has made it difficult for users to trust online feedback.

This project aims to build a practical and explainable system that:
- Detects whether a review is likely to be *Genuine* or *Deceptive*
- Provides a *confidence score*
- Shows the *key words and writing patterns* that influenced the prediction
- Evaluates models across different datasets to test generalization

## ✨ Key Features

- *Review Analysis* – Paste any product/service review and get instant prediction
- *Confidence Score* – Probability-based confidence with Low / Medium / High suspicion level
- *Explainable Indicators* – Highlights important words and linguistic patterns
- *Model Comparison* – Performance comparison of multiple ML algorithms
- *User Accounts & History* – Store previously checked reviews
- *REST API* – Clean and secure backend API

## 🛠️ Tech Stack

| Layer          | Technologies                                      |
|----------------|---------------------------------------------------|
| *Frontend*   | React.js                                          |
| *Backend*    | Python, FastAPI                                   |
| *ML / NLP*   | scikit-learn, TF-IDF, NLTK / spaCy                |
| *Models*     | Logistic Regression, Random Forest, SVM           |
| *Database*   | PostgreSQL                                        |
| *Tools*      | Git, Postman, Pytest                              |

## 📊 Datasets

Large dataset files are excluded from Git via `.gitignore` (`data/`). Download the datasets locally into `data/` using the links below:

- **Deceptive Opinion Spam Corpus (Hotel Reviews)**
  - Official archive (`op_spam_v1.4.zip`): https://myleott.com/op_spam_v1.4.zip ([Dataset page](https://myleott.com/op-spam.html))
  - Kaggle CSV version: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus
- **Yelp Filtered Review Datasets**
  - Official Stony Brook ODDS pages: [YelpChi](http://odds.cs.stonybrook.edu/yelpchi-dataset/) | [YelpZip](http://odds.cs.stonybrook.edu/yelpzip-dataset/)
  - Yelp Chicago Review Text + Labels CSV (`yelp_reviews.csv`): https://raw.githubusercontent.com/darshandagly/Fake-Review-Detection/master/Data/df.csv
  - Yelp Metadata & Graph CSVs (`YelpCSV`): https://github.com/likhithseera/SemanticWebMine-FakeReviewDetect/raw/main/DATA/YelpCSV-20240304T105840Z-001.zip
- Additional public labelled fake review datasets for cross-dataset evaluation

> Only publicly available research datasets will be used. No private or personal data is collected.

## 🎯 Project Objectives

- Build a complete text preprocessing and feature extraction pipeline
- Train and compare Logistic Regression, Random Forest, and SVM models
- Evaluate models using Precision, Recall, F1-Score, and False Positive Rate
- Test model generalization on reviews from different sources
- Develop a user-friendly web application with explainable results

## 📁 Project Structure

```bash
Fake-Review-Detection-System/
├── backend/                 # FastAPI application
├── frontend/                # React application
├── models/                  # Trained ML models
├── data/                    # Sample / processed datasets
├── notebooks/               # Jupyter notebooks for experiments
├── docs/                    # Documentation & reports
├── .gitignore
└── README.md
