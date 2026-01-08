# Duplicate-Question-Pairs 
**📌 Project Overview**

Quora receives millions of questions every day. Many of these questions are semantically identical but phrased differently, leading to redundancy and poor user experience.

This project aims to identify whether two given questions are duplicates or not using Natural Language Processing (NLP) and Machine Learning techniques.

The model learns semantic similarity between question pairs and predicts whether they convey the same meaning.

**🎯 Objective**

Detect duplicate question pairs automatically

Reduce redundancy in question-answer platforms

Apply real-world NLP preprocessing, feature engineering, and modeling

**📊 Dataset**

Source: Quora Question Pairs Dataset

Description:

Question 1

Question 2

Label (1 → Duplicate, 0 → Not Duplicate)

Size: 400k+ question pairs

🛠️ Tech Stack

Programming Language: Python

Libraries & Tools:

NumPy

Pandas

Scikit-learn

NLTK

Matplotlib / Seaborn

NLP Techniques:

Text cleaning & normalization

Tokenization

Stopword removal

Lemmatization

TF-IDF Vectorization

**🧠 Approach & Methodology**
1️⃣ Data Preprocessing

Lowercasing text

Removing HTML tags, punctuation, and special characters

Stopword removal

Lemmatization

###2️⃣ Feature Engineering

Question length features

Word overlap ratio

Common word count

TF-IDF vectors for semantic representation

3️⃣ Model Building

Machine Learning models used:

Logistic Regression

Random Forest Classifier

XGBoost (optional / advanced)

Train-test split for evaluation

4️⃣ Model Evaluation

Accuracy

Precision

Recall

F1-score

Confusion Matrix

**📈 Results**

Achieved high accuracy in detecting duplicate question pairs

TF-IDF + Logistic Regression provided a strong baseline

Feature engineering significantly improved model performance
