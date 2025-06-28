# Codec-Technologies-Internship
# 📊 Machine Learning Projects: Churn Prediction & Fraud Detection

This repository contains two hands-on supervised ML projects:
1. **Customer Churn Prediction**
2. **Credit Card Fraud Detection**

Both projects apply data preprocessing, model training, and evaluation using Python and scikit-learn.

---

## 🔹 Project 1: Customer Churn Prediction

### 🎯 Objective
Predict whether a telecom customer is likely to cancel their subscription based on usage patterns, payment method, contract type, and more.

### 📂 Dataset
- Name: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Rows: ~7,043
- Target variable: `Churn`

### 🛠️ Tools & Libraries
- `pandas` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for ML modeling
- `imbalanced-learn` for handling imbalanced data

### 🔎 Key Tasks
- Handled missing values in `TotalCharges`
- Converted categorical data using one-hot encoding
- Trained a `RandomForestClassifier`
- Evaluated model using F1-score, confusion matrix, and ROC-AUC

---

## 🔹 Project 2: Credit Card Fraud Detection

### 🎯 Objective
Identify fraudulent credit card transactions in a highly imbalanced dataset using supervised learning and resampling techniques.

### 📂 Dataset
- Name: `creditcard.csv`
- Rows: 284,807
- Target variable: `Class` (1 = Fraud, 0 = Legit)

### 🛠️ Tools & Libraries
- `pandas`, `numpy` for data handling
- `scikit-learn` for ML algorithms
- `StandardScaler` for scaling `Amount`
- `SMOTE` (from `imbalanced-learn`) to balance the classes

### 🔎 Key Tasks
- Dropped `Time`, scaled `Amount`
- Used `SMOTE` to oversample fraud cases
- Trained a `RandomForestClassifier`
- Evaluated model using precision, recall, F1-score, and AUC

---

## 🧠 Conclusion & Learnings

### ✅ What I Learned
- Handling real-world class imbalance using **SMOTE**
- Preprocessing techniques for both structured and anonymized data
- Model evaluation using **F1-score** and **ROC-AUC**, not just accuracy
- Importance of **feature scaling** and **encoding** in ML workflows
- How to structure end-to-end ML pipelines using `scikit-learn`

Both projects strengthened my understanding of supervised classification, real-world data challenges, and industry-standard ML workflows.

