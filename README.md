# 💳 Credit Card Fraud Detection using Local Outlier Factor (LOF)

## 📌 Project Overview

This project detects fraudulent credit card transactions using the **Local Outlier Factor (LOF)** algorithm, an unsupervised machine learning technique for anomaly detection.

A Streamlit web application is included to allow users to upload a compatible credit card transaction dataset and predict anomalous transactions.

---

## 🎯 Objective

- Detect anomalous credit card transactions.
- Apply the Local Outlier Factor (LOF) algorithm.
- Build an interactive Streamlit application for prediction.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Credit_Card_Fraud_Detection_LOF/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── lof_model.pkl
│   └── scaler.pkl
│
├── notebook/
│   └── LOF_Project.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── lof_score_distribution.png
│   ├── correlation_heatmap.png
│   └── amount_boxplot.png
```

---

## ⚙️ Machine Learning Workflow

1. Data preprocessing
2. Feature scaling
3. Train Local Outlier Factor (LOF)
4. Model evaluation
5. Save model using Joblib
6. Deploy using Streamlit

---

## 🚀 How to Run

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📊 Features

- Upload CSV dataset
- Automatic preprocessing
- Fraud prediction
- Prediction summary
- Download prediction results
- Visualization dashboard

---

## 📌 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

Features:

- Time
- V1–V28
- Amount

Target:

- Class (0 = Normal, 1 = Fraud)

---

## 👨‍💻 Author

Arnab Bol

Academic Internship Project

NIT Silchar
