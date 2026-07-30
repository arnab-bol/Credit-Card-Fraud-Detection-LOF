# 💳 Credit Card Fraud Detection using Local Outlier Factor (LOF)

## 📌 Project Overview

This project detects fraudulent credit card transactions using the **Local Outlier Factor (LOF)** algorithm, an unsupervised machine learning technique for anomaly detection.

A Streamlit web application is included to allow users to upload a compatible credit card transaction dataset and detect anomalous (potentially fraudulent) transactions.

---

## 🎯 Objective

- Detect fraudulent credit card transactions using anomaly detection.
- Apply the Local Outlier Factor (LOF) algorithm.
- Build an interactive Streamlit web application.
- Visualize fraud detection results.

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
Credit-Card-Fraud-Detection-LOF/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── sample_creditcard.csv
│
├── models/
│   ├── lof_model_sample.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── credit_card_lof.ipynb
│   ├── create_sample_model.ipynb
│   └── learning_lof.ipynb
│
├── results/
│   ├── amount_boxplot.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── lof_score_distribution.png
```

---

## 🤖 Why Local Outlier Factor (LOF)?

Local Outlier Factor (LOF) is an **unsupervised anomaly detection algorithm** that identifies observations having significantly lower local density than their neighboring data points.

Since fraudulent transactions are extremely rare compared to legitimate transactions, LOF is well suited for detecting suspicious activities without requiring labeled training data.

---

## ⚙️ Machine Learning Workflow

1. Data Loading
2. Data Preprocessing
3. Feature Scaling
4. Train Local Outlier Factor (LOF)
5. Predict Outliers
6. Model Evaluation
7. Save Model using Joblib
8. Deploy using Streamlit

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/arnab-bol/Credit-Card-Fraud-Detection-LOF.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Features

- Upload CSV dataset
- Automatic preprocessing
- Fraud detection using LOF
- Prediction summary
- Download prediction results
- Interactive visualization dashboard

---

## 📌 Dataset

Dataset Used:

**Credit Card Fraud Detection Dataset**

Features include:

- Time
- V1 – V28 (PCA-transformed features)
- Amount

Target Variable:

- **0 → Normal Transaction**
- **1 → Fraudulent Transaction**

> **Note:** A sample dataset is included in this repository for demonstration purposes.

---

## 📈 Visualizations

The project generates the following visualizations:

- Class Distribution
- Correlation Heatmap
- Amount Distribution
- LOF Score Distribution
- Confusion Matrix

---

## 🔮 Future Improvements

- Isolation Forest implementation
- One-Class SVM comparison
- Hyperparameter tuning
- Real-time fraud detection
- Deep Learning based anomaly detection

---

## 👨‍💻 Author

**Arnab Bol**

B.Tech Computer Science & Engineering

Academic Internship Project

National Institute of Technology (NIT) Silchar

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.