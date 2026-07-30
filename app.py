import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load model and scaler
import os

if os.path.exists("models/lof_model.pkl"):
    lof = joblib.load("models/lof_model.pkl")
else:
    lof = joblib.load("models/lof_model_sample.pkl")

scaler = joblib.load("models/scaler.pkl")


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("💳 Fraud Detection App")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📤 Upload CSV",
        "📊 Results",
        "📈 Visualizations",
        "ℹ️ About Project"
    ]
)

# ---------------- Home Page ----------------
if page == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection using Local Outlier Factor (LOF)")

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
This application detects fraudulent credit card transactions using the
**Local Outlier Factor (LOF)** algorithm, an unsupervised anomaly detection
technique.

The model identifies unusual transaction patterns without requiring labeled
training data.

### Workflow
1. Upload your own CSV OR use the sample dataset.
2. Data preprocessing and feature scaling.
3. LOF predicts anomalous transactions.
4. Results and visualizations are displayed.

### Features
- Upload custom CSV files
- Built-in sample dataset
- Automatic feature scaling
- Fraud prediction
- Download prediction results
- Interactive visualizations
""")

    st.success("✅ Model Loaded Successfully")
    st.write("""
**Model Details**

- Algorithm: Local Outlier Factor (LOF)
- n_neighbors = 20
- contamination = 0.0017
- novelty = True
""")
    st.info(
    "📌 This project uses the Local Outlier Factor (LOF) algorithm "
    "to identify anomalous credit card transactions."
    )


# ---------------- Upload ----------------
elif page == "📤 Upload CSV":

    st.title("📤 Upload Dataset")

    st.info("Choose one of the following options:")

    option = st.radio(
    "Select Dataset",
    ["⭐ Use Sample Dataset (Recommended)", "📤 Upload Your Own CSV"],
    index=0
    )   

    df = None

    # ---------------- Upload Own CSV ----------------
    if option == "📤 Upload Your Own CSV":

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset Uploaded Successfully")

    # ---------------- Sample Dataset ----------------
    elif option == "⭐ Use Sample Dataset (Recommended)":
        

        sample_path = "dataset/sample_creditcard.csv"

        if os.path.exists(sample_path):

            df = pd.read_csv(sample_path)

            st.success("✅ Sample Dataset Loaded Successfully")

            with open(sample_path, "rb") as file:
                st.download_button(
                    label="📥 Download Sample Dataset",
                    data=file,
                    file_name="sample_creditcard.csv",
                    mime="text/csv"
                )

        else:
            st.error("❌ Sample dataset not found.")

    # ---------------- Prediction ----------------
    if df is not None:

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.write("### Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        # Prepare features
        if "Class" in df.columns:
            features = df.drop(columns=["Class"])
        else:
            features = df.copy()

        st.write("✅ Features Prepared")

        try:

            scaled_features = scaler.transform(features)

            st.write("✅ Scaling Successful")

            predictions = lof.predict(scaled_features)

            df["Prediction"] = np.where(
                predictions == -1,
                "Anomalous",
                "Normal"
            )

            st.success("✅ Prediction Completed!")

            st.dataframe(df.head(10))

            st.session_state["results"] = df
            st.session_state["features"] = features

        except Exception as e:
            st.error(f"❌ Error while processing file: {e}")
# ---------------- Results ----------------
elif page == "📊 Results":

    st.title("📊 Prediction Results")

    if "results" in st.session_state:

        result_df = st.session_state["results"]

        st.subheader("Prediction Results")

        st.dataframe(result_df)

        st.subheader("Prediction Summary")

        normal = (result_df["Prediction"] == "Normal").sum()
        fraud = (result_df["Prediction"] == "Anomalous").sum()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Normal Transactions", normal)

        with col2:
            st.metric("Anomalous Transactions", fraud)

        st.bar_chart(result_df["Prediction"].value_counts())

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Please upload a dataset first.")

# ---------------- Visualizations ----------------
# ---------------- Visualizations ----------------
elif page == "📈 Visualizations":

    st.title("📈 Dynamic Visualizations")

    if "results" not in st.session_state:

        st.warning("⚠️ Please upload a dataset first.")

    else:

        df = st.session_state["results"]

        # ---------------- Prediction Distribution ----------------

        st.subheader("Prediction Distribution")

        st.bar_chart(df["Prediction"].value_counts())
        fig, ax = plt.subplots(figsize=(5,5))

        df["Prediction"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)
        plt.close(fig)

        # ---------------- Amount Histogram ----------------

        if "Amount" in df.columns:

            st.subheader("Transaction Amount Distribution")

            fig, ax = plt.subplots(figsize=(8,4))

            ax.hist(df["Amount"], bins=50)

            ax.set_xlabel("Transaction Amount")

            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

        # ---------------- Box Plot ----------------

        if "Amount" in df.columns:

            st.subheader("Amount Box Plot")

            fig, ax = plt.subplots(figsize=(8,6))

            sns.boxplot(
                x="Prediction",
                y="Amount",
                data=df,
                ax=ax
            )

            ax.set_title("Transaction Amount Distribution by Prediction")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Transaction Amount")

            st.pyplot(fig)
            plt.close(fig)

        # ---------------- Heatmap ----------------

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(12,8))

        sns.heatmap(
            df.select_dtypes(include=np.number).corr(),
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        # ---------------- Confusion Matrix ----------------

        if "Class" in df.columns:

            st.subheader("Confusion Matrix")

            y_true = df["Class"]

            y_pred = np.where(df["Prediction"]=="Anomalous",1,0)

            cm = confusion_matrix(y_true,y_pred)

            fig, ax = plt.subplots(figsize=(5,5))

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Normal","Fraud"]
            )

            disp.plot(ax=ax)

            st.pyplot(fig)
            plt.close(fig)

            # ---------------- Metrics ----------------

            st.subheader("Evaluation Metrics")

            col1,col2,col3,col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{accuracy_score(y_true,y_pred):.4f}"
            )

            col2.metric(
                    "Precision",
                f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
            )

            col3.metric(
                "Recall",
                f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
            )

            col4.metric(
                "F1 Score",
                f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
            )

        else:

            st.info("Dataset has no Class column. Confusion Matrix unavailable.")

        # ---------------- Top Anomalies ----------------

        st.subheader("Detected Anomalous Transactions")

        fraud_df = df[df["Prediction"]=="Anomalous"]
        st.write(f"Total Detected Anomalies: {len(fraud_df)}")
        

        if len(fraud_df) > 0:
            st.dataframe(fraud_df.head(20))
        else:
            st.success("✅ No anomalous transactions detected.")
# ---------------- About ----------------
elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
    ## Credit Card Fraud Detection using Local Outlier Factor

    ### Objective

    Detect anomalous credit card transactions using an unsupervised machine learning algorithm.

    ### Algorithm

    Local Outlier Factor (LOF)

    ### Dataset

    European Credit Card Transactions Dataset

    - Total Transactions: 284,807
    - Fraud Transactions: 492
    - Features: 30

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    - Joblib

    ### Developed For

    Academic Internship Project

    NIT Silchar
    """)


# Footer
st.markdown("---")
st.caption("Developed by Arnab Bol | Academic Internship, NIT Silchar")