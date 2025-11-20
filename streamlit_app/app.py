import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math

from utils_plots import (
    plot_roc_curve,
    plot_precision_recall,
    plot_confusion_matrix
)

result = {}

# ============================
# CONFIGURE PAGE
# ============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ============================
# API URLS
# ============================
API_SINGLE = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"
API_BATCH = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict_batch"
API_MODELS = "https://credit-card-fraud-detection-ml-webapp.onrender.com/get-models"

# ============================
# PREMIUM UI STYLING
# ============================
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #0E1117;
        }
        .sub-text {
            font-size: 18px;
            text-align: center;
            color: #4F4F4F;
        }
        .result-card {
            padding: 20px;
            border-radius: 15px;
            background: #f5f7fa;
            border: 1px solid #e2e2e2;
            text-align: center;
            margin-top: 20px;
        }
        .probability-box {
            font-size: 26px;
            font-weight: 700;
            color: #0077ff;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Select a model, enter values or upload a CSV file, and detect fraud instantly.</p>", unsafe_allow_html=True)
st.write("")

# ============================
# SIDEBAR
# ============================
st.sidebar.header("⚙️ Choose Model")

try:
    models = requests.get(API_MODELS).json().get("available_models", ["logreg", "rf"])
except:
    models = ["logreg", "rf"]

model = st.sidebar.radio(
    "Select a Machine Learning Model:",
    models,
    index=0,
)

st.sidebar.write("---")
mode = st.sidebar.selectbox(
    "Choose Input Method:",
    ["Manual Input (5-6 values)", "Upload CSV File (FAST MODE)"]
)

# ============================
# API CALL (SINGLE)
# ============================
def call_api(features_list, model_selected):
    payload = {"features": features_list}
    try:
        resp = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload)
        return resp.json(), resp.status_code
    except:
        return {"error": "Server unreachable"}, 500

# ============================
# API CALL (BATCH)
# ============================
def predict_in_chunks(df, model_name="rf", chunk_size=8000):
    total = len(df)
    chunks = math.ceil(total / chunk_size)

    preds = []
    probs = []

    progress = st.progress(0)
    status = st.empty()
    t0 = time.time()

    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        batch = df.iloc[start:end].values.tolist()

        payload = {"features": batch}

        r = requests.post(f"{API_BATCH}?model={model_name}", json=payload)
        out = r.json()

        preds.extend(out["predictions"])
        probs.extend(out["probabilities"])

        progress.progress(end / total)
        elapsed = time.time() - t0
        eta = (elapsed / end) * (total - end)
        status.text(f"Processed {end}/{total} rows — ETA {eta/60:.2f} mins")

    df["prediction"] = preds
    df["fraud_probability"] = probs

    return df

# ============================
# MODE 1 — MANUAL INPUT
# ============================
if mode == "Manual Input (5-6 values)":
    st.subheader("🧮 Manual Input Mode")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.number_input("Feature 1", 0.0)
        f2 = st.number_input("Feature 2", 0.0)
        f3 = st.number_input("Feature 3", 0.0)

    with col2:
        f4 = st.number_input("Feature 4", 0.0)
        f5 = st.number_input("Feature 5", 0.0)
        f6 = st.number_input("Feature 6", 0.0)

    if st.button("🚀 Predict Fraud", use_container_width=True):
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24

        with st.spinner("Analyzing transaction..."):
            result, status = call_api(features, model)

        if status == 200:
            pred = result['prediction']
            prob = result['fraud_probability']

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Prediction Result")

            if pred == 1:
                st.error("⚠️ **FRAUD DETECTED**")
            else:
                st.success("✅ **Legitimate Transaction**")

            st.markdown(f"<p class='probability-box'>Fraud Probability: {prob}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("API Error!")

# ============================
# MODE 2 — BATCH CSV
# ============================
if mode == "Upload CSV File (FAST MODE)":
    st.subheader("📂 Upload CSV File (FAST MODE)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows", use_container_width=True):
            st.write("Running fast batch predictions...")

            out_df = predict_in_chunks(df, model_name=model, chunk_size=8000)

            st.success("Batch Prediction Complete!")
            st.dataframe(out_df.head())

            # Download CSV
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

# ============================
# VISUALIZATION PLACEHOLDER
# ============================
st.subheader("📊 Model Performance Visualizations")
st.info("📌 Visualizations will appear here when batch prediction includes ground-truth labels.")
