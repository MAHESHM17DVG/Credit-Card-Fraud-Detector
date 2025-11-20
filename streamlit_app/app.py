# FINAL LUXURY FRONTEND (APPLE-LEVEL UI + FULL PROBABILITY DISPLAY + CREDIT SECTION)

import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ===================================================
# CUSTOM PREMIUM UI (APPLE‑STYLE GLASS + SHADOWS + BLUR)
# ===================================================
st.markdown("""
<style>
    /* Global Background */
    body {
        background: #f2f5f7;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.35);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.20);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 25px;
        margin-top: 20px;
    }

    /* Main title */
    .main-title {
        font-size: 52px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #0a84ff, #5ac8fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        margin-bottom: -10px;
    }

    .sub-text {
        text-align: center;
        font-size: 20px;
        color: #4c4c4c;
        margin-bottom: 20px;
    }

    /* Probability box */
    .prob-box {
        font-size: 34px;
        font-weight: 900;
        text-align: center;
        margin-top: 10px;
    }

    /* Developer credit */
    .credit {
        margin-top: 50px;
        text-align: center;
        font-size: 16px;
        opacity: 0.7;
    }

    .credit a {
        color: #007aff;
        font-weight: bold;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================
# HEADER
# ===================================================
st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Premium Intelligent Fraud Prediction – Powered by Advanced Machine Learning</p>", unsafe_allow_html=True)

# ===================================================
# SESSION STATE
# ===================================================
if "y_true" not in st.session_state:
    st.session_state.y_true = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None

# ===================================================
# API ROUTES
# ===================================================
API_SINGLE = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"
API_BATCH = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict-batch"
API_MODELS = "https://credit-card-fraud-detection-ml-webapp.onrender.com/get-models"

# ===================================================
# SIDEBAR (Glass Look Optional)
# ===================================================
st.sidebar.title("⚙️ Model Controls")
try:
    models = requests.get(API_MODELS, timeout=5).json().get("available_models", ["logreg", "rf"])
except:
    models = ["logreg", "rf"]

model = st.sidebar.radio("Select Model", models)
mode = st.sidebar.selectbox("Input Method", ["Manual Input (6 values)", "Upload CSV File"])

# ===================================================
# FEATURE ORDER
# ===================================================
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df):
    df_local = df.copy().reset_index(drop=True)
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])
    numeric = df_local.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric) >= 30:
        feat_df = df_local[numeric[:30]].copy()
        feat_df.columns = KAGGLE_ORDER
        return feat_df
    for i in range(30 - len(df_local.columns)):
        df_local[f"pad_{i}"] = 0.0
    feat_df = df_local.iloc[:, :30]
    feat_df.columns = KAGGLE_ORDER
    return feat_df

# ===================================================
# API CALL – SINGLE
# ===================================================
def call_api(features, model_selected):
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json={"features": features}, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

# ===================================================
# API CALL – BATCH
# ===================================================
def predict_in_chunks(df, model_name="rf", chunk_size=4000):
    feat_df = prepare_features_df(df)
    n = len(feat_df)
    preds, probs = [], []

    st.info(f"Processing {n:,} rows...")
    progress = st.progress(0)

    for i in range(math.ceil(n / chunk_size)):
        s, e = i * chunk_size, min((i + 1) * chunk_size, n)
        batch = feat_df.iloc[s:e].values.tolist()
        try:
            r = requests.post(f"{API_BATCH}?model={model_name}", json={"features": batch}, timeout=300)
            out = r.json()
        except Exception:
            st.error(f"Chunk {i+1}: Error contacting server")
            return None
        preds.extend(out.get("predictions", []))
        probs.extend(out.get("probabilities", []))
        progress.progress(e / n)

    out_df = df.reset_index(drop=True).iloc[:n]
    out_df["prediction"] = preds
    out_df["fraud_probability"] = probs
    st.session_state.out_df = out_df
    return out_df

# ===================================================
# MANUAL MODE UI (APPLE GLASS STYLE)
# ===================================================
if mode == "Manual Input (6 values)":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧮 Manual Fraud Prediction – Enter 6 Basic Financial Features")

    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Feature 1", 0.0)
        f2 = st.number_input("Feature 2", 0.0)
        f3 = st.number_input("Feature 3", 0.0)
    with col2:
        f4 = st.number_input("Feature 4", 0.0)
        f5 = st.number_input("Feature 5", 0.0)
        f6 = st.number_input("Feature 6", 0.0)

    if st.button("🚀 Predict Fraud Now", use_container_width=True):
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        result, status = call_api(features, model)

        if status == 200:
            pred = result.get("prediction")
            prob = round(float(result.get("fraud_probability")) * 100, 2)

            st.write("<div class='glass-card'>", unsafe_allow_html=True)

            if pred == 1:
                st.error(f"⚠️ HIGH RISK FRAUD DETECTED – Probability: {prob}%")
            else:
                st.success(f"✅ Transaction Looks Legitimate – Probability Fraud: {prob}%")

            st.markdown(f"<p class='prob-box'>Fraud Likelihood: <b>{prob}%</b></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# CSV MODE
# ===================================================
if mode == "Upload CSV File":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload CSV for Bulk Fraud Analysis")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if "Class" in df:
            st.session_state.y_true = df["Class"].copy()

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            out_df = predict_in_chunks(df, model)

            if out_df is not None:
                st.success("Batch Prediction Completed")
                st.dataframe(out_df.head())

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions CSV", csv, "predictions.csv")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# DEVELOPER CREDIT (PER YOUR REQUEST)
# ===================================================
st.markdown("""
<div class='credit'>
    Developed with ❤️ by <a href='https://github.com/SRIHARSHA-BHARAD
# ===================================================
# DEVELOPER CREDIT (PER YOUR REQUEST)
# ===================================================
st.markdown("""
<div class='credit'>
    Developed with ❤️ by <a href='https://github.com/SRIHARSHA-BHARADWAJ' target='_blank'>SRIHARSHA-BHARADWAJ</a><br>
    Premium Apple‑grade UI • Powered by Machine Learning ⚡
</div>
""", unsafe_allow_html=True)

