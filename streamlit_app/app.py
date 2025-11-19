import streamlit as st
import requests
import numpy as np
import pandas as pd
from utils_plots import plot_roc_curve, plot_precision_recall, plot_confusion_matrix

# ============================
# CONFIGURE PAGE
# ============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ============================
# API URL (Render backend)
# ============================
API_URL = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"

# ============================
# STYLING (Premium UI)
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
model = st.sidebar.radio(
    "Select a Machine Learning Model:",
    ["logreg", "rf"],
    index=0,
)

st.sidebar.write("---")
st.sidebar.info(f"🧠 Using model: **{model.upper()}**")

st.sidebar.write("---")
mode = st.sidebar.selectbox(
    "Choose Input Method:",
    ["Manual Input (5-6 values)", "Upload CSV File"]
)

# ============================
# MAIN LOGIC
# ============================

def call_api(features_list, model_selected):
    """Sends the features to the backend API + returns output."""
    payload = {"features": features_list}
    try:
        response = requests.post(f"{API_URL}?model={model_selected}", json=payload)
        return response.json(), response.status_code
    except:
        return {"error": "Server unreachable"}, 500


# ============================
# MODE 1: MANUAL INPUT (simplified UI)
# ============================
if mode == "Manual Input (5-6 values)":
    st.subheader("🧮 Manual Input Mode")
    st.write("Enter **first 6 features** and system will auto-fill the rest with 0.")

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
            st.json(result)


# ============================
# MODE 2: CSV UPLOAD
# ============================
elif mode == "Upload CSV File":
    st.subheader("📂 Upload CSV File")
    st.write("Upload a CSV containing **30 numeric features per row**.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows", use_container_width=True):
            predictions = []
            st.write("Running predictions...")

            for index, row in df.iterrows():
                features = row.values.tolist()
                result, status = call_api(features, model)

                if status == 200:
                    predictions.append(result)
                else:
                    predictions.append({"error": "API Issue"})

            st.success("Completed!")
            st.write("### Results:")
            st.json(predictions)

st.subheader("📊 Model Performance Visualizations")

# Only works if backend returns true labels + probabilities
if 'y_true' in result and 'y_prob' in result:
    st.write("### ROC Curve")
    st.pyplot(plot_roc_curve(result['y_true'], result['y_prob']))

    st.write("### Precision-Recall Curve")
    st.pyplot(plot_precision_recall(result['y_true'], result['y_prob']))

    st.write("### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(result['y_true'], result['y_pred']))
else:
    st.info("Plotting available only for batch predictions or CSV uploads.")
