# app.py -- Professional Premium Credit Card Fraud Detection Frontend
# - Minimal, premium UI
# - Robust backend method handling (explicit 405 handling)
# - Manual + CSV bulk prediction
# - Feature ordering/padding (30 features)
# - SVG gauge + crisp risk messaging
# - Developer credit footer (no emoji/heart)

import streamlit as st
import requests
import numpy as np
import pandas as pd
import math
import time
from typing import List

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection — Professional",
    page_icon="💳",
    layout="wide",
)

# ----------------------------
# Clean CSS (minimal / premium)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f7fafc 0%, #ffffff 100%); color-scheme: light; }
    .panel { background: rgba(255,255,255,0.80); border-radius:14px; padding:18px; box-shadow:0 6px 24px rgba(12,20,30,0.06); border:1px solid rgba(15,23,42,0.03); }
    .title { font-size:38px; font-weight:800; margin:0; background:linear-gradient(90deg,#0a74ff,#06c); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .subtitle { color:#374151; margin-top:6px; margin-bottom:10px; }
    .muted { color:#6b7280; font-size:13px; }
    .btn-primary > button { background: linear-gradient(90deg,#0a74ff,#06c); color:#fff; border-radius:8px; padding:8px 12px; font-weight:700; }
    .risk-badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; }
    .credit { text-align:center; color:#6b7280; font-size:13px; margin-top:22px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Endpoints
# ----------------------------
API_BASE = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = API_BASE + "/predict"
API_BATCH = API_BASE + "/predict-batch"
API_MODELS = API_BASE + "/get-models"

# ----------------------------
# Session state init
# ----------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# ----------------------------
# Try to fetch models list (safe)
# ----------------------------
try:
    models_list = requests.get(API_MODELS, timeout=5).json().get("available_models", ["logreg", "rf"])
except Exception:
    models_list = ["logreg", "rf"]

# ----------------------------
# Feature ordering
# ----------------------------
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy().reset_index(drop=True)
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])
    present = [c for c in KAGGLE_ORDER if c in df_local.columns]
    if len(present) >= 12:
        for c in KAGGLE_ORDER:
            if c not in df_local.columns:
                df_local[c] = 0.0
        feat_df = df_local[KAGGLE_ORDER].astype(float)
        return feat_df
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 30:
        feat_df = df_local[numeric_cols[:30]].copy()
        feat_df.columns = KAGGLE_ORDER
        return feat_df
    for i in range(30 - df_local.shape[1]):
        df_local[f"pad_{i}"] = 0.0
    feat_df = df_local.iloc[:, :30].copy()
    feat_df.columns = KAGGLE_ORDER
    return feat_df.fillna(0.0).astype(float)


# ----------------------------
# SVG gauge (subtle)
# ----------------------------
def render_probability_gauge(percent: float, size: int = 190) -> str:
    pct = max(0.0, min(100.0, float(percent)))
    r = 80
    circ = 2 * math.pi * r
    filled = (pct / 100.0) * circ
    empty = circ - filled
    if pct < 30:
        color = "#059669"
    elif pct < 60:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f"""
    <div style="width:{size}px;text-align:center">
      <svg width="{size}" height="{size}" viewBox="0 0 200 200">
        <g transform="translate(100,100)">
          <circle r="{r}" fill="none" stroke="#eef2f7" stroke-width="14"/>
          <circle r="{r}" fill="none" stroke="{color}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="{filled} {empty}" transform="rotate(-90)"></circle>
          <text x="0" y="-6" text-anchor="middle" font-size="22" font-weight="700" fill="#0b1220">{pct:.2f}%</text>
          <text x="0" y="18" text-anchor="middle" font-size="11" fill="#6b7280">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """


# ----------------------------
# Backend call helpers (robust 405 handling)
# ----------------------------
def post_with_method_fallback_single(features: List[float], model_selected: str):
    """
    Try POST to API_SINGLE?model=<name>. If 405 returned, try POST to API_SINGLE without query param.
    If still fails, return the last response object and status.
    """
    payload = {"features": features}
    headers = {"Content-Type": "application/json"}
    # 1) preferred: POST with query param
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, headers=headers, timeout=12)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        return r.json(), 200
    if r.status_code == 405:
        # fallback: try without querystring
        try:
            r2 = requests.post(API_SINGLE, json=payload, headers=headers, timeout=12)
        except Exception as e:
            return {"error": str(e)}, 500
        if r2.status_code == 200:
            return r2.json(), 200
        # return first 405 detail if present
        try:
            return r2.json(), r2.status_code
        except Exception:
            return {"detail": r2.text or r.text}, r2.status_code
    # other statuses
    try:
        return r.json(), r.status_code
    except Exception:
        return {"detail": r.text}, r.status_code


def post_batch_with_fallback(batch: List[List[float]], model_selected: str, timeout: int = 300):
    payload = {"features": batch}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_BATCH}?model={model_selected}", json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        return r.json(), 200
    if r.status_code == 405:
        try:
            r2 = requests.post(API_BATCH, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            return {"error": str(e)}, 500
        if r2.status_code == 200:
            return r2.json(), 200
        try:
            return r2.json(), r2.status_code
        except Exception:
            return {"detail": r2.text or r.text}, r2.status_code
    try:
        return r.json(), r.status_code
    except Exception:
        return {"detail": r.text}, r.status_code


# ----------------------------
# Layout - Header
# ----------------------------
st.markdown('<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center">', unsafe_allow_html=True)
st.markdown('<div><h1 class="title">Credit Card Fraud Detection</h1><div class="subtitle">Enterprise-grade predictions — clear probabilities and actions.</div></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.markdown("### Controls")
model = st.sidebar.radio("Model", models_list)
mode = st.sidebar.selectbox("Input", ["Manual (6 features)", "CSV Upload (bulk)"])
st.sidebar.checkbox("Show raw backend responses", key="show_raw")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="muted">Contact: https://github.com/SRIHARSHA-BHARADWAJ</div>', unsafe_allow_html=True)

# ----------------------------
# Main columns
# ----------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Manual single prediction")
    st.markdown('<div class="muted">Enter six numeric features; app will pad to 30 features expected by the model.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Feature 1", value=0.0, format="%.6f", key="f1")
        f2 = st.number_input("Feature 2", value=0.0, format="%.6f", key="f2")
        f3 = st.number_input("Feature 3", value=0.0, format="%.6f", key="f3")
    with c2:
        f4 = st.number_input("Feature 4", value=0.0, format="%.6f", key="f4")
        f5 = st.number_input("Feature 5", value=0.0, format="%.6f", key="f5")
        f6 = st.number_input("Feature 6", value=0.0, format="%.6f", key="f6")

    if st.button("Run single prediction"):
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        st.info("Requesting prediction from backend...")
        out, status = post_with_method_fallback_single(features, model_selected=model)

        if status != 200:
            st.error(f"Backend error (status {status}). See details below.")
            st.write(out)
            st.session_state.logs.append({"type": "error", "status": status, "detail": out})
            # Helpful CURL hints when 405
            if status == 405:
                st.warning("Server returned 405 Method Not Allowed. This means the backend refused the HTTP method. Try testing the backend directly with curl:")
                st.code(f"curl -X POST '{API_SINGLE}?model={model}' -H 'Content-Type: application/json' -d '{{\"features\":[0,0,...]}}'", language="bash")
        else:
            # success
            st.session_state.last_result = out
            pred = out.get("prediction")
            prob = out.get("fraud_probability")
            prob_pct = None
            try:
                prob_pct = round(float(prob) * 100.0, 2) if prob is not None else None
            except Exception:
                prob_pct = None

            st.success("Prediction received")
            if prob_pct is None:
                st.markdown('<div class="muted">Discrete prediction only (probability not returned).</div>', unsafe_allow_html=True)
            else:
                # risk badge
                if prob_pct >= 70:
                    st.markdown(f"<div class='risk-badge' style='background:#fff0f0;border:1px solid rgba(220,38,38,0.12);color:#b91c1c'>HIGH RISK — {prob_pct}%</div>", unsafe_allow_html=True)
                elif prob_pct >= 40:
                    st.markdown(f"<div class='risk-badge' style='background:#fffbeb;border:1px solid rgba(245,158,11,0.12);color:#92400e'>ELEVATED — {prob_pct}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-badge' style='background:#ecfdf5;border:1px solid rgba(16,185,129,0.12);color:#065f46'>LOW — {prob_pct}%</div>", unsafe_allow_html=True)

            if st.session_state.get("show_raw"):
                st.json(out)

    st.markdown('</div>', unsafe_allow_html=True)

    # CSV bulk
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Bulk CSV prediction")
    st.markdown('<div class="muted">Upload a CSV. If columns match V1..V28, Amount, Time they will be aligned automatically.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV read failed: {e}")
            df = None

        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head())
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"].copy()
            if st.button("Run batch prediction"):
                st.info("Preparing features")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                st.write(f"Prepared {n} rows with {feat_df.shape[1]} features.")
                chunk_size = 4000
                preds = []
                probs = []
                pbar = st.progress(0)
                status_text = st.empty()
                start = time.time()
                error = False
                for i in range(math.ceil(n / chunk_size)):
                    s = i * chunk_size
                    e = min((i + 1) * chunk_size, n)
                    batch = feat_df.iloc[s:e].values.tolist()
                    status_text.info(f"Processing chunk {i+1}: rows {s}-{e}")
                    out, stcode = post_batch_with_fallback(batch, model_selected=model, timeout=300)
                    if stcode != 200:
                        st.error(f"Batch failed at chunk {i+1}: status {stcode}")
                        st.write(out)
                        st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
                        error = True
                        break
                    preds.extend(out.get("predictions", []))
                    probs.extend(out.get("probabilities", []))
                    pbar.progress(min(1.0, (e / n)))
                    time.sleep(0.08)
                if not error:
                    out_df = df.reset_index(drop=True).iloc[:n].copy()
                    out_df["prediction"] = preds
                    out_df["fraud_probability"] = probs
                    st.session_state.last_result = None
                    st.session_state.last_prob = None
                    st.session_state.out_df = out_df
                    st.success("Batch complete")
                    st.dataframe(out_df.head())
                    buf = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", buf, "predictions.csv", "text/csv")
                    st.write(f"Time: {time.time()-start:.2f}s")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Session & logs")
    if len(st.session_state.logs) == 0:
        st.markdown('<div class="muted">No logs yet</div>', unsafe_allow_html=True)
    else:
        for e in st.session_state.logs[-10:]:
            st.write(e)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Last single result")
    if st.session_state.last_result is None:
        st.markdown('<div class="muted">Not available</div>', unsafe_allow_html=True)
    else:
        st.json(st.session_state.last_result)
        if st.session_state.last_prob is not None:
            st.components.v1.html(render_probability_gauge(st.session_state.last_prob, size=180), height=240)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer credit
st.markdown(
    """
    <div class="credit">
        Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a>
        &nbsp;·&nbsp; Professional UI · AI fraud detection
    </div>
    """,
    unsafe_allow_html=True,
)

# Helpful operator note
st.markdown('<div class="muted" style="margin-top:10px;font-size:12px;">Notes: If you encounter Method Not Allowed (405), ensure your backend accepts POST and CORS is configured for the frontend origin; test using the curl command shown in the error block.</div>', unsafe_allow_html=True)
