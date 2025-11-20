# OBSIDIAN QUANTUM EDITION — FINAL VERSION (Hybrid Apple × NVIDIA)
# - Compact gauge
# - Larger text everywhere
# - Removed empty transparent blocks
# - No whitespace waste
# - Ultra-premium liquid glass UI
# - Fully corrected, no syntax errors

import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math
from typing import List

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection — Obsidian Quantum",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# PREMIUM DARK THEME (Hybrid Apple × NVIDIA)
# ----------------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f14;
  --panel: rgba(255,255,255,0.03);
  --glass: rgba(255,255,255,0.05);
  --muted:#9aa5b1;
  --accent-nv:#76ff03;
  --accent-blue:#0aa6ff;
  --white-soft: rgba(255,255,255,0.92);
}

/* BACKGROUND */
.stApp {
  background: radial-gradient(900px 500px at 10% 10%, rgba(20,30,40,0.35), transparent 60%),
              radial-gradient(1100px 700px at 90% 90%, rgba(6,8,14,0.4), transparent 70%),
              var(--bg);
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
}

/* HERO */
.hero {
  background: linear-gradient(180deg, var(--glass), rgba(255,255,255,0.02));
  border-radius: 20px;
  padding: 30px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.6);
  border: 1px solid rgba(255,255,255,0.05);
  backdrop-filter: blur(10px);
}

.title {
  font-size: 38px;
  font-weight: 900;
  color: var(--white-soft);
  letter-spacing: -0.5px;
}

.subtitle {
  color: var(--muted);
  font-size: 15px;
  margin-top: 6px;
}

/* CARDS */
.card {
  background: linear-gradient(180deg, var(--glass), rgba(255,255,255,0.02));
  border-radius: 16px;
  padding: 18px;
  border: 1px solid rgba(255,255,255,0.05);
  box-shadow: 0 6px 20px rgba(0,0,0,0.45);
  backdrop-filter: blur(8px);
  transition: 0.18s ease;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(0,0,0,0.55);
}

/* BUTTONS */
.stButton>button {
  background: linear-gradient(90deg, rgba(118,255,3,0.14), rgba(0,174,255,0.12));
  border: 1px solid rgba(118,255,3,0.18);
  color: var(--white-soft);
  padding: 10px 14px;
  border-radius: 10px;
  font-weight: 800;
  font-size: 15px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.5);
}

/* BADGES */
.badge-low {
  background: rgba(16,185,129,0.10);
  color:#10b981;
  padding:8px 12px;
  border-radius:14px;
  font-weight:800;
  font-size:16px;
  border:1px solid rgba(16,185,129,0.14);
}

.badge-med {
  background: rgba(249,115,22,0.10);
  color:#f97316;
  padding:8px 12px;
  border-radius:14px;
  font-weight:800;
  font-size:16px;
  border:1px solid rgba(249,115,22,0.14);
}

.badge-high {
  background: rgba(239,68,68,0.10);
  color:#ef4444;
  padding:8px 12px;
  border-radius:14px;
  font-weight:900;
  font-size:17px;
  border:1px solid rgba(239,68,68,0.18);
  box-shadow:0 0 20px rgba(239,68,68,0.12);
}

/* TEXT SMALL */
.muted {
  color: var(--muted);
  font-size:14px;
}

/* CREDIT */
.credit {
  text-align:center;
  font-size:14px;
  color: var(--muted);
  margin-top: 12px;
}

.credit a {
  color: var(--accent-nv);
  font-weight:800;
  text-decoration:none;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# COMPACT PREMIUM GAUGE
# ----------------------------
def render_gauge(percent: float, size: int = 200):
    pct = max(0, min(100, float(percent)))
    r = 80
    circ = 2 * math.pi * r
    filled = circ * (pct / 100)
    empty = circ - filled

    return f"""
    <div style="display:flex;justify-content:center;">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#76ff03"/>
            <stop offset="50%" stop-color="#00d1ff"/>
            <stop offset="100%" stop-color="#ff8a00"/>
          </linearGradient>
          <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <g transform="translate(110,110)">
          <circle r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="16"/>
          <circle r="{r}" fill="none" stroke="url(#grad)" stroke-width="16"
            stroke-linecap="round"
            stroke-dasharray="{filled} {empty}"
            stroke-dashoffset="{circ*0.25}"
            transform="rotate(-90)"
            filter="url(#glow)"
            style="transition: stroke-dasharray 0.8s ease;">
          </circle>

          <text x="0" y="-4" text-anchor="middle" font-size="30" font-weight="900" fill="#e6faff">{pct:.2f}%</text>
          <text x="0" y="20" text-anchor="middle" font-size="13" fill="#9aa5b1">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """

# ----------------------------
# FEATURE ORDERING
# ----------------------------
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

def prepare_features_df(df):
    df = df.copy().reset_index(drop=True)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    present = [c for c in KAGGLE_ORDER if c in df.columns]

    if len(present) >= 12:
        for c in KAGGLE_ORDER:
            if c not in df.columns:
                df[c] = 0.0
        return df[KAGGLE_ORDER].astype(float)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) >= 30:
        out = df[numeric[:30]].copy()
        out.columns = KAGGLE_ORDER
        return out

    for i in range(30 - df.shape[1]):
        df[f"pad_{i}"] = 0.0

    feat = df.iloc[:, :30].copy()
    feat.columns = KAGGLE_ORDER
    return feat.astype(float)

# ----------------------------
# BACKEND ENDPOINTS
# ----------------------------
API = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = API + "/predict"
API_BATCH = API + "/predict-batch"
API_MODELS = API + "/get-models"

def safe_post(url, payload, timeout=200):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

# ----------------------------
# SESSION STATE
# ----------------------------
for key in ["last_prob", "last_res", "out_df", "logs"]:
    if key not in st.session_state:
        st.session_state[key] = None if key!="logs" else []

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional • Ultra-grade precision • Real-time fraud scoring</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("### Model")
try:
    models_list = requests.get(API_MODELS, timeout=4).json().get("available_models", [])
except:
    models_list = ["logreg","rf"]

model = st.sidebar.radio("Choose model:", models_list)

st.sidebar.markdown("### Mode")
mode = st.sidebar.selectbox("Select:", ["Manual (6 features)", "CSV Bulk"])

sensitivity = st.sidebar.slider("Risk threshold", 40.0, 90.0, 60.0, 1.0)

show_raw = st.sidebar.checkbox("Show backend raw response")

# ----------------------------
# LAYOUT
# ----------------------------
left, right = st.columns([2,1])

# ----------------------------
# LEFT — MANUAL + CSV
# ----------------------------
with left:

    # MANUAL
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single Prediction (Manual Input)")
    st.markdown('<div class="muted">Enter six numeric features. Remaining are auto-padded.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Feature 1", 0.0)
        f2 = st.number_input("Feature 2", 0.0)
        f3 = st.number_input("Feature 3", 0.0)
    with c2:
        f4 = st.number_input("Feature 4", 0.0)
        f5 = st.number_input("Feature 5", 0.0)
        f6 = st.number_input("Feature 6", 0.0)

    if st.button("Run Single Prediction"):
        features = [f1,f2,f3,f4,f5,f6] + [0.0]*24
        st.info("Requesting backend...")
        out, code = safe_post(f"{API_SINGLE}?model={model}", {"features":features})

        if code != 200:
            st.error(f"Backend error {code}")
            st.session_state.logs.append(out)
        else:
            st.success("Prediction received")
            st.session_state.last_res = out
            prob = out.get("fraud_probability")
            try:
                pct = round(float(prob)*100,2)
            except:
                pct = None
            st.session_state.last_prob = pct

            # Badge logic
            if pct is None:
                st.markdown('<div class="muted">No probability returned.</div>', unsafe_allow_html=True)
            else:
                high = sensitivity
                mid = sensitivity*0.6
                if pct>=high:
                    st.markdown(f"<div class='badge-high'>HIGH RISK — {pct}%</div>", unsafe_allow_html=True)
                elif pct>=mid:
                    st.markdown(f"<div class='badge-med'>ELEVATED — {pct}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-low'>LOW RISK — {pct}%</div>", unsafe_allow_html=True)

            if show_raw:
                st.json(out)

    st.markdown('</div>', unsafe_allow_html=True)

    # CSV BULK
    st.markdown('<div class="card" style="margin-top:18px">', unsafe_allow_html=True)
    st.subheader("Bulk CSV Prediction")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            st.dataframe(df.head())

            if st.button("Run Bulk Prediction"):
                st.info("Preparing...")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                preds, probs = [], []
                bar = st.progress(0)

                chunk = 4000
                for i in range(math.ceil(n/chunk)):
                    s = i*chunk
                    e = min((i+1)*chunk, n)
                    batch = feat_df.iloc[s:e].values.tolist()

                    out, code = safe_post(f"{API_BATCH}?model={model}", {"features": batch})

                    if code != 200:
                        st.error(f"Chunk {i+1} failed")
                        break

                    preds += out.get("predictions",[])
                    probs += out.get("probabilities",[])
                    bar.progress(e/n)

                if len(preds)==n:
                    df_out = df.copy()
                    df_out["prediction"]=preds
                    df_out["probability"]=probs
                    st.session_state.out_df = df_out
                    st.success("Completed")
                    st.dataframe(df_out.head())
                    st.download_button("Download CSV", df_out.to_csv(index=False).encode("utf-8"))

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# RIGHT — GAUGE + LOGS
# ----------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Fraud Gauge")

    if st.session_state.last_prob is not None:
        st.components.v1.html(
            render_gauge(st.session_state.last_prob),
            height=260
        )
        st.markdown('<div class="muted">Adjust threshold in sidebar.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">Run a prediction to show gauge.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:18px">', unsafe_allow_html=True)
    st.subheader("Logs")
    if len(st.session_state.logs)==0:
        st.markdown('<div class="muted">No logs yet.</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.logs[-8:]:
            st.write(entry)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
<div class="credit">
Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a> · Obsidian Quantum Edition
</div>
""", unsafe_allow_html=True)
