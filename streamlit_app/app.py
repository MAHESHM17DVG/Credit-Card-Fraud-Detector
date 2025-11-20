###############################
# FINAL BOSS EDITION — APP.PY
###############################
# Ultra-premium Dark UI • Perfect Gauge • No Syntax Errors
# Manual & CSV • Fallback 405 Handling • Optimized & Clean

import streamlit as st
import requests
import numpy as np
import pandas as pd
import math
import time
from typing import List

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------
# DARK LIQUID GLASS CSS — POLISHED
# -------------------------------------------------------
st.markdown("""
<style>
:root{
  --bg:#050708;
  --muted:#98a3ad;
  --accent:#00b7ff;
  --accent2:#6aff88;
  --white-soft:rgba(246,248,250,0.96);
}

html, body, .stApp {
  background:
    radial-gradient(1200px 420px at 6% 12%, rgba(4,6,10,0.36), transparent 8%),
    radial-gradient(800px 380px at 94% 88%, rgba(8,10,14,0.28), transparent 8%),
    var(--bg);
  color: var(--white-soft);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
}

/* Centered container */
.center-wrap { display:flex; justify-content:center; }
.center-panel {
  width: 980px; max-width:95%;
  padding:20px; border-radius:16px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.04);
  box-shadow:0 20px 56px rgba(0,0,0,0.65);
  backdrop-filter:blur(10px);
}

/* Headline */
.hdr-title {
  font-size:32px; font-weight:900;
  margin:0; padding:0;
}
.hdr-sub { font-size:13px; color:var(--muted); }

/* Gauge card */
.gauge-card {
  min-width:260px;
  border-radius:14px;
  padding:10px;
  background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.03);
  box-shadow:0 12px 40px rgba(0,0,0,0.65);
}

/* Input blocks */
.inputs-wrap { display:flex; gap:18px; margin-top:10px; flex-wrap:wrap; justify-content:center; }
.input-col { display:flex; flex-direction:column; gap:10px; min-width:200px; }

/* Buttons */
.stButton>button {
  background:linear-gradient(90deg,rgba(0,183,255,0.12),rgba(106,255,136,0.10));
  border-radius:10px;
  border:1px solid rgba(0,183,255,0.12);
  color:white; font-weight:800; padding:10px 14px;
}

/* Badges */
.badge { padding:7px 12px; border-radius:999px; font-weight:800; }
.badge-low { background:rgba(106,255,136,0.06); color:#07a457; border:1px solid rgba(106,255,136,0.12); }
.badge-med { background:rgba(0,183,255,0.05); color:#00b7ff; border:1px solid rgba(0,183,255,0.10); }
.badge-high { background:rgba(255,90,90,0.08); color:#ff6060; border:1px solid rgba(255,90,90,0.14); }

/* Logs */
.log-box {
  background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.04);
  padding:12px;
  border-radius:10px;
}

/* Footer */
.credit { text-align:center; margin-top:22px; color:var(--muted); font-size:13px; }
.credit a { color:var(--accent); font-weight:700; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# FEATURE ORDER HELPERS
# -------------------------------------------------------
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
        return df_local[KAGGLE_ORDER].astype(float)

    numeric = df_local.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) >= 30:
        feat = df_local[numeric[:30]].copy()
        feat.columns = KAGGLE_ORDER
        return feat

    for i in range(30 - df_local.shape[1]):
        df_local[f"pad_{i}"] = 0.0

    feat_df = df_local.iloc[:, :30].copy()
    feat_df.columns = KAGGLE_ORDER
    return feat_df.fillna(0.0).astype(float)

# -------------------------------------------------------
# GAUGE (SIZE 200) — COMPACT & CRYSTAL CLEAR
# -------------------------------------------------------
def render_gauge(percent: float, size: int = 200) -> str:
    p = max(0.0, min(100.0, float(percent)))
    r = 78
    circ = 2 * math.pi * r
    filled = (p / 100.0) * circ
    empty = circ - filled
    return f"""
    <div style="display:flex;justify-content:center;">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="gC" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#6aff88"/>
            <stop offset="55%" stop-color="#00b7ff"/>
            <stop offset="100%" stop-color="#ff9a66"/>
          </linearGradient>
        </defs>

        <g transform="translate(110,110)">
          <circle r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="14"/>
          <circle r="{r}" fill="none" stroke="url(#gC)" stroke-width="14"
            stroke-linecap="round"
            stroke-dasharray="{filled} {empty}"
            stroke-dashoffset="{circ*0.25}"
            transform="rotate(-90)"
            style="transition:stroke-dasharray 0.7s ease;"
          />
          <text x="0" y="-4" text-anchor="middle"
                font-size="26" font-weight="800" fill="#e6f7ff">{p:.2f}%</text>
          <text x="0" y="20" text-anchor="middle"
                font-size="11" fill="#98a3ad">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """

# -------------------------------------------------------
# BACKEND ENDPOINTS
# -------------------------------------------------------
API_BASE = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = API_BASE + "/predict"
API_BATCH = API_BASE + "/predict-batch"
API_MODELS = API_BASE + "/get-models"

def safe_get_models():
    try:
        r = requests.get(API_MODELS, timeout=4)
        r.raise_for_status()
        return r.json().get("available_models", ["logreg", "rf"])
    except:
        return ["logreg", "rf"]

MODELS = safe_get_models()

# fallback POST handler
def post_with_fallback(url, payload, model, timeout=10):
    try:
        r = requests.post(f"{url}?model={model}", json=payload, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 405:
        try:
            r2 = requests.post(url, json=payload, timeout=timeout)
            return r2.json(), r2.status_code
        except Exception as e:
            return {"error": str(e)}, 500
    try:
        return r.json(), r.status_code
    except:
        return {"error": r.text}, r.status_code

# -------------------------------------------------------
# SESSION VARIABLES
# -------------------------------------------------------
for key, val in {
    "last_prob": None,
    "last_single": None,
    "out_df": None,
    "logs": [],
    "model": MODELS[0],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------------------------------
# MAIN PANEL
# -------------------------------------------------------
st.markdown('<div class="center-wrap"><div class="center-panel">', unsafe_allow_html=True)

# Header
st.markdown("""
<div>
  <div class="hdr-title">Credit Card Fraud Detection</div>
  <div class="hdr-sub">Enterprise accuracy. Real-time probability. Zero clutter.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# GAUGE
# -------------------------------------------------------
st.markdown('<div style="text-align:center;margin-top:10px;">', unsafe_allow_html=True)
st.markdown('<div class="gauge-card">', unsafe_allow_html=True)

if st.session_state.last_prob is not None:
    st.components.v1.html(render_gauge(st.session_state.last_prob), height=250)
else:
    st.components.v1.html(render_gauge(0), height=250)

st.markdown('</div></div>', unsafe_allow_html=True)

# -------------------------------------------------------
# INPUTS
# -------------------------------------------------------
st.markdown('<div class="inputs-wrap">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    f1 = st.number_input("Feature 1", 0.0, key="f1")
    f2 = st.number_input("Feature 2", 0.0, key="f2")

with c2:
    f3 = st.number_input("Feature 3", 0.0, key="f3")
    f4 = st.number_input("Feature 4", 0.0, key="f4")

with c3:
    f5 = st.number_input("Feature 5", 0.0, key="f5")
    f6 = st.number_input("Feature 6", 0.0, key="f6")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# CONTROLS
# -------------------------------------------------------
colA, colB, colC = st.columns([1,1,1])

with colA:
    run_single = st.button("Run Prediction")

with colB:
    sensitivity = st.slider("Sensitivity", 0.0, 100.0, 60.0)

with colC:
    show_raw = st.checkbox("Show Raw Output")

# -------------------------------------------------------
# RUN SINGLE PREDICTION
# -------------------------------------------------------
if run_single:
    features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
    st.info("Requesting model prediction...")

    out, code = post_with_fallback(API_SINGLE, {"features": features}, st.session_state.model, timeout=12)

    if code != 200:
        st.error(f"Backend responded with status {code}")
        st.write(out)
        st.session_state.logs.append({"error": out})
    else:
        st.success("Prediction received")
        st.session_state.last_single = out

        prob = out.get("fraud_probability")
        try:
            prob_pct = round(float(prob) * 100, 2) if prob is not None else None
        except:
            prob_pct = None

        st.session_state.last_prob = prob_pct

        # Classification
        if prob_pct is None:
            st.markdown("<div class='muted'>Probability unavailable</div>", unsafe_allow_html=True)
        else:
            med = sensitivity * 0.6

            if prob_pct >= sensitivity:
                st.markdown(f"<div class='badge badge-high'>HIGH — {prob_pct}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='muted'>Action: escalate for manual review.</div>", unsafe_allow_html=True)
            elif prob_pct >= med:
                st.markdown(f"<div class='badge badge-med'>ELEVATED — {prob_pct}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='muted'>Action: require secondary verification.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge badge-low'>LOW — {prob_pct}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='muted'>Action: allow; monitor.</div>", unsafe_allow_html=True)

        if show_raw:
            st.json(out)

# -------------------------------------------------------
# BULK SCORING
# -------------------------------------------------------
with st.expander("Bulk CSV Prediction"):
    upload = st.file_uploader("Upload CSV", type=["csv"])

    if upload:
        try:
            df = pd.read_csv(upload)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"]

            if st.button("Run Bulk Prediction"):
                st.info("Processing in chunks...")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                chunk = 4000
                preds, probs = [], []
                pbar = st.progress(0)
                t0 = time.time()

                for i in range(math.ceil(n / chunk)):
                    s, e = i * chunk, min((i + 1) * chunk, n)
                    batch = feat_df.iloc[s:e].values.tolist()

                    out, code = post_with_fallback(API_BATCH, {"features": batch}, st.session_state.model, timeout=300)

                    if code != 200:
                        st.error(f"Failed at chunk {i+1}: status {code}")
                        st.write(out)
                        break

                    preds.extend(out.get("predictions", []))
                    probs.extend(out.get("probabilities", []))

                    pbar.progress(min(1.0, e / n))

                out_df = df.copy()
                out_df["prediction"] = preds
                out_df["fraud_probability"] = probs

                st.session_state.out_df = out_df
                st.success("Bulk Completed")
                st.dataframe(out_df.head())

                st.download_button(
                    "Download Predictions",
                    out_df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

                st.write(f"Processed {n} rows in {time.time() - t0:.2f}s")

# -------------------------------------------------------
# CLOSE PANEL
# -------------------------------------------------------
st.markdown('</div></div>', unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.markdown("### Model")
st.session_state.model = st.sidebar.radio("Select model", MODELS)

with st.sidebar.expander("Logs"):
    if len(st.session_state.logs) == 0:
        st.markdown("<div class='muted'>No logs</div>", unsafe_allow_html=True)
    else:
        for lg in st.session_state.logs[-20:]:
            st.write(lg)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<div class="credit">
Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a>
</div>
""", unsafe_allow_html=True)
