# app.py -- Premium Credit Card Fraud Detection Frontend (Professional / Apple-style UI)
# - Ultra-polished glass UI
# - Dynamic SVG probability gauge + clear sentence
# - Manual input + CSV batch prediction
# - Robust feature ordering + padding
# - Model selection + backend error handling
# - Developer credit (no heart) linking to GitHub account SRIHARSHA-BHARADWAJ
# - No plotting libraries required

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
    page_title="Credit Card Fraud Detection — Premium",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# CSS / STYLES: Glassmorphism + Professional
# ----------------------------
st.markdown(
    """
    <style>
    /* Basic body */
    .stApp {
        background: linear-gradient(180deg, #eef4f8 0%, #fcfdff 100%);
        color-scheme: light;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Big container card */
    .top-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.70), rgba(255,255,255,0.55));
        box-shadow: 0 10px 30px rgba(12, 24, 40, 0.08);
        border-radius: 18px;
        padding: 28px 28px;
        border: 1px solid rgba(15, 23, 41, 0.04);
        backdrop-filter: blur(10px);
    }

    .headline {
        font-size: 44px;
        font-weight: 800;
        margin: 0;
        line-height: 1;
        background: linear-gradient(90deg, #0b78ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .subhead {
        font-size: 16px;
        color: #4b5563;
        margin-top: 6px;
        margin-bottom: 16px;
    }

    /* Small card areas */
    .card {
        background: rgba(255,255,255,0.65);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(15, 23, 41, 0.04);
        box-shadow: 0 6px 18px rgba(12, 24, 40, 0.04);
    }

    /* Gauge area */
    .gauge-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }

    .prob-title {
        font-weight: 700;
        font-size: 14px;
        color: #0f172a;
        margin-bottom: 6px;
    }

    .prob-text {
        font-size: 18px;
        color: #111827;
        margin-top: 6px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0b78ff, #00c6ff);
        color: white;
        font-weight: 700;
        border: none;
        padding: 10px 14px;
        border-radius: 10px;
    }

    .muted {
        color: #6b7280;
        font-size: 13px;
    }

    /* Footer credit */
    .credit {
        text-align: center;
        font-size: 13px;
        color: #6b7280;
        margin-top: 26px;
    }
    .credit a { color: #0b78ff; font-weight: 700; text-decoration: none; }

    /* Responsive columns inside streamlit */
    @media (max-width: 900px) {
        .headline { font-size: 34px; }
    }

    /* subtle pulse for high-risk label */
    .pulse {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(255,50,50,0.08), rgba(255,50,50,0.04));
        border: 1px solid rgba(255,50,50,0.12);
        color: #b91c1c;
        animation: pulse 1.8s infinite ease-in-out;
        font-weight: 700;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(185,28,28,0.10); }
        70% { box-shadow: 0 0 0 8px rgba(185,28,28,0); }
        100% { box-shadow: 0 0 0 0 rgba(185,28,28,0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helper: SVG Gauge generator (inline HTML)
# ----------------------------
def render_probability_gauge(percent: float, size: int = 220) -> str:
    """
    Returns an HTML string with an SVG circular gauge showing 'percent' (0-100).
    Animated stroke and a color gradient from green->yellow->red depending on percent.
    """
    pct = max(0.0, min(100.0, float(percent)))
    # Stroke dash calculation for circle (radius 90)
    radius = 90
    circumference = 2 * math.pi * radius
    filled = (pct / 100.0) * circumference
    empty = circumference - filled

    # choose color
    if pct < 20:
        color = "#10b981"  # green
    elif pct < 50:
        color = "#f59e0b"  # amber
    else:
        color = "#ef4444"  # red

    svg = f"""
    <div style="width:{size}px; text-align:center;">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#06b6d4" />
            <stop offset="50%" stop-color="#0ea5e9" />
            <stop offset="100%" stop-color="{color}" />
          </linearGradient>
        </defs>

        <g transform="translate(110,110)">
          <!-- background circle -->
          <circle r="{radius}" fill="none" stroke="#e6eef6" stroke-width="18" />
          <!-- foreground -->
          <circle r="{radius}" fill="none" stroke="url(#g1)" stroke-width="18"
            stroke-linecap="round"
            stroke-dasharray="{filled} {empty}"
            stroke-dashoffset="{circumference * 0.25}"
            transform="rotate(-90)"
            style="transition: stroke-dasharray 1s ease, stroke 0.5s ease;">
          </circle>

          <!-- center text -->
          <text x="0" y="-10" text-anchor="middle" font-size="26" font-weight="700" fill="#0b1220">{pct:.2f}%</text>
          <text x="0" y="18" text-anchor="middle" font-size="12" fill="#374151">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """
    return svg


# ----------------------------
# Session state initialization
# ----------------------------
if "y_true" not in st.session_state:
    st.session_state.y_true = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# ----------------------------
# API endpoints (backend)
# ----------------------------
API_SINGLE = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"
API_BATCH = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict-batch"
API_MODELS = "https://credit-card-fraud-detection-ml-webapp.onrender.com/get-models"

# safe fetch of models
try:
    models_list = requests.get(API_MODELS, timeout=5).json().get("available_models", ["logreg", "rf"])
except Exception:
    models_list = ["logreg", "rf"]

# ----------------------------
# Feature order helpers
# ----------------------------
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with EXACTLY 30 columns in KAGGLE_ORDER.
    If input contains named columns V1..V28/Amount/Time -> reorder and fill missing with zeros.
    Otherwise uses numeric columns in order and pads to 30 features.
    """
    df_local = df.copy().reset_index(drop=True)
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])

    # If all or many of expected columns present -> reorder with padding for missing
    present = [c for c in KAGGLE_ORDER if c in df_local.columns]
    if len(present) >= 12:  # decent chance this is a labelled Kaggle-like DF
        for c in KAGGLE_ORDER:
            if c not in df_local.columns:
                df_local[c] = 0.0
        feat_df = df_local[KAGGLE_ORDER].astype(float)
        return feat_df

    # Else fallback to numeric columns
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 30:
        feat_df = df_local[numeric_cols[:30]].copy()
        feat_df.columns = KAGGLE_ORDER
        return feat_df

    # Pad missing columns if fewer than 30
    for i in range(30 - df_local.shape[1]):
        df_local[f"pad_{i}"] = 0.0
    feat_df = df_local.iloc[:, :30].copy()
    feat_df.columns = KAGGLE_ORDER
    feat_df = feat_df.fillna(0.0).astype(float)
    return feat_df


# ----------------------------
# API calls
# ----------------------------
def call_single_api(features: List[float], model_selected: str = "rf"):
    payload = {"features": features}
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, timeout=12)
        r.raise_for_status()
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500


def call_batch_api(batch: List[List[float]], model_selected: str = "rf", timeout: int = 300):
    payload = {"features": batch}
    try:
        r = requests.post(f"{API_BATCH}?model={model_selected}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500


# ----------------------------
# UI LAYOUT - header + description
# ----------------------------
with st.container():
    st.markdown('<div class="top-card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center;">', unsafe_allow_html=True)
    st.markdown('<div style="flex:1">', unsafe_allow_html=True)
    st.markdown('<h1 class="headline">Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subhead">Enterprise-level prediction — clear probability, concise guidance, and bulk processing.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.markdown("### Model & Input")
model = st.sidebar.radio("Select model", models_list)
mode = st.sidebar.selectbox("Input method", ["Manual (6 features)", "CSV Upload (bulk)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Advanced (optional)")
st.sidebar.checkbox("Show raw backend responses in logs", key="show_raw", value=False)
st.sidebar.markdown("")

# ----------------------------
# Main interactive area: two columns
# ----------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    # Manual input card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Manual prediction")
    st.markdown('<div class="muted">Enter six representative numeric features. Remaining features will be zero-padded to 30 inputs expected by the model.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Feature 1 (numeric)", value=0.0, format="%.6f", step=0.01, key="f1")
        f2 = st.number_input("Feature 2 (numeric)", value=0.0, format="%.6f", step=0.01, key="f2")
        f3 = st.number_input("Feature 3 (numeric)", value=0.0, format="%.6f", step=0.01, key="f3")
    with c2:
        f4 = st.number_input("Feature 4 (numeric)", value=0.0, format="%.6f", step=0.01, key="f4")
        f5 = st.number_input("Feature 5 (numeric)", value=0.0, format="%.6f", step=0.01, key="f5")
        f6 = st.number_input("Feature 6 (numeric)", value=0.0, format="%.6f", step=0.01, key="f6")

    st.write("")  # spacing
    run_manual = st.button("Run Single Prediction", key="run_manual")

    if run_manual:
        # build 30-length input vector: place 6 inputs into first V1..V6 slots, rest zeros
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        st.info("Contacting backend for prediction...")
        out, status = call_single_api(features, model_selected=model)
        if status != 200:
            st.error(f"Prediction failed: {out.get('error')}")
            st.session_state.logs.append({"type": "error", "msg": str(out)})
        else:
            pred = out.get("prediction")
            prob = out.get("fraud_probability")
            try:
                prob_pct = round(float(prob) * 100.0, 2) if prob is not None else None
            except Exception:
                prob_pct = None

            # Store last
            st.session_state.last_result = out
            st.session_state.last_pred = pred
            st.session_state.last_prob = prob_pct

            # Professional sentence and display
            if prob_pct is None:
                st.warning("Model returned prediction but no probability. Displaying discrete decision only.")
            else:
                if prob_pct >= 70:
                    st.markdown(f"<div class='pulse'>HIGH RISK — Fraud probability {prob_pct}%</div>", unsafe_allow_html=True)
                elif prob_pct >= 40:
                    st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:#fff7ed;border:1px solid #fcd34d;color:#92400e;font-weight:700;'>ELEVATED RISK — {prob_pct}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:#ecfdf5;border:1px solid #34d399;color:#065f46;font-weight:700;'>LOW RISK — {prob_pct}%</div>", unsafe_allow_html=True)

            # Show gauge on right column (update via session state)
            with col_right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div style="display:flex; flex-direction:column; align-items:center;">', unsafe_allow_html=True)
                if prob_pct is not None:
                    st.components.v1.html(render_probability_gauge(prob_pct, size=240), height=300)
                    # A short professional guidance sentence
                    if prob_pct >= 70:
                        st.markdown('<div class="muted">Recommendation: Block and escalate to fraud team for manual investigation.</div>', unsafe_allow_html=True)
                    elif prob_pct >= 40:
                        st.markdown('<div class="muted">Recommendation: Apply secondary verification (OTP / 2FA).</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="muted">Recommendation: Allow transaction; monitor for anomalies.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="muted">No probability available to render gauge.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # optionally show raw backend output (for debugging)
            if st.session_state.get("show_raw", False):
                st.write("Raw backend response:")
                st.json(out)

    st.markdown("</div>", unsafe_allow_html=True)

    # CSV Upload card
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Bulk prediction (CSV)")
    st.markdown('<div class="muted">Upload a CSV containing features. If your CSV has a "Class" column, it will be used as true labels for local reference.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file (csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())

            # show detected columns
            st.markdown("**Detected columns:**")
            st.write(df.columns.tolist())

            # store labels if available
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"].copy()

            if st.button("Run batch prediction on CSV", key="run_batch"):
                # Prepare features & do chunked prediction
                st.info("Preparing features and calling backend in chunks...")
                feat_df = prepare_features_df(df)
                # Confirm number rows
                n = len(feat_df)
                st.write(f"Prepared {n} rows and {feat_df.shape[1]} features (sent in model order).")

                # chunking
                chunk_size = 4000
                preds = []
                probs = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                start = time.time()
                error_flag = False

                for i in range(math.ceil(n / chunk_size)):
                    s = i * chunk_size
                    e = min((i + 1) * chunk_size, n)
                    batch = feat_df.iloc[s:e].values.tolist()
                    status_text.info(f"Processing chunk {i+1} ({s}:{e})...")
                    out, stat = call_batch_api(batch, model_selected=model, timeout=300)
                    if stat != 200:
                        st.error(f"Batch call failed at chunk {i+1}: {out.get('error')}")
                        st.session_state.logs.append({"type": "error", "msg": str(out)})
                        error_flag = True
                        break
                    preds.extend(out.get("predictions", []))
                    probs.extend(out.get("probabilities", []))
                    progress_bar.progress(min(1.0, (e / n)))
                    # small throttle to keep UI responsive
                    time.sleep(0.1)

                if not error_flag:
                    # build output DataFrame
                    out_df = df.reset_index(drop=True).iloc[: n].copy()
                    out_df["prediction"] = preds
                    out_df["fraud_probability"] = probs
                    st.session_state.out_df = out_df
                    st.success("Batch prediction complete.")
                    st.dataframe(out_df.head())

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

                    duration = time.time() - start
                    st.write(f"Processed {n} rows in {duration:.2f} seconds.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # Right column cards: quick status, logs, last result
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Status & Logs")
    if len(st.session_state.logs) == 0:
        st.markdown('<div class="muted">No logs yet. Backend calls will appear here if enabled.</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.logs[-10:]:
            st.write(entry)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Last Result (single prediction)")
    if st.session_state.last_result is None:
        st.markdown('<div class="muted">No single prediction has been run in this session.</div>', unsafe_allow_html=True)
    else:
        out = st.session_state.last_result
        st.json(out)
        if st.session_state.last_prob is not None:
            st.components.v1.html(render_probability_gauge(st.session_state.last_prob, size=200), height=280)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer credit area (professional; no heart)
# ----------------------------
st.markdown(
    """
    <div class="credit">
        Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a>
        · Premium UI · AI-powered fraud detection
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpful notes for operator
# ----------------------------
st.markdown("")
st.markdown(
    """
    <div style="font-size:12px;color:#6b7280;margin-top:10px;">
      Notes: The frontend sends numeric feature vectors (30 values) to the backend. If your CSV columns match the familiar V1..V28, Amount, Time names,
      the frontend will align and reorder them automatically. Otherwise, numeric columns are used in their current order and padded as needed.
      For best results, train and serve the backend with the exact same feature ordering used during training.
    </div>
    """,
    unsafe_allow_html=True,
)

# End of file
