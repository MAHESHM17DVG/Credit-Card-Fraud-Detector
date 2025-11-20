# app.py
# Final Premium Hybrid Theme — Cyan + Subtle Green Accent
# - Dark liquid-glass UI
# - Hybrid cyan + green accent
# - Manual + CSV bulk prediction
# - Robust backend method fallback (handles 405)
# - Animated SVG gauge, sensitivity slider, compact professional text
# - Developer credit (professional)
# - Feature-ordering/padding logic retained

import streamlit as st
import requests
import numpy as np
import pandas as pd
import math
import time
from typing import List

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Hybrid accent CSS (cyan + subtle green) — premium dark
# ----------------------------
st.markdown(
    """
    <style>
    :root{
      --bg: #06080a;
      --panel: rgba(255,255,255,0.03);
      --glass: rgba(255,255,255,0.035);
      --muted: #98a3ad;
      --accent-cyan: #00b7ff;
      --accent-green: #6aff88;
      --soft-white: rgba(245,248,250,0.95);
    }

    html, body, .stApp {
      background: radial-gradient(1000px 500px at 8% 12%, rgba(3,7,15,0.36), transparent 6%),
                  radial-gradient(800px 450px at 92% 88%, rgba(2,6,10,0.32), transparent 6%),
                  var(--bg);
      color: var(--soft-white);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Hero */
    .hero {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 16px;
      padding: 22px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.04);
      backdrop-filter: blur(8px) saturate(120%);
      margin-bottom: 14px;
    }
    .title {
      font-size: 36px;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.4px;
    }
    .subtitle { color: var(--muted); margin-top:6px; font-size:13px; }

    /* Card */
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.01));
      border-radius: 12px;
      padding: 14px;
      border: 1px solid rgba(255,255,255,0.03);
      box-shadow: 0 6px 22px rgba(0,0,0,0.6);
      backdrop-filter: blur(6px);
      transition: transform .16s ease, box-shadow .16s ease;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 20px 48px rgba(0,0,0,0.65); }

    /* Buttons */
    .stButton>button {
      background: linear-gradient(90deg, rgba(0,183,255,0.10), rgba(106,255,136,0.06));
      border: 1px solid rgba(0,183,255,0.08);
      color: var(--soft-white);
      padding: 9px 14px;
      border-radius: 10px;
      font-weight: 800;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }

    .muted { color: var(--muted); font-size:13px; }
    .small { font-size:12px; color:var(--muted); }

    /* Badges */
    .badge-low { background: rgba(106,255,136,0.05); color: #05a457; padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid rgba(106,255,136,0.06); }
    .badge-med { background: rgba(0,183,255,0.05); color: #0aa6ff; padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid rgba(0,183,255,0.06); }
    .badge-high { background: rgba(255,90,90,0.05); color: #ff6b6b; padding:6px 10px; border-radius:999px; font-weight:800; border:1px solid rgba(255,90,90,0.08); box-shadow:0 6px 20px rgba(255,90,90,0.04); }

    .credit { text-align:center; color:var(--muted); font-size:13px; margin-top:18px; }
    .credit a { color: var(--accent-cyan); font-weight:700; text-decoration:none; }

    .gauge-wrap { display:flex; align-items:center; justify-content:center; flex-direction:column; gap:8px; }

    .divider { height:1px; width:100%; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); margin:10px 0; }

    @media (max-width: 900px) {
      .title { font-size: 30px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Feature ordering helper
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


# ----------------------------
# Animated SVG gauge (hybrid colors)
# ----------------------------
def render_gauge(percent: float, size: int = 220) -> str:
    p = max(0.0, min(100.0, float(percent)))
    r = 86
    circ = 2 * math.pi * r
    filled = (p / 100.0) * circ
    empty = circ - filled
    return f"""
    <div class="gauge-wrap" style="width:{size}px">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#6aff88"/>
            <stop offset="55%" stop-color="#00b7ff"/>
            <stop offset="100%" stop-color="#ff9a66"/>
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <g transform="translate(110,110)">
          <circle r="{r}" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="18"/>
          <circle r="{r}" fill="none" stroke="url(#g)" stroke-width="18" stroke-linecap="round"
            stroke-dasharray="{filled} {empty}" stroke-dashoffset="{circ*0.25}" transform="rotate(-90)"
            style="transition: stroke-dasharray 900ms cubic-bezier(.2,.9,.3,1), stroke 900ms ease;" filter="url(#glow)"/>
          <circle r="58" fill="rgba(10,12,14,0.6)" stroke="rgba(255,255,255,0.02)" stroke-width="1.2"/>
          <text x="0" y="-10" text-anchor="middle" font-size="28" font-weight="800" fill="#e6f7ff">{p:.2f}%</text>
          <text x="0" y="18" text-anchor="middle" font-size="11" fill="#98a3ad">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """


# ----------------------------
# Backend endpoints & resilient calls
# ----------------------------
API_BASE = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = API_BASE + "/predict"
API_BATCH = API_BASE + "/predict-batch"
API_MODELS = API_BASE + "/get-models"


def try_get_models():
    try:
        r = requests.get(API_MODELS, timeout=4)
        r.raise_for_status()
        return r.json().get("available_models", ["logreg", "rf"])
    except Exception:
        return ["logreg", "rf"]


models_list = try_get_models()


def post_single_with_fallback(features: List[float], model_selected: str):
    payload = {"features": features}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, headers=headers, timeout=12)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        try:
            return r.json(), 200
        except:
            return {"detail": r.text}, r.status_code
    if r.status_code == 405:
        try:
            r2 = requests.post(API_SINGLE, json=payload, headers=headers, timeout=12)
        except Exception as e:
            return {"error": str(e)}, 500
        try:
            return r2.json(), r2.status_code
        except:
            return {"detail": r2.text}, r2.status_code
    try:
        return r.json(), r.status_code
    except:
        return {"detail": r.text}, r.status_code


def post_batch_with_fallback(batch: List[List[float]], model_selected: str, timeout: int = 300):
    payload = {"features": batch}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_BATCH}?model={model_selected}", json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        try:
            return r.json(), 200
        except:
            return {"detail": r.text}, r.status_code
    if r.status_code == 405:
        try:
            r2 = requests.post(API_BATCH, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            return {"error": str(e)}, 500
        try:
            return r2.json(), r2.status_code
        except:
            return {"detail": r2.text}, r2.status_code
    try:
        return r.json(), r.status_code
    except:
        return {"detail": r.text}, r.status_code


# ----------------------------
# Session state init
# ----------------------------
if "last_single" not in st.session_state:
    st.session_state.last_single = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None
if "logs" not in st.session_state:
    st.session_state.logs = []


# ----------------------------
# Top hero (clean)
# ----------------------------
st.markdown('<div class="hero"><div style="display:flex; justify-content:space-between; align-items:center;">', unsafe_allow_html=True)
st.markdown('<div><div class="title">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.markdown("### Controls")
model = st.sidebar.radio("Model", models_list)
mode = st.sidebar.selectbox("Mode", ["Manual (6)", "CSV Bulk"])
sensitivity = st.sidebar.slider("Sensitivity threshold", 0.0, 100.0, 60.0, 1.0)
show_raw = st.sidebar.checkbox("Show raw responses (debug)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small">Threshold tunes HIGH/ELEVATED classification</div>', unsafe_allow_html=True)

# ----------------------------
# Main layout: left / right
# ----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single prediction")
    st.markdown('<div class="muted">Enter six numeric features. Remaining values are zero-padded to 30.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Feature 1", value=0.0, format="%.6f", key="f1")
        f2 = st.number_input("Feature 2", value=0.0, format="%.6f", key="f2")
        f3 = st.number_input("Feature 3", value=0.0, format="%.6f", key="f3")
    with c2:
        f4 = st.number_input("Feature 4", value=0.0, format="%.6f", key="f4")
        f5 = st.number_input("Feature 5", value=0.0, format="%.6f", key="f5")
        f6 = st.number_input("Feature 6", value=0.0, format="%.6f", key="f6")

    run = st.button("Run prediction")
    if run:
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        st.info("Requesting prediction...")
        out, stcode = post_single_with_fallback(features, model_selected=model)
        if stcode != 200:
            st.error(f"Backend error (status {stcode}).")
            st.write(out)
            st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
            if stcode == 405:
                st.warning("405: Method Not Allowed. Test backend POST /predict and CORS policy.")
                st.code(f"curl -X POST '{API_SINGLE}?model={model}' -H 'Content-Type: application/json' -d '{{\"features\":[0,0,...]}}'", language="bash")
        else:
            st.session_state.last_single = out
            prob = out.get("fraud_probability", None)
            try:
                prob_pct = round(float(prob) * 100.0, 2) if prob is not None else None
            except:
                prob_pct = None
            st.session_state.last_prob = prob_pct

            st.success("Result")
            if prob_pct is None:
                st.markdown('<div class="muted">Discrete prediction only (no probability).</div>', unsafe_allow_html=True)
            else:
                hp = sensitivity
                mp = sensitivity * 0.6
                if prob_pct >= hp:
                    st.markdown(f"<div class='badge-high'>HIGH — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Action: escalate for review.</div>', unsafe_allow_html=True)
                elif prob_pct >= mp:
                    st.markdown(f"<div class='badge-med'>ELEVATED — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Action: require secondary verification.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-low'>LOW — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Action: allow; monitor.</div>', unsafe_allow_html=True)

            if show_raw:
                st.json(out)

    st.markdown('</div>', unsafe_allow_html=True)

    # spacer
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # CSV bulk card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch scoring (CSV)")
    st.markdown('<div class="muted">Upload CSV. If columns match V1..V28, Amount, Time they are auto-aligned; otherwise numeric order is used.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            df = None

        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head())
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"].copy()
            if st.button("Run batch scoring"):
                st.info("Preparing batches...")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                st.write(f"Prepared {n} rows.")
                chunk_size = 4000
                preds, probs = [], []
                pbar = st.progress(0)
                status = st.empty()
                start = time.time()
                error = False
                for i in range(math.ceil(n / chunk_size)):
                    s = i * chunk_size
                    e = min((i + 1) * chunk_size, n)
                    batch = feat_df.iloc[s:e].values.tolist()
                    status.info(f"Processing chunk {i+1}: rows {s}-{e}")
                    out, stcode = post_batch_with_fallback(batch, model_selected=model, timeout=300)
                    if stcode != 200:
                        st.error(f"Batch failed at chunk {i+1}: status {stcode}")
                        st.write(out)
                        st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
                        error = True
                        break
                    preds.extend(out.get("predictions", []))
                    probs.extend(out.get("probabilities", []))
                    pbar.progress(min(1.0, e / n))
                    time.sleep(0.06)
                if not error:
                    out_df = df.reset_index(drop=True).iloc[:n].copy()
                    out_df["prediction"] = preds
                    out_df["fraud_probability"] = probs
                    st.session_state.out_df = out_df
                    st.success("Batch complete")
                    st.dataframe(out_df.head())
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv_bytes, "predictions.csv", "text/csv")
                    st.write(f"Processed {n} rows in {time.time()-start:.2f} s")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live gauge")
    if st.session_state.last_prob is not None:
        st.components.v1.html(render_gauge(st.session_state.last_prob, size=260), height=320)
        st.markdown('<div class="muted">Use sensitivity slider to tune classification thresholds.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">Run a prediction to populate the gauge.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Session log")
    if len(st.session_state.logs) == 0:
        st.markdown('<div class="muted">No logs yet.</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.logs[-12:]:
            st.write(entry)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer credit
st.markdown(
    """
    <div class="credit">
      Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a>
    </div>
    """,
    unsafe_allow_html=True,
)
