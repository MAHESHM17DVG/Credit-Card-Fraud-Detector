# app.py
# Ultra-premium Dark Theme — Liquid Glass · NVIDIA/Apple inspired
# - Manual + CSV bulk prediction
# - Robust backend method fallback (405 handling)
# - Animated SVG gauge with glow and smooth transitions
# - Sensitivity slider (threshold) to control risk classification
# - Cohesive dark theme, liquid glass panels, slide transitions
# - Developer credit (professional)
# - All previous feature-ordering / padding logic retained

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
    page_title="Credit Card Fraud — Obsidian Edition",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Ultra-premium dark CSS (liquid glass + neon accent)
# ----------------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#0b0f14;
      --panel: rgba(255,255,255,0.03);
      --glass: rgba(255,255,255,0.04);
      --glass-2: rgba(255,255,255,0.02);
      --muted:#9aa5b1;
      --accent-nv:#76ff03; /* Nvidia-like neon green */
      --accent-apple:#0aa6ff;
      --soft-white: rgba(255,255,255,0.90);
    }

    html, body, .stApp {
      background: radial-gradient(1200px 600px at 10% 10%, rgba(14,30,40,0.35), transparent 6%),
                  radial-gradient(900px 500px at 90% 90%, rgba(8,10,16,0.4), transparent 6%),
                  var(--bg);
      color: var(--soft-white);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Top hero */
    .hero {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 18px;
      padding: 26px;
      box-shadow: 0 10px 40px rgba(2,6,10,0.7), inset 0 1px 0 rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.04);
      backdrop-filter: blur(8px) saturate(120%);
      margin-bottom: 18px;
    }
    .title {
      font-size: 40px;
      font-weight: 900;
      letter-spacing: -0.6px;
      margin: 0;
      display:flex;
      align-items:center;
      gap:12px;
    }
    .subtitle {
      color: var(--muted);
      margin-top:6px;
      font-size:14px;
    }

    /* Liquid cards */
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 14px;
      padding: 16px;
      border: 1px solid rgba(255,255,255,0.04);
      box-shadow: 0 6px 24px rgba(2,6,10,0.6);
      backdrop-filter: blur(6px);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(2,6,10,0.7); }

    /* Neon accent buttons */
    .stButton>button {
      background: linear-gradient(90deg, rgba(118,255,3,0.12), rgba(10,166,255,0.06));
      border: 1px solid rgba(118,255,3,0.12);
      color: var(--soft-white);
      padding: 10px 14px;
      border-radius: 10px;
      font-weight: 800;
      box-shadow: 0 6px 18px rgba(2,6,10,0.6);
    }

    /* Small text/guides */
    .muted { color: var(--muted); font-size:13px; }
    .small { font-size:12px; color:var(--muted); }

    /* Risk badge styles */
    .badge-low { background: rgba(16,185,129,0.06); color:#10b981; padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid rgba(16,185,129,0.08);}
    .badge-med { background: rgba(249,115,22,0.06); color:#f97316; padding:6px 10px; border-radius:999px; font-weight:700; border:1px solid rgba(249,115,22,0.08);}
    .badge-high { background: rgba(239,68,68,0.06); color:#ef4444; padding:6px 10px; border-radius:999px; font-weight:800; border:1px solid rgba(239,68,68,0.12); box-shadow: 0 6px 20px rgba(239,68,68,0.06);}

    /* footer */
    .credit { text-align:center; color:var(--muted); font-size:13px; margin-top:18px; }
    .credit a { color: var(--accent-nv); font-weight:800; text-decoration:none; }

    /* slider (Streamlit native) extra spacing */
    .stSlider { margin-top:6px; margin-bottom:6px; }

    /* Gauge wrapper */
    .gauge-wrap { display:flex; align-items:center; justify-content:center; flex-direction:column; gap:8px; }

    /* subtle glass divider */
    .divider { height:1px; width:100%; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); margin:12px 0; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers (feature ordering)
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
# SVG animated gauge (more fluid, glow)
# ----------------------------
def render_gauge(percent: float, size: int = 220) -> str:
    p = max(0.0, min(100.0, float(percent)))
    r = 86
    circ = 2 * math.pi * r
    filled = (p / 100.0) * circ
    empty = circ - filled
    # color gradient from neon green to cyan to orange to red
    return f"""
    <div class="gauge-wrap" style="width:{size}px">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#76ff03"/>
            <stop offset="50%" stop-color="#00d1ff"/>
            <stop offset="100%" stop-color="#ff8a00"/>
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
            style="transition: stroke-dasharray 900ms cubic-bezier(.22,.9,.31,1), stroke 900ms ease;" filter="url(#glow)"/>
          <circle r="58" fill="rgba(6,8,10,0.6)" stroke="rgba(255,255,255,0.02)" stroke-width="1.2"/>
          <text x="0" y="-10" text-anchor="middle" font-size="28" font-weight="800" fill="#e6f7ff">{p:.2f}%</text>
          <text x="0" y="18" text-anchor="middle" font-size="11" fill="#9aa5b1">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """


# ----------------------------
# Backend endpoints & helpers (robust with method fallback)
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
    # Preferred: query param
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, headers=headers, timeout=12)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        try:
            return r.json(), 200
        except: return {"detail": r.text}, r.status_code
    if r.status_code == 405:
        # fallback to POST without query param
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
# Top hero
# ----------------------------
st.markdown('<div class="hero"><div style="display:flex; justify-content:space-between; align-items:center;">', unsafe_allow_html=True)
st.markdown('<div><div class="title">Credit Card Fraud Detection<span style="color:var(--accent-nv);font-size:18px;margin-left:6px">OBSIDIAN</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Minimal · professional · high-fidelity. Clear probability, actionable guidance.</div></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar controls (includes sensitivity slider)
# ----------------------------
st.sidebar.markdown("### Controls")
model = st.sidebar.radio("Model", models_list)
mode = st.sidebar.selectbox("Mode", ["Manual (6 features)", "CSV Bulk"])
sensitivity = st.sidebar.slider("Sensitivity threshold (affects HIGH/ELEVATED cutoff)", 0.0, 100.0, 60.0, 1.0)
show_raw = st.sidebar.checkbox("Show raw backend responses (debug)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small">Status: Professional UI • Liquid glass • NVIDIA green accents</div>', unsafe_allow_html=True)

# ----------------------------
# Main layout: two columns
# ----------------------------
left_col, right_col = st.columns([2, 1])

# Left column: manual + csv
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single prediction (manual)")
    st.markdown('<div class="muted">Enter six numeric features representative of your input vector. Remaining features auto-padded to 30.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Feature 1", value=0.0, format="%.6f", key="f1")
        f2 = st.number_input("Feature 2", value=0.0, format="%.6f", key="f2")
        f3 = st.number_input("Feature 3", value=0.0, format="%.6f", key="f3")
    with c2:
        f4 = st.number_input("Feature 4", value=0.0, format="%.6f", key="f4")
        f5 = st.number_input("Feature 5", value=0.0, format="%.6f", key="f5")
        f6 = st.number_input("Feature 6", value=0.0, format="%.6f", key="f6")

    run = st.button("Run single prediction")
    if run:
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        st.info("Requesting model prediction...")
        out, stcode = post_single_with_fallback(features, model_selected=model)
        if stcode != 200:
            st.error(f"Backend error (status {stcode}). See details below.")
            st.write(out)
            st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
            if stcode == 405:
                st.warning("405: Method Not Allowed. Try ensuring your backend accepts POST, and check CORS. Example curl:")
                st.code(f"curl -X POST '{API_SINGLE}?model={model}' -H 'Content-Type: application/json' -d '{{\"features\":[0,0,...]}}'", language="bash")
        else:
            st.success("Prediction received")
            st.session_state.last_single = out
            prob = out.get("fraud_probability", None)
            try:
                prob_pct = round(float(prob) * 100.0, 2) if prob is not None else None
            except:
                prob_pct = None
            st.session_state.last_prob = prob_pct

            # risk classification using sensitivity slider
            if prob_pct is None:
                st.markdown('<div class="muted">Model returned prediction but no probability available.</div>', unsafe_allow_html=True)
            else:
                hp = sensitivity  # e.g., 60 default
                mp = sensitivity * 0.6
                if prob_pct >= hp:
                    st.markdown(f"<div class='badge-high'>HIGH RISK — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Recommendation: block transaction and escalate for human review.</div>', unsafe_allow_html=True)
                elif prob_pct >= mp:
                    st.markdown(f"<div class='badge-med'>ELEVATED RISK — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Recommendation: require secondary verification (OTP / 2FA).</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-low'>LOW RISK — {prob_pct}%</div>", unsafe_allow_html=True)
                    st.markdown('<div class="muted">Recommendation: allow and monitor.</div>', unsafe_allow_html=True)

            if show_raw:
                st.json(out)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # CSV Bulk block
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Bulk predictions (CSV)")
    st.markdown('<div class="muted">Upload a CSV. If it contains V1..V28, Amount, Time they will be aligned automatically. Otherwise numeric columns are used and padded.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            df = None

        if df is not None:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"].copy()
            if st.button("Run bulk prediction on CSV"):
                st.info("Preparing and sending batches to backend...")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                st.write(f"Prepared {n} rows with {feat_df.shape[1]} features.")
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
                    status.info(f"Processing chunk {i+1} rows {s}-{e}")
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
                    st.success("Bulk prediction complete")
                    st.dataframe(out_df.head())
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")
                    st.write(f"Processed {n} rows in {time.time() - start:.2f} seconds")
    st.markdown('</div>', unsafe_allow_html=True)

# Right column: gauge and logs
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live gauge")
    if st.session_state.last_prob is not None:
        st.components.v1.html(render_gauge(st.session_state.last_prob, size=260), height=320)
        st.markdown('<div class="muted">Adjust sensitivity slider in the sidebar to tune HIGH/ELEVATED thresholds.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">Run a single prediction to populate the gauge.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Session logs")
    if len(st.session_state.logs) == 0:
        st.markdown('<div class="muted">No logs yet — backend responses and errors will appear here.</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.logs[-12:]:
            st.write(entry)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer / credit
st.markdown(
    f"""
    <div class="credit">
      Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a>
      · Obsidian Edition · Professional AI Fraud Detection
    </div>
    """, unsafe_allow_html=True
)

# small operator notes
st.markdown('<div class="muted" style="margin-top:12px;font-size:12px;">Notes: The frontend sends 30 numeric features. For best accuracy ensure backend model uses the same ordering used during training. If you get 405 errors, check backend method & CORS; the UI attempts fallback POST forms automatically and provides curl guidance.</div>', unsafe_allow_html=True)

# End file
