from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

# -------------------------------
# APP CONFIG
# -------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraud using Logistic Regression & Random Forest",
    version="1.0.0"
)

# Enable GZip compression → 5x speedup for large requests
app.add_middleware(GZipMiddleware, minimum_size=1000)

# -------------------------------
# MODEL PATHS
# -------------------------------
MODEL_DIR = Path("models")

MODEL_URLS = {
    "logreg": "https://github.com/SRIHARSHA-BHARADWAJ/Credit-Card-Fraud-Detection-ML-WebApp/releases/download/v1.0.0/logreg.pkl",
    "rf": "https://github.com/SRIHARSHA-BHARADWAJ/Credit-Card-Fraud-Detection-ML-WebApp/releases/download/v1.0.0/rf.pkl",
}

MODEL_CACHE = {}

# -------------------------------
# Download model if missing
# -------------------------------
import requests

def download_model(model_name: str):
    dest = MODEL_DIR / f"{model_name}.pkl"
    url = MODEL_URLS.get(model_name)

    if not url:
        raise HTTPException(status_code=400, detail=f"No download URL for {model_name}")

    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Downloading model: {model_name} ...")
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {url}")

    with open(dest, "wb") as f:
        f.write(r.content)

    print(f"Saved model: {dest}")
    return dest

# -------------------------------
# Load Model
# -------------------------------
def load_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = MODEL_DIR / f"{model_name}.pkl"

    if not model_path.exists():
        model_path = download_model(model_name)

    try:
        model = joblib.load(model_path)
    except:
        raise HTTPException(status_code=500, detail="Model corrupted or unreadable.")

    MODEL_CACHE[model_name] = model
    return model

# -------------------------------
# INPUT SCHEMAS
# -------------------------------
class FeatureInput(BaseModel):
    features: List[float]

class BatchFeatures(BaseModel):
    features: List[List[float]]

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API running!"}

# -------------------------------
# LIST MODELS
# -------------------------------
@app.get("/get-models")
def get_models():
    return {"available_models": list(MODEL_URLS.keys())}

# -------------------------------
# SINGLE PREDICT
# -------------------------------
@app.post("/predict")
def predict(input_data: FeatureInput, model: str = "logreg"):

    model_obj = load_model(model)
    x = np.array(input_data.features).reshape(1, -1)

    pred = int(model_obj.predict(x)[0])

    try:
        prob = float(model_obj.predict_proba(x)[0][1])
    except:
        prob = None

    return {
        "model_used": model,
        "prediction": pred,
        "fraud_probability": prob
    }

# -------------------------------
# BATCH PREDICT (FAST)
# -------------------------------
@app.post("/predict-batch")
def predict_batch(input_data: dict, model: str = "rf"):
    try:
        # 1. Load model
        model_obj = load_model(model)

        # 2. Extract features list
        rows = input_data.get("features")
        if rows is None:
            raise HTTPException(status_code=400, detail="Missing 'features' key.")

        X = np.array(rows)

        # 3. Predictions
        preds = model_obj.predict(X).tolist()

        # 4. Probabilities
        try:
            probs = model_obj.predict_proba(X)[:, 1].tolist()
        except:
            probs = [None] * len(preds)

        return {
            "predictions": preds,
            "probabilities": probs
        }

    except Exception as e:
        return {"detail": f"Batch prediction failed: {str(e)}"}
