from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
import requests

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraud using Logistic Regression & Random Forest",
    version="1.0.0"
)

# Allow frontend (Streamlit / Render) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# URLs will be added later after models uploaded
AVAILABLE_MODELS = {
    "logreg": (
        MODEL_DIR / "logreg.pkl",
        "https://github.com/SRIHARSHA-BHARADWAJ/Credit-Card-Fraud-Detection-ML-WebApp/releases/download/v1.0.0/logreg.pkl"
    ),
    "rf": (
        MODEL_DIR / "rf.pkl",
        "https://github.com/SRIHARSHA-BHARADWAJ/Credit-Card-Fraud-Detection-ML-WebApp/releases/download/v1.0.0/rf.pkl"
    ),
}


MODEL_CACHE = {}

class FeatureInput(BaseModel):
    features: List[float]


def download_file(url, dest_path):
    try:
        print(f"Downloading: {url}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("Downloaded:", dest_path)
    except Exception as e:
        raise Exception(f"Download failed: {e}")


def load_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Model not found")

    local_path, url = AVAILABLE_MODELS[model_name]

    if not local_path.exists():
        if url:
            download_file(url, local_path)
        else:
            raise HTTPException(status_code=500, detail="Model file missing")

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model = joblib.load(local_path)
    MODEL_CACHE[model_name] = model
    return model


@app.get("/")
def home():
    return {"message": "Fraud Detection API running!"}


@app.get("/get-models")
def get_models():
    return {"available_models": list(AVAILABLE_MODELS.keys())}


@app.post("/predict")
def predict(input_data: FeatureInput, model: str = "logreg"):
    try:
        model_obj = load_model(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    x = np.array(input_data.features).reshape(1, -1)
    pred = int(model_obj.predict(x)[0])

    try:
        prob = float(model_obj.predict_proba(x)[0][1])
    except:
        prob = "N/A"

    return {
        "model_used": model,
        "prediction": pred,
        "fraud_probability": prob
    }
