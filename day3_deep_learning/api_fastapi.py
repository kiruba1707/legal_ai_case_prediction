"""
day3_deep_learning/api_fastapi.py
FastAPI deployment serving the CNN model (with tuned XGBoost as fallback/ensemble option).
Run: uvicorn day3_deep_learning.api_fastapi:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from day3_deep_learning.train_cnn import CNN1D

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="Legal Case Outcome Prediction API", version="1.0")

_artifacts = {}


@app.on_event("startup")
def load_artifacts():
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "cnn_scaler.pkl"))
    input_dim = joblib.load(os.path.join(MODEL_DIR, "cnn_input_dim.pkl"))

    model = CNN1D(input_dim)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cnn_model.pt"), map_location="cpu"))
    model.eval()

    xgb_model = None
    xgb_path = os.path.join(MODEL_DIR, "tuned_xgboost.pkl")
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)

    _artifacts.update(dict(model=model, scaler=scaler, feature_cols=feature_cols, xgb_model=xgb_model))


class CaseFeatures(BaseModel):
    features: Dict[str, float]


@app.get("/")
def health():
    return {"status": "ok", "message": "Legal Case Outcome Prediction API running."}


@app.post("/predict/cnn")
def predict_cnn(payload: CaseFeatures):
    feature_cols = _artifacts["feature_cols"]
    try:
        row = np.array([[payload.features.get(c, 0.0) for c in feature_cols]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    row_scaled = _artifacts["scaler"].transform(row)
    x_tensor = torch.tensor(row_scaled, dtype=torch.float32)
    with torch.no_grad():
        proba = _artifacts["model"](x_tensor).item()

    return {
        "model": "cnn",
        "predicted_outcome": "Allowed" if proba > 0.5 else "Dismissed",
        "probability_favorable": round(proba, 4),
    }


@app.post("/predict/xgboost")
def predict_xgboost(payload: CaseFeatures):
    if _artifacts["xgb_model"] is None:
        raise HTTPException(status_code=404, detail="Tuned XGBoost model not found. Run Day2 tuning first.")
    feature_cols = _artifacts["feature_cols"]
    row = [[payload.features.get(c, 0.0) for c in feature_cols]]
    proba = _artifacts["xgb_model"].predict_proba(row)[0][1]
    return {
        "model": "tuned_xgboost",
        "predicted_outcome": "Allowed" if proba > 0.5 else "Dismissed",
        "probability_favorable": round(float(proba), 4),
    }
