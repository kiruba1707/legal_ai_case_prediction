"""
day2_advanced_models/app_streamlit_day2.py
Streamlit App — Case Outcome Prediction using tuned XGBoost
Run: streamlit run day2_advanced_models/app_streamlit_day2.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Legal Case Outcome Predictor", layout="centered")
st.title("⚖️ Legal Case Outcome Predictor (Day 2 — Tuned XGBoost)")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "tuned_xgboost.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, feature_cols


try:
    model, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error("Tuned model not found. Run day2_advanced_models/hyperparameter_tuning.py first.")
    st.stop()

st.subheader("Enter Case Metadata")
inputs = {}
cols = st.columns(2)
for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        inputs[col] = st.number_input(col, value=0.0, step=1.0)

if st.button("Predict Outcome"):
    X_input = pd.DataFrame([inputs])[feature_cols]
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    label = "Allowed / Favorable" if pred == 1 else "Dismissed / Unfavorable"
    st.success(f"Predicted Outcome: **{label}**")
    st.metric("Confidence (probability of favorable outcome)", f"{proba:.2%}")
