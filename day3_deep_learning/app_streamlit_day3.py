"""
day3_deep_learning/app_streamlit_day3.py
Streamlit front-end that calls the FastAPI deployment (Day3 Streamlit Deployment deliverable).
Run FastAPI first: uvicorn day3_deep_learning.api_fastapi:app --reload --port 8000
Then:             streamlit run day3_deep_learning/app_streamlit_day3.py
"""
import os
import joblib
import requests
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
API_URL = os.environ.get("LEGAL_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Legal Case Predictor — Deployed", layout="centered")
st.title("⚖️ Legal Case Outcome Predictor — Day 3 (CNN via FastAPI)")

try:
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
except FileNotFoundError:
    st.error("feature_columns.pkl not found. Run Day1/Day2 pipeline first.")
    st.stop()

model_choice = st.radio("Select model", ["cnn", "xgboost"], horizontal=True)

st.subheader("Enter Case Metadata")
inputs = {}
cols = st.columns(2)
for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        inputs[col] = st.number_input(col, value=0.0, step=1.0)

if st.button("Predict via API"):
    try:
        resp = requests.post(f"{API_URL}/predict/{model_choice}", json={"features": inputs}, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        st.success(f"Predicted Outcome: **{result['predicted_outcome']}**")
        st.metric("Probability of favorable outcome", f"{result['probability_favorable']:.2%}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Start it with: uvicorn day3_deep_learning.api_fastapi:app --reload")
    except Exception as e:
        st.error(f"Error: {e}")
