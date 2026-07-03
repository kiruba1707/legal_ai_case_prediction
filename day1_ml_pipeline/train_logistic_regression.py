"""day1_ml_pipeline/train_logistic_regression.py"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.preprocessing import full_pipeline_load, scale_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    X, y, _ = full_pipeline_load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train)

    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("Logistic Regression model trained and saved.")
    return model, scaler, X_test_s, y_test


if __name__ == "__main__":
    train()
