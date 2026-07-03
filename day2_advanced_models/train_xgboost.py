"""day2_advanced_models/train_xgboost.py"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from utils.preprocessing import full_pipeline_load

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    X, y, _ = full_pipeline_load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    print("XGBoost model trained and saved.")
    return model, X_test, y_test


if __name__ == "__main__":
    train()
