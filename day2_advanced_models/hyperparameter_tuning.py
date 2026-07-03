"""
day2_advanced_models/hyperparameter_tuning.py
RandomizedSearchCV tuning of XGBoost -> produces the Day2 deliverable:
models/tuned_xgboost.pkl
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from utils.preprocessing import full_pipeline_load
from day1_ml_pipeline.evaluate import evaluate_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def tune():
    X, y, _ = full_pipeline_load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base_model = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base_model, param_distributions=PARAM_DIST,
        n_iter=25, scoring="f1", cv=cv, random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best Params:", search.best_params_)
    print("Best CV F1:", search.best_score_)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "tuned_xgboost.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

    evaluate_model(model=best_model, X_test=X_test, y_test=y_test, model_name="tuned_xgboost")
    print("Tuned XGBoost saved to models/tuned_xgboost.pkl (Day2 deliverable).")
    return best_model


if __name__ == "__main__":
    tune()
