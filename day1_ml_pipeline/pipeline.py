"""day1_ml_pipeline/pipeline.py — Complete Day1 ML Pipeline: EDA -> FE -> Train -> Evaluate"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from day1_ml_pipeline.eda import run_eda
from day1_ml_pipeline.feature_engineering import run_feature_engineering
from day1_ml_pipeline.train_logistic_regression import train
from day1_ml_pipeline.evaluate import evaluate_model


def run_day1_pipeline():
    print("STEP 1/4: EDA")
    run_eda()

    print("\nSTEP 2/4: Feature Engineering")
    run_feature_engineering()

    print("\nSTEP 3/4: Train Logistic Regression")
    model, scaler, X_test_s, y_test = train()

    print("\nSTEP 4/4: Evaluation")
    evaluate_model(model=model, X_test=X_test_s, y_test=y_test, model_name="logistic_regression")

    print("\nDay 1 pipeline complete. Model + reports saved in /models and /reports.")


if __name__ == "__main__":
    run_day1_pipeline()
