"""day1_ml_pipeline/evaluate.py — evaluation metrics + report for any trained sklearn-style model"""
import sys, os, json#project path handle panna,file/folder operations,report JSON file save panna
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))#Project oda vera folders la irukkura files import panna.

import joblib#Saved model load panna.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt#Confusion matrix and ROC curve images save panna.

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.model_selection import train_test_split#Train and test split create panna
from utils.preprocessing import full_pipeline_load, scale_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")#folder location store pannudhu
os.makedirs(REPORT_DIR, exist_ok=True)#Reports folder illa na create pannum.

def evaluate_model(model_path=None, model=None, X_test=None, y_test=None, model_name="logistic_regression"):
    if model is None:
        model = joblib.load(model_path or os.path.join(MODEL_DIR, f"{model_name}.pkl"))#Model already memory la irukka?illana logistic_regression.pkl

    if X_test is None or y_test is None:#Test data ready ah irukka? illana X, y, _ = full_pipeline_load() load pannum
        X, y, _ = full_pipeline_load()#illana X, y, _ = full_pipeline_load() load pannum
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)#20000 case la 16000 training 4000 testing
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))#Training time la use panna same scaler load pannum
        X_test = scaler.transform(X_test)#Test data scale pannum.

    y_pred = model.predict(X_test)#Model prediction pannum
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred#Probability calculate pannum.Case1 → 90% chance outcome=1 Case2 → 20% chance outcome=1

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),#2500 crt 4000 total 25000/4000=62.5%
        "precision": precision_score(y_test, y_pred, zero_division=0),#TP / (TP + FP) Positive nu sonna cases la correct evlo.
        "recall": recall_score(y_test, y_pred, zero_division=0),#TP / (TP + FN) Actual positive cases la identify pannadhu evlo.
        "f1_score": f1_score(y_test, y_pred, zero_division=0),#Precision and recall oda balance score.
        "roc_auc": roc_auc_score(y_test, y_proba),#Model evlo nalla separate pannudhu
    }

    cm = confusion_matrix(y_test, y_pred)#create TP TN FP FN graph 
    plt.figure(figsize=(4, 4))#
    plt.imshow(cm, cmap="Blues")#Blue color confusion matrix create pannum
    plt.title(f"Confusion Matrix - {model_name}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")#Numbers graph la display pannum.
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(os.path.join(REPORT_DIR, f"confusion_matrix_{model_name}.png"), bbox_inches="tight")#reports/confusion_matrix_logistic_regression.png
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(os.path.join(REPORT_DIR, f"roc_curve_{model_name}.png"), bbox_inches="tight")
    plt.close()#ROC graph generate pannum reports/roc_curve_logistic_regression.png

    report_path = os.path.join(REPORT_DIR, f"evaluation_{model_name}.json")#evaluation_logistic_regression.json
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)#Metrics save pannum.accuracy":0.6255,precision":0.6292,recall":0.6110

    print(f"\n=== {model_name} Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\n", classification_report(y_test, y_pred, zero_division=0))
    print(f"Report saved to {report_path}")#print output accuracy":0.6255,precision":0.6292,recall":0.6110

    print(f"\n=== {model_name} Evaluation ===")
    return metrics


if __name__ == "__main__":
    evaluate_model(model_name="logistic_regression")
