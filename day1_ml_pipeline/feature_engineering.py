"""day1_ml_pipeline/feature_engineering.py — builds and saves engineered feature set"""
import sys, os#python path manage panna
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))#utils/preprocessing.py  import panna use aagudhu.

import joblib#Python objects save panna use pannuvom. label_encoders.pkl save panna use pannrom
from utils.preprocessing import load_raw_data, engineer_features

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")#Output files save panna location set pannrom legal_ai_pipeline/data
os.makedirs(OUT_DIR, exist_ok=True)#data folder illa na create pannum


def run_feature_engineering():
    raw = load_raw_data()
    feat_df, encoders = engineer_features(raw)#Feature Engineering execute pannrom new columns ah add apnnum .text to no
    feat_df.to_csv(os.path.join(OUT_DIR, "judgments_features.csv"), index=False)#Feature engineered dataset save pannrom. data/judgments_features.csv
    joblib.dump(encoders, os.path.join(OUT_DIR, "label_encoders.pkl"))#text to no save agum data/label_encoders.pkl
    print("Engineered features saved:", feat_df.shape)#(20000,13)
    print("Columns:", list(feat_df.columns))#(court,judge_name) Final machine learning features display pannrom.
    return feat_df, encoders#Processed dataset and encoders (high court ->0) return pannrom.


if __name__ == "__main__":
    run_feature_engineering()
