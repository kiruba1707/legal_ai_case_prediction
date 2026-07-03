"""
utils/preprocessing.py
Shared data loading + feature engineering for judgments.csv
Expected/assumed columns (adjust COLUMN_MAP if your schema differs):
    case_id, court, judge_name, case_type, filing_date, decision_date,
    sections_involved, num_hearings, petitioner_type, respondent_type,
    case_description, outcome   (outcome: 0=Dismissed, 1=Allowed)
If judgments.csv is missing, a synthetic dataset with this schema is
generated so the pipeline is runnable end-to-end without real data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "judgments.csv")
TARGET_COL = "outcome"


def _generate_synthetic(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    courts = ["Supreme Court", "High Court", "District Court", "Tribunal"]
    case_types = ["Civil", "Criminal", "Constitutional", "Corporate", "Family"]
    parties = ["Individual", "Company", "Government", "NGO"]
    filing_dates = pd.date_range("2015-01-01", "2023-12-31", periods=n)

    df = pd.DataFrame({
        "case_id": [f"C{i:05d}" for i in range(n)],
        "court": rng.choice(courts, n),
        "judge_name": rng.choice([f"Judge_{i}" for i in range(30)], n),
        "case_type": rng.choice(case_types, n),
        "filing_date": filing_dates,
        "decision_date": filing_dates + pd.to_timedelta(rng.integers(30, 900, n), unit="D"),
        "sections_involved": rng.integers(1, 10, n),
        "num_hearings": rng.integers(1, 40, n),
        "petitioner_type": rng.choice(parties, n),
        "respondent_type": rng.choice(parties, n),
        "case_description": ["Case involving dispute over contractual and statutory obligations."] * n,
    })
    logits = (
        0.05 * df["num_hearings"]
        - 0.3 * df["sections_involved"]
        + rng.normal(0, 2, n)#-2.1 -1.5 -0.8 0.2 0.9 1.7 2.5 2.5 median=0.2
    )
    df[TARGET_COL] = (logits > np.median(logits)).astype(int)
    return df


def load_raw_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"[WARN] {DATA_PATH} not found. Using synthetic judgments dataset.")
        df = _generate_synthetic()
    return df


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    for col in ["filing_date", "decision_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")#string date to datetime format

    if "filing_date" in df.columns and "decision_date" in df.columns:
        df["case_duration_days"] = (df["decision_date"] - df["filing_date"]).dt.days
        df["filing_year"] = df["filing_date"].dt.year
        df["filing_month"] = df["filing_date"].dt.month

    if "case_description" in df.columns:
        df["description_length"] = df["case_description"].astype(str).apply(len)
        df["word_count"] = df["case_description"].astype(str).apply(lambda x: len(x.split()))

    df = df.fillna({
        "sections_involved": 0, "num_hearings": 0,
        "case_duration_days": df.get("case_duration_days", pd.Series([0])).median(),
    })#fill missing value with default values

    drop_cols = [c for c in ["case_id", "filing_date", "decision_date", "case_description"] if c in df.columns]
    df = df.drop(columns=drop_cols)#remove unwanted columns

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()#text to computer understanding numeric values
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def get_feature_target_split(df: pd.DataFrame):#separate feature input :X court,hearing,section and target_ y outcome
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def scale_features(X_train, X_test):#convert difffernt range values into same scale no of hearing=30 after 0.65
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def full_pipeline_load():
    """One-call loader used by all Day1-3 scripts."""
    raw = load_raw_data()
    feat_df, encoders = engineer_features(raw)
    X, y = get_feature_target_split(feat_df)
    return X, y, encoders
