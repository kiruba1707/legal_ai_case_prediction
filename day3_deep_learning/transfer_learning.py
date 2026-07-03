"""
day3_deep_learning/transfer_learning.py
Transfer learning on case_description text using a pretrained sentence-transformer
(all-MiniLM-L6-v2) for embeddings + a fine-tuned classifier head.
Falls back to TF-IDF if sentence-transformers/torch download is unavailable (offline env).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_raw_data, TARGET_COL

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(texts, show_progress_bar=True)
        joblib.dump("sentence-transformer", os.path.join(MODEL_DIR, "transfer_embedder_type.pkl"))
        return embeddings
    except Exception as e:
        print(f"[WARN] sentence-transformers unavailable ({e}). Falling back to TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=256)
        embeddings = vectorizer.fit_transform(texts).toarray()
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump("tfidf", os.path.join(MODEL_DIR, "transfer_embedder_type.pkl"))
        return embeddings


def train():
    df = load_raw_data()
    if "case_description" not in df.columns:
        raise ValueError("Dataset requires 'case_description' text column for transfer learning.")

    texts = df["case_description"].astype(str).tolist()
    y = df[TARGET_COL].values

    X_emb = get_embeddings(texts)
    X_train, X_test, y_train, y_test = train_test_split(X_emb, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Transfer-learning classifier test accuracy: {acc:.4f}")

    joblib.dump(clf, os.path.join(MODEL_DIR, "transfer_learning_head.pkl"))
    print("Transfer learning model saved to models/transfer_learning_head.pkl")
    return clf


if __name__ == "__main__":
    train()
