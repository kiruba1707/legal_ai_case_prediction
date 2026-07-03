"""Run once to create a sample data/judgments.csv (replace with real data later)."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import _generate_synthetic

if __name__ == "__main__":
    df = _generate_synthetic(n=2000)
    out_path = os.path.join(os.path.dirname(__file__), "judgments.csv")
    df.to_csv(out_path, index=False)
    print(f"Sample judgments.csv written to {out_path} — shape {df.shape}")
