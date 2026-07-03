"""day1_ml_pipeline/eda.py — Exploratory Data Analysis on judgments.csv"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))#Project oda vera folders la irukkura modules access panna path add pannrom.

import matplotlib#graph
matplotlib.use("Agg")#Graph screen la display pannaama direct PNG file aa save panna use pannom
import matplotlib.pyplot as plt
import seaborn as sns#Beautiful charts and statistical plots create panna use pannrom.
from utils.preprocessing import load_raw_data, TARGET_COL

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)#Reports folder illa na create pannum.


def run_eda():
    df = load_raw_data()
    print("Shape:", df.shape)#check
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df[TARGET_COL].value_counts(normalize=True))

    if TARGET_COL in df.columns:#cloumn irukkanu check pannuthu
        plt.figure(figsize=(5, 4))#graph size
        sns.countplot(x=TARGET_COL, data=df)#Outcome distribution bar chart create pannrom.
        plt.title("Outcome Distribution")
        plt.savefig(os.path.join(OUT_DIR, "target_distribution.png"), bbox_inches="tight")#save pannrom
        plt.close()

    num_cols = df.select_dtypes(include="number").columns#select numeric columns
    if len(num_cols) > 1:#At least 2 numeric columns irundha heatmap create pannrom.
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")#Correlation matrix calculate pannrom.num of hearing vs outcome section involved vs outcome
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), bbox_inches="tight")
        plt.close()

    df.describe(include="all").to_csv(os.path.join(OUT_DIR, "eda_summary_stats.csv"))#Summary statistics generate in reports eda_summary_states.csv.mean,median,min,max,count
    print(f"\nEDA artifacts saved to {OUT_DIR}")#EDA artifacts saved to reports


if __name__ == "__main__":
    run_eda()
