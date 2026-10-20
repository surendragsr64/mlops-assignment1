import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RAW_FILE = os.path.join(RAW_DIR, "heart_disease_raw.csv")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "heart_disease_clean.csv")

def download_dataset():
    """Fetch Heart Disease dataset from UCI repository and save raw copy."""
    os.makedirs(RAW_DIR, exist_ok=True)
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target into one DataFrame
    df = pd.concat([X, y], axis=1)
    df.to_csv(RAW_FILE, index=False)
    print(f"Raw dataset saved to {RAW_FILE}")

def preprocess_dataset():
    """Clean and preprocess dataset, then save to processed folder."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = pd.read_csv(RAW_FILE)

    # Drop missing values and duplicates
    df = df.dropna().drop_duplicates()

    # Example encoding: ensure categorical columns are properly formatted
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({0: "female", 1: "male"})

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_FILE}")

if __name__ == "__main__":
    download_dataset()
    preprocess_dataset()
