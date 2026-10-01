"""Load and clean the raw UCI Heart Disease (Cleveland) data.

Cleaning steps:
  1. Assign the 14 documented column names (raw file has no header).
  2. Treat '?' as missing; drop the few rows with missing `ca` / `thal`.
  3. Convert the multi-class target `num` (0-4) to binary `target`
     (0 = no disease, 1 = disease present).
  4. Cast columns to proper numeric / integer types.

Usage:
    python -m src.data_processing
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_FILE = ROOT / "data" / "processed" / "heart_clean.csv"

COLUMNS = [
    "age",  # age in years
    "sex",  # 1 = male, 0 = female
    "cp",  # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",  # serum cholesterol (mg/dl)
    "fbs",  # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",  # resting ECG result (0-2)
    "thalach",  # maximum heart rate achieved
    "exang",  # exercise-induced angina (1 = yes)
    "oldpeak",  # ST depression induced by exercise
    "slope",  # slope of peak exercise ST segment (1-3)
    "ca",  # number of major vessels coloured by fluoroscopy (0-3)
    "thal",  # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",  # diagnosis (0 = no disease, 1-4 = disease severity)
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the raw headerless CSV, treating '?' as missing."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe with a binary target."""
    df = df.copy()
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype(float)
    return df


def load_clean(path: Path = CLEAN_FILE) -> pd.DataFrame:
    """Load the processed dataset produced by `main()`."""
    return pd.read_csv(path)


def main() -> None:
    raw = load_raw()
    missing = int(raw.isna().any(axis=1).sum())
    df = clean(raw)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Raw rows: {len(raw)} | rows with missing values dropped: {missing}")
    print(f"Clean rows: {len(df)} | saved to {CLEAN_FILE}")
    print(f"Class balance:\n{df[TARGET].value_counts().to_string()}")


if __name__ == "__main__":
    main()
