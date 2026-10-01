"""Load the saved model pipeline and score raw patient records.

The saved pipeline already contains the preprocessing step, so callers pass
raw feature values exactly as they appear in the cleaned dataset.

Usage:
    python -m src.predict            # scores a built-in sample patient
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

# Probability-of-disease bands used to give a human-readable risk level.
RISK_BANDS = [(0.30, "low"), (0.65, "moderate"), (1.01, "high")]

SAMPLE_PATIENT = {
    "age": 54,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 2,
    "thalach": 140,
    "exang": 1,
    "oldpeak": 1.6,
    "slope": 2,
    "ca": 1,
    "thal": 7,
}


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def risk_level(probability: float) -> str:
    return next(label for upper, label in RISK_BANDS if probability < upper)


def predict(records: dict | list[dict], model=None) -> list[dict]:
    """Score one or more patient records.

    Returns, per record: predicted class (0/1), confidence of that class,
    probability of disease and a risk level (low / moderate / high).
    """
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    model = model or load_model()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        results.append(
            {
                "prediction": pred,
                "label": "disease" if pred else "no disease",
                "confidence": round(float(p if pred else 1 - p), 4),
                "probability_disease": round(float(p), 4),
                "risk_level": risk_level(float(p)),
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(predict(SAMPLE_PATIENT)[0], indent=2))
