"""Shared fixtures: small synthetic datasets so tests run fast and offline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_processing import COLUMNS, clean
from src.train import build_pipeline


def make_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Raw-format rows (14 columns incl. multi-class `num`) with a learnable signal."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)  # hidden risk drives both features and diagnosis
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 77, n).astype(float),
            "sex": rng.integers(0, 2, n).astype(float),
            "cp": np.where(risk > 0.5, 4, rng.integers(1, 4, n)).astype(float),
            "trestbps": rng.integers(95, 200, n).astype(float),
            "chol": rng.integers(130, 400, n).astype(float),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
            "thalach": (200 - risk * 90 + rng.normal(0, 8, n)).round(),
            "exang": (risk > 0.6).astype(float),
            "oldpeak": (risk * 4 + rng.normal(0, 0.3, n)).clip(0).round(1),
            "slope": rng.integers(1, 4, n).astype(float),
            "ca": np.where(risk > 0.5, rng.integers(1, 4, n), 0).astype(float),
            "thal": np.where(risk > 0.5, 7, 3).astype(float),
            "num": np.where(risk > 0.5, rng.integers(1, 5, n), 0),
        }
    )
    return df[COLUMNS]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    df = make_raw()
    df.loc[[3, 10], "ca"] = np.nan
    df.loc[5, "thal"] = np.nan
    return df


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture
def fitted_model(clean_df):
    from src.features import FEATURES

    model = build_pipeline(LogisticRegression(max_iter=1000))
    return model.fit(clean_df[FEATURES], clean_df["target"])


@pytest.fixture
def patient() -> dict:
    return {
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
