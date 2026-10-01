import numpy as np

from src.data_processing import NUMERIC_FEATURES
from src.features import (
    BINARY_FEATURES,
    FEATURES,
    NOMINAL_FEATURES,
    build_preprocessor,
)


def test_feature_groups_cover_all_13_inputs_once():
    assert len(FEATURES) == 13
    assert len(set(FEATURES)) == 13
    assert set(NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES) == set(FEATURES)


def test_output_shape_matches_one_hot_expansion(clean_df):
    X = clean_df[FEATURES]
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)
    n_onehot = sum(X[c].nunique() for c in NOMINAL_FEATURES)
    assert out.shape == (len(X), len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + n_onehot)
    assert not np.isnan(out).any()


def test_numeric_columns_are_standardised(clean_df):
    X = clean_df[FEATURES]
    out = build_preprocessor().fit_transform(X)[:, : len(NUMERIC_FEATURES)]
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-8)


def test_unseen_category_does_not_crash(clean_df):
    X = clean_df[FEATURES]
    pre = build_preprocessor().fit(X)
    new = X.iloc[[0]].copy()
    new["thal"] = 99  # category never seen in training
    out = pre.transform(new)
    assert out.shape[1] == pre.transform(X.iloc[[0]]).shape[1]
