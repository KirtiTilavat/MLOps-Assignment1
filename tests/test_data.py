import pandas as pd

from src.data_processing import (
    CATEGORICAL_FEATURES,
    COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    load_raw,
)


def test_load_raw_parses_question_marks_as_missing(tmp_path):
    raw = tmp_path / "raw.data"
    raw.write_text(
        "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
        "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,3.0,2\n"
    )
    df = load_raw(raw)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    assert df["ca"].isna().sum() == 1


def test_clean_drops_rows_with_missing_values(raw_df):
    n_missing = raw_df.isna().any(axis=1).sum()
    out = clean(raw_df)
    assert n_missing == 3
    assert out.isna().sum().sum() == 0
    assert len(out) == len(raw_df) - n_missing


def test_clean_creates_binary_target_and_drops_num(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    expected = (raw_df.dropna()["num"] > 0).astype(int).to_numpy()
    assert (out[TARGET].to_numpy() == expected).all()


def test_clean_sets_column_types(raw_df):
    out = clean(raw_df)
    for col in CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(out[col]), col
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_float_dtype(out[col]), col


def test_clean_removes_duplicates_and_does_not_mutate_input(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    before = doubled.copy()
    out = clean(doubled)
    assert len(out) == len(clean(raw_df))
    pd.testing.assert_frame_equal(doubled, before)
