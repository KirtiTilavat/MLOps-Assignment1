import math

from sklearn.linear_model import LogisticRegression

from src.features import FEATURES
from src.train import MODEL_SPECS, build_pipeline, evaluate, split_data, tune


def test_split_is_stratified_and_reproducible(clean_df):
    X_tr, X_te, y_tr, y_te = split_data(clean_df)
    assert len(X_te) == math.ceil(len(clean_df) * 0.2)  # sklearn rounds test size up
    assert abs(y_tr.mean() - y_te.mean()) < 0.1
    X_tr2, *_ = split_data(clean_df)
    assert list(X_tr.index) == list(X_tr2.index)


def test_pipeline_predicts_valid_probabilities(fitted_model, clean_df):
    proba = fitted_model.predict_proba(clean_df[FEATURES])
    assert proba.shape == (len(clean_df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_all_metrics_in_range(fitted_model, clean_df):
    metrics = evaluate(fitted_model, clean_df[FEATURES], clean_df["target"])
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_model_learns_signal(clean_df):
    X_tr, X_te, y_tr, y_te = split_data(clean_df)
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    assert evaluate(model, X_te, y_te)["recall"] >= 0.7


def test_tune_runs_with_small_grid(clean_df):
    X_tr, _, y_tr, _ = split_data(clean_df)
    estimator, _ = MODEL_SPECS["logistic_regression"]
    search = tune("lr", estimator, {"clf__C": [0.1, 1.0]}, X_tr, y_tr)
    assert search.best_params_["clf__C"] in (0.1, 1.0)
    assert hasattr(search.best_estimator_, "predict_proba")
