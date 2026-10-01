"""Train, tune and evaluate heart-disease classifiers; save the best pipeline.

Models: Logistic Regression and Random Forest, each tuned with GridSearchCV
(5-fold stratified CV). Tuning and final model selection use **recall**, because
missing a patient who has heart disease (false negative) is the costliest error
in a screening setting. The final model is chosen on mean CV recall (ROC-AUC
as tie-breaker); the held-out test set is only used for the final report.

Usage:
    python -m src.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_processing import TARGET, load_clean  # noqa: E402
from src.features import FEATURES, build_preprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "screenshots" / "training"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlartifacts").as_uri()
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL = "heart-disease-classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SELECTION_METRIC = "recall"
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

MODEL_SPECS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l1", "l2"],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 4, 8],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__class_weight": [None, "balanced"],
        },
    ),
}


def split_data(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", estimator)])


def evaluate(model, X, y) -> dict:
    """Compute classification metrics on a held-out set."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def tune(name, estimator, grid, X_train, y_train) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(estimator),
        grid,
        scoring=CV_SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Mean CV scores of the best parameter set, for every metric."""
    i = search.best_index_
    return {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in CV_SCORING}


def save_plots(name, model, X_test, y_test) -> list[Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["No disease", "Disease"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {name}")
    paths.append(PLOT_DIR / f"{name}_confusion_matrix.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title(f"ROC curve — {name}")
    paths.append(PLOT_DIR / f"{name}_roc_curve.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


def setup_mlflow() -> None:
    """Use a local SQLite tracking DB; store artifacts under mlartifacts/."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_parent_run(name, search, result, plots, X_train) -> str:
    """Log the tuned model's summary: best params, CV + test metrics, plots, model.

    Returns the logged model URI (used to register the selected model).
    """
    mlflow.set_tags({"model_type": name, "selection_metric": SELECTION_METRIC})
    mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
    mlflow.log_params(
        {
            "cv_folds": 5,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_candidates": len(search.cv_results_["params"]),
        }
    )
    mlflow.log_metrics({f"cv_{k}": v for k, v in result["cv"].items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in result["test"].items()})
    for path in plots:
        mlflow.log_artifact(str(path), artifact_path="plots")
    pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
        PLOT_DIR / f"{name}_cv_results.csv", index=False
    )
    mlflow.log_artifact(str(PLOT_DIR / f"{name}_cv_results.csv"), artifact_path="cv")
    # The model is trained right here, so the tree node structure used by
    # Random Forest is safe to whitelist for MLflow's skops serialisation.
    info = mlflow.sklearn.log_model(
        search.best_estimator_,
        name="model",
        input_example=X_train.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    return info.model_uri


def log_child_runs(name, search) -> None:
    """One nested run per hyperparameter candidate with its mean CV scores."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}_cand{i:02d}", nested=True):
            mlflow.set_tag("model_type", name)
            mlflow.log_params({k.replace("clf__", ""): v for k, v in params.items()})
            mlflow.log_metrics({f"cv_{m}": float(res[f"mean_test_{m}"][i]) for m in CV_SCORING})
            mlflow.log_metric("cv_rank_recall", int(res[f"rank_test_{SELECTION_METRIC}"][i]))


def main() -> None:
    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    setup_mlflow()

    results = {}
    for name, (estimator, grid) in MODEL_SPECS.items():
        with mlflow.start_run(run_name=name) as run:
            search = tune(name, estimator, grid, X_train, y_train)
            results[name] = {
                "best_params": search.best_params_,
                "cv": cv_summary(search),
                "test": evaluate(search.best_estimator_, X_test, y_test),
                "model": search.best_estimator_,
                "run_id": run.info.run_id,
            }
            plots = save_plots(name, search.best_estimator_, X_test, y_test)
            results[name]["model_uri"] = log_parent_run(name, search, results[name], plots, X_train)
            log_child_runs(name, search)
        print(f"\n== {name} ==")
        print("best params:", search.best_params_)
        print("CV  :", {k: round(v, 3) for k, v in results[name]["cv"].items()})
        print("Test:", {k: round(v, 3) for k, v in results[name]["test"].items()})

    # Select on CV scores so the test set remains an unbiased final check.
    best_name = max(
        results,
        key=lambda n: (results[n]["cv"][SELECTION_METRIC], results[n]["cv"]["roc_auc"]),
    )
    best = results[best_name]
    print(f"\nSelected model: {best_name} (by CV {SELECTION_METRIC})")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "model.joblib")
    metadata = {
        "model_name": best_name,
        "selection_metric": SELECTION_METRIC,
        "features": FEATURES,
        "best_params": best["best_params"],
        "cv_metrics": best["cv"],
        "test_metrics": best["test"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_models": {
            n: {"cv": r["cv"], "test": r["test"], "best_params": r["best_params"]}
            for n, r in results.items()
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved {MODEL_DIR / 'model.joblib'} and metadata.json")

    # Mark the winning run in MLflow so it is easy to find in the UI.
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_model", "true")
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"))
    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    print(f"Registered '{REGISTERED_MODEL}' version {version.version}")
    print(f"View runs with: mlflow ui --backend-store-uri {MLFLOW_URI}")


if __name__ == "__main__":
    main()
