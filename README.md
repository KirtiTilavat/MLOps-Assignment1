# Heart Disease Risk Prediction — End-to-End MLOps

[![CI](https://github.com/KirtiTilavat/MLOps-Assignment1/actions/workflows/ci.yml/badge.svg)](https://github.com/KirtiTilavat/MLOps-Assignment1/actions/workflows/ci.yml)

MLOps Assignment 01 (AIMLCZG523). A classifier that predicts heart-disease risk from
patient health data (UCI Heart Disease dataset), served as a containerised, monitored API.

## Project structure

```
data/            raw/ (downloaded, git-ignored) and processed/ (cleaned CSV)
scripts/         data download script
notebooks/       EDA, training and inference notebooks
src/             data processing, feature pipeline, training, prediction
app/             FastAPI serving application
tests/           pytest unit tests
models/          saved model pipeline + metadata
k8s/             Kubernetes manifests
monitoring/      Prometheus / Grafana configuration
.github/         GitHub Actions CI/CD workflow
screenshots/     images used in the report
report/          final written report
```

## Setup

Requires Python 3.13, Docker Desktop (with Kubernetes enabled) and kubectl.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows note:** keep the project in a short path (e.g. `C:\Users\<you>\...\assignment1`).
> Jupyter installs files with very long names and pip fails if the total path exceeds 260 characters.

## Run the pipeline

```powershell
python scripts/download_data.py   # 1. download raw UCI Cleveland data -> data/raw/
python -m src.data_processing     # 2. clean -> data/processed/heart_clean.csv
python -m src.train               # 3. tune + evaluate models, log to MLflow, save best model
python -m src.predict             # 4. score a sample patient with the saved model
```

Notebooks: `notebooks/01_eda.ipynb` (EDA), `02_training.ipynb` (modelling walkthrough),
`03_inference.ipynb` (using the saved / registered model).

## Experiment tracking (MLflow)

Training logs to a local SQLite store (`mlflow.db`, artifacts in `mlartifacts/`):
one parent run per model with best params, CV + test metrics, plots and the model,
plus a nested child run for every hyperparameter candidate. The selected model is
registered as `heart-disease-classifier`.

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000/#/experiments/1/runs
```

## Model

The saved artifact `models/model.joblib` is a single scikit-learn `Pipeline`
(preprocessing + classifier), so raw patient values go in and predictions come out.
`models/metadata.json` records the features, chosen hyperparameters and metrics.

## Tests and linting

```powershell
pytest --cov                      # 24 unit tests on synthetic data (offline, ~6 s)
flake8 src tests scripts app
black --check src tests scripts app
```

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push / pull request to `main`:

| Job | Steps | Artifacts |
|---|---|---|
| **lint** | flake8, black --check | — |
| **test** (needs lint) | pytest with coverage | `test-reports` (JUnit XML, coverage XML) |
| **train** (needs test) | download → clean → train + MLflow → smoke-test prediction | `model-and-training-artifacts` (model, metadata, plots, MLflow DB) |

Each job only runs if the previous one succeeded, so a lint or test error fails the
pipeline with a clear log. The train job also writes a metrics table to the run summary.

_Further sections (API, deployment, monitoring) are added as each phase is completed._
