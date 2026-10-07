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

## Prediction API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | patient JSON (13 validated fields) → prediction, confidence, probability, risk level |
| GET | `/health` | liveness / readiness check |
| GET | `/` | model name, training date, test metrics |
| GET | `/docs` | interactive Swagger UI |

Run without Docker: `uvicorn app.main:app --port 8000`

## Docker

```powershell
docker build -t heart-disease-api:1.1 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.1

# PowerShell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -InFile sample_request.json
# bash / Git Bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Example response:

```json
{"prediction": 1, "label": "disease", "confidence": 0.9337,
 "probability_disease": 0.9337, "risk_level": "high", "model_name": "logistic_regression"}
```

The Dockerfile is a **multi-stage build**:

1. **trainer** stage — installs the full `requirements.txt`, downloads the data and trains
   the model (`src.train`), so the model is reproducibly built from source.
2. **runtime** stage — installs only `requirements-api.txt` and copies the trained
   `model.joblib` + `metadata.json` from the trainer stage.

This reduced the image from 1.11 GB (single stage, full requirements) to ~486 MB.
The container runs as a non-root user and has a Docker `HEALTHCHECK` on `/health`.

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
| **docker** (needs test) | build image, start container, test `/health`, `/predict`, invalid input → 422 | container logs |
| **train** (needs test) | download → clean → train + MLflow → smoke-test prediction | `model-and-training-artifacts` (model, metadata, plots, MLflow DB) |

Each job only runs if the previous one succeeded, so a lint or test error fails the
pipeline with a clear log. The train job also writes a metrics table to the run summary.

## Kubernetes deployment (Docker Desktop)

Manifests in `k8s/`: a dedicated `heart-disease` namespace, a 2-replica `Deployment`
(rolling update with `maxUnavailable: 0`, CPU/memory requests and limits, readiness and
liveness probes on `/health`, non-root) and a `LoadBalancer` `Service` on port 80.

Enable Kubernetes in Docker Desktop (Settings → Kubernetes), then:

```powershell
kubectl config use-context docker-desktop
docker build -t heart-disease-api:1.1 .   # Docker Desktop k8s uses the local image store

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl -n heart-disease rollout status deployment/heart-disease-api
kubectl -n heart-disease get pods,svc

# Service is exposed at http://localhost (port 80)
Invoke-RestMethod http://localhost/health
Invoke-RestMethod -Method Post -Uri http://localhost/predict `
  -ContentType "application/json" -InFile sample_request.json
```

Tear down with `kubectl delete namespace heart-disease`.

_Further sections (monitoring) are added as each phase is completed._
