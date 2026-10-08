"""FastAPI service that serves the heart-disease risk model.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.monitoring import (
    MODEL_INFO,
    log_event,
    metrics_response,
    observe_request,
    record_prediction,
)
from app.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.predict import load_metadata, load_model, predict

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load once at startup so every request reuses the same pipeline.
    state["model"] = load_model()
    state["metadata"] = load_metadata()
    meta = state["metadata"]
    MODEL_INFO.labels(meta.get("model_name", "unknown"), meta.get("trained_at", "unknown")).set(1)
    log_event("model_loaded", model_name=meta.get("model_name"), trained_at=meta.get("trained_at"))
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the risk of heart disease from 13 clinical attributes "
    "(UCI Heart Disease, Cleveland).",
    version="1.0.0",
    lifespan=lifespan,
)
app.middleware("http")(observe_request)


@app.get("/")
def root() -> dict:
    meta = state.get("metadata", {})
    return {
        "service": "heart-disease-risk-api",
        "model_name": meta.get("model_name"),
        "trained_at": meta.get("trained_at"),
        "test_metrics": meta.get("test_metrics"),
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = "model" in state
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientFeatures, request: Request) -> PredictionResponse:
    features = patient.model_dump()
    [result] = predict(features, model=state["model"])
    record_prediction(request.state.request_id, features, result)
    return PredictionResponse(**result, model_name=state["metadata"].get("model_name", "unknown"))


@app.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()
