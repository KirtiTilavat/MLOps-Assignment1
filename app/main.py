"""FastAPI service that serves the heart-disease risk model.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.predict import load_metadata, load_model, predict

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load once at startup so every request reuses the same pipeline.
    state["model"] = load_model()
    state["metadata"] = load_metadata()
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the risk of heart disease from 13 clinical attributes "
    "(UCI Heart Disease, Cleveland).",
    version="1.0.0",
    lifespan=lifespan,
)


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
def predict_endpoint(patient: PatientFeatures) -> PredictionResponse:
    [result] = predict(patient.model_dump(), model=state["model"])
    return PredictionResponse(**result, model_name=state["metadata"].get("model_name", "unknown"))
