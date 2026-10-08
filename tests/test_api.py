import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.monitoring import JsonFormatter


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the startup (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_root_reports_model(client):
    body = client.get("/").json()
    assert body["model_name"]
    assert "test_metrics" in body


def test_predict_valid_patient(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_level"] in ("low", "moderate", "high")


def test_predict_high_vs_low_risk(client, patient):
    low = {**patient, "cp": 2, "thalach": 185, "exang": 0, "oldpeak": 0.0, "ca": 0, "thal": 3}
    high_p = client.post("/predict", json=patient).json()["probability_disease"]
    low_p = client.post("/predict", json=low).json()["probability_disease"]
    assert high_p > low_p


@pytest.mark.parametrize(
    "change",
    [{"age": 500}, {"cp": 9}, {"thal": 5}, {"chol": -1}, {"extra_field": 1}],
)
def test_predict_rejects_invalid_values(client, patient, change):
    assert client.post("/predict", json={**patient, **change}).status_code == 422


def test_predict_rejects_missing_field(client, patient):
    patient.pop("thalach")
    assert client.post("/predict", json=patient).status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(client, patient):
    client.post("/predict", json=patient)
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert 'heart_api_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_request_duration_seconds_bucket" in text
    assert "heart_api_model_info" in text


def test_invalid_requests_are_counted_by_status(client, patient):
    client.post("/predict", json={**patient, "age": 500})
    text = client.get("/metrics").text
    assert 'heart_api_requests_total{method="POST",path="/predict",status="422"}' in text


def test_request_id_header_and_request_logs(client, patient, caplog):
    with caplog.at_level(logging.INFO, logger="heart_api"):
        r = client.post("/predict", json=patient, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    events = {
        rec.getMessage(): rec.extra_fields
        for rec in caplog.records
        if getattr(rec, "extra_fields", {}).get("request_id") == "abc123"
    }
    assert events["request"]["status"] == 200
    assert events["request"]["path"] == "/predict"
    assert events["prediction"]["risk_level"] == r.json()["risk_level"]


def test_log_lines_are_json():
    record = logging.LogRecord("heart_api", logging.INFO, "", 0, "request", None, None)
    record.extra_fields = {"status": 200, "path": "/predict"}
    entry = json.loads(JsonFormatter().format(record))
    assert entry["msg"] == "request" and entry["status"] == 200 and entry["level"] == "INFO"
