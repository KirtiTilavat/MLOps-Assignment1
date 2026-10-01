import pytest
from fastapi.testclient import TestClient

from app.main import app


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
