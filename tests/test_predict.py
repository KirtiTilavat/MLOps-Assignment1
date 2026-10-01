import pytest

from src.predict import predict, risk_level

EXPECTED_KEYS = {"prediction", "label", "confidence", "probability_disease", "risk_level"}


def test_single_record_output_fields(fitted_model, patient):
    [res] = predict(patient, model=fitted_model)
    assert set(res) == EXPECTED_KEYS
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1
    assert res["label"] == ("disease" if res["prediction"] else "no disease")


def test_confidence_matches_probability(fitted_model, patient):
    [res] = predict(patient, model=fitted_model)
    p = res["probability_disease"]
    assert res["confidence"] == pytest.approx(p if res["prediction"] else 1 - p, abs=1e-4)


def test_batch_prediction(fitted_model, patient):
    low_risk = {**patient, "cp": 2, "thalach": 190, "exang": 0, "oldpeak": 0.0, "ca": 0, "thal": 3}
    results = predict([patient, low_risk], model=fitted_model)
    assert len(results) == 2
    assert results[0]["probability_disease"] > results[1]["probability_disease"]


def test_missing_feature_raises(fitted_model, patient):
    bad = {k: v for k, v in patient.items() if k != "chol"}
    with pytest.raises(ValueError, match="chol"):
        predict(bad, model=fitted_model)


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, "low"),
        (0.349, "low"),
        (0.35, "moderate"),
        (0.64, "moderate"),
        (0.65, "high"),
        (1.0, "high"),
    ],
)
def test_risk_level_bands(p, expected):
    assert risk_level(p) == expected
