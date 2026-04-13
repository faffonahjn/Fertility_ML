"""Integration tests -- FastAPI endpoints."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.serving.api import app

client = TestClient(app)

HIGH_RISK = {
    "female_age": 42, "male_age": 45, "bmi": 31.0,
    "menstrual_regularity": "Irregular", "pcos": "Yes",
    "stress_level": "High", "smoking": "Yes",
    "alcohol_intake": "High", "sperm_count_million_per_ml": 12.0,
    "motility_pct": 22.0, "trying_duration_months": 30,
    "treatment_type": "None",
}
LOW_RISK = {
    "female_age": 26, "male_age": 28, "bmi": 22.0,
    "menstrual_regularity": "Regular", "pcos": "No",
    "stress_level": "Low", "smoking": "No",
    "alcohol_intake": "None", "sperm_count_million_per_ml": 75.0,
    "motility_pct": 80.0, "trying_duration_months": 6,
    "treatment_type": "IVF",
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_schema():
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert "pregnancy_success" in body
    assert "success_probability" in body
    assert "outcome_label" in body
    assert "risk_level" in body
    assert 0.0 <= body["success_probability"] <= 1.0

def test_low_risk_predicts_success():
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    assert r.json()["pregnancy_success"] == 1

def test_high_risk_predicts_failure():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    assert r.json()["pregnancy_success"] == 0

def test_batch_predict():
    r = client.post("/predict/batch", json=[LOW_RISK, HIGH_RISK])
    assert r.status_code == 200
    body = r.json()
    assert body["total_records"] == 2
    assert body["success_count"] + body["failure_count"] == 2

def test_invalid_pcos_rejected():
    r = client.post("/predict", json={**LOW_RISK, "pcos": "Maybe"})
    assert r.status_code == 422

def test_batch_limit():
    r = client.post("/predict/batch", json=[LOW_RISK] * 501)
    assert r.status_code == 400
