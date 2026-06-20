import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocess import predict_medicare


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_payload():
    return {
        "data_set": "WC",
        "pay_cat": "PI",
        "pay_code": 328,
        "pay_type": "SYS",
        "paid_1": 24488.39,
        "paid_3": 47490.06,
        "amount": 5019.64,
        "proc_unit": 49,
        "cont_num": 5414,
        "claim_open": 1,
        "date_v1m_xmit_flag": 0,
        "is_us_claimant": 1,
        "orm_threshold_met": 1,
        "tpoc_threshold_met": 1,
        "is_wc": 1,
        "pay_code_bucket": "ORM",
        "is_excluded_coverage": 0,
        "is_excluded_line": 0,
        "days_open": 0.0,
        "age_at_event": 70.5,
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["target"] == "is_medicare_reportable"
    assert body["feature_count"] > 0


def test_predict(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_medicare_reportable"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Medicare Reportable", "Not Reportable")


def test_model_sample(client):
    response = client.get("/model/sample")
    assert response.status_code == 200
    body = response.json()
    assert body["data_set"]
    predict_response = client.post("/predict", json=body)
    assert predict_response.status_code == 200


def test_predict_preprocess_unit(sample_payload):
    artifact = joblib.load("models/medicare_classifier.pkl")
    label, proba = predict_medicare(
        artifact["model"], sample_payload, artifact["feature_columns"]
    )
    assert label in (0, 1)
    assert 0.0 <= proba <= 1.0
