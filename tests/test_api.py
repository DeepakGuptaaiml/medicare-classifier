import joblib
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import app.main as main_module
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


@pytest.fixture
def mock_rag_agent(monkeypatch):
    """Mock RAG agent so CI tests do not call Hugging Face APIs."""
    fake = MagicMock()
    fake.get_status.return_value = {
        "documents_loaded": 3,
        "vector_store_ready": True,
        "llm_ready": True,
    }

    def _ask(question: str, max_chunks: int = 3):
        if "quantum physics" in question.lower():
            return {
                "question": question,
                "answer": "I cannot find this in the policy documents.",
                "sources": [],
                "chunks_used": [],
            }
        return {
            "question": question,
            "answer": (
                "ORM threshold for WC claims: paid_3 (medical paid) must exceed "
                "$750.00, effective 01/01/2010 (mci_reference.txt)."
            ),
            "sources": ["mci_reference.txt"],
            "chunks_used": ["ORM threshold: paid_3 > $750.00 effective 01/01/2010"],
        }

    fake.ask.side_effect = _ask
    monkeypatch.setenv("HF_API_TOKEN", "test-token-for-ci")
    monkeypatch.setattr(main_module, "get_hf_api_token", lambda: "test-token-for-ci")
    main_module.rag_agent_instance = fake
    yield fake
    main_module.rag_agent_instance = None


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


def test_rag_status_endpoint(client):
    response = client.get("/rag/status")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "hf_token_configured" in body


def test_rag_status_endpoint_loaded(mock_rag_agent, client):
    response = client.get("/rag/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ready", "ready_to_load")
    assert body["hf_token_configured"] is True


def test_ask_valid_question(mock_rag_agent, client):
    response = client.post(
        "/ask",
        json={"question": "What are the ORM threshold rules for WC claims?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "ORM" in body["answer"]
    assert body["processing_time_ms"] > 0


def test_ask_empty_question(client, mock_rag_agent):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_response_has_sources(mock_rag_agent, client):
    response = client.post(
        "/ask",
        json={"question": "What is MMSEA Section 111?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) >= 1


def test_ask_processing_time(mock_rag_agent, client):
    response = client.post(
        "/ask",
        json={"question": "What are the pay code differences between WC and Non-WC?"},
    )
    assert response.status_code == 200
    assert response.json()["processing_time_ms"] > 0


def test_ask_out_of_scope(mock_rag_agent, client):
    response = client.post(
        "/ask",
        json={"question": "Explain quantum physics in detail"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "cannot find" in body["answer"].lower()


def test_ask_without_hf_token(monkeypatch, client):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_KEY", raising=False)
    monkeypatch.setattr(main_module, "get_hf_api_token", lambda: "")
    monkeypatch.setattr(main_module, "azure_search_configured", lambda: False)
    main_module.rag_agent_instance = None
    response = client.post(
        "/ask",
        json={"question": "What is MMSEA Section 111?"},
    )
    assert response.status_code == 503
    assert "AZURE_SEARCH" in response.json()["detail"] or "HF_API_TOKEN" in response.json()["detail"]


def test_ask_with_azure_search_only(monkeypatch, client):
    """RAG works with Azure Search even when HF token is absent."""
    fake = MagicMock()
    fake.get_status.return_value = {
        "documents_loaded": 0,
        "vector_store_ready": True,
        "llm_ready": False,
        "retrieval_backend": "azure_search",
    }
    fake.ask.return_value = {
        "question": "What are the ORM threshold rules?",
        "answer": "Based on policy documents:\n\nORM threshold: paid_3 > $750.00",
        "sources": ["mci_reference.txt"],
        "chunks_used": ["ORM threshold: paid_3 > $750.00"],
    }

    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.setattr(main_module, "get_hf_api_token", lambda: "")
    monkeypatch.setattr(main_module, "azure_search_configured", lambda: True)
    main_module.rag_agent_instance = fake

    response = client.post(
        "/ask",
        json={"question": "What are the ORM threshold rules?"},
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == "retrieval-only"
    main_module.rag_agent_instance = None
