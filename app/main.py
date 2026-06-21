import json
import random
import time

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from agent.rag_agent import RAGConfigurationError, get_rag_agent
from app.config import HF_API_TOKEN, RAG_DOCS_PATH, RAG_MODEL_ID
from app.config import MODEL_PATH, SAMPLE_CLAIMS_PATH
from app.preprocess import load_preprocess_config, predict_medicare
from app.schemas import (
    AskRequest,
    AskResponse,
    ClaimFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RAGStatusResponse,
)

artifact: dict = {}

# Lazy singleton for RAG agent — initialized on first /ask request
_rag_agent = None

LABEL_MAP = {0: "Not Reportable", 1: "Medicare Reportable"}

RAG_UNAVAILABLE_MSG = (
    "RAG agent unavailable: HF_API_TOKEN not configured. "
    "Set HF_API_TOKEN environment variable to enable."
)


def _load_sample_claims() -> list[dict]:
    if not SAMPLE_CLAIMS_PATH.exists():
        return []
    with open(SAMPLE_CLAIMS_PATH, encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    artifact["data"] = joblib.load(MODEL_PATH)
    artifact["config"] = load_preprocess_config()
    yield
    artifact.clear()


app = FastAPI(
    title="Medicare Classifier API",
    description="Predict Medicare reportable claims under MMSEA Section 111 (V1M extraction).",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_artifact() -> dict:
    data = artifact.get("data")
    if data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return data


@app.get("/health", response_model=HealthResponse)
def health():
    data = artifact.get("data")
    return HealthResponse(
        status="ok" if data else "starting",
        model_loaded=data is not None,
        model_name=data.get("model_name") if data else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    data = _get_artifact()
    return ModelInfoResponse(
        model_name=data["model_name"],
        sampling_strategy=data["sampling_strategy"],
        target=data["target"],
        feature_count=len(data["feature_columns"]),
        metrics_test=data.get("metrics_test", {}),
    )


@app.get("/model/options")
def model_options():
    config = artifact.get("config") or load_preprocess_config()
    return config.get("categorical_options", {})


@app.get("/model/sample", response_model=ClaimFeatures)
def model_sample():
    samples = _load_sample_claims()
    if not samples:
        raise HTTPException(status_code=503, detail="No sample claims available")
    return random.choice(samples)


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimFeatures):
    data = _get_artifact()
    try:
        label, proba = predict_medicare(
            data["model"],
            claim.model_dump(),
            data["feature_columns"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        is_medicare_reportable=label,
        probability=round(proba, 4),
        label=LABEL_MAP[label],
        model_name=data["model_name"],
        target=data["target"],
    )


def _lazy_load_rag_agent():
    """Initialize RAG agent on first use (singleton, thread-safe in rag_agent module)."""
    global _rag_agent
    if not HF_API_TOKEN:
        raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_MSG)
    if _rag_agent is None:
        try:
            _rag_agent = get_rag_agent()
        except RAGConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _rag_agent


def _count_policy_documents() -> int:
    if not RAG_DOCS_PATH.exists():
        return 0
    return len(list(RAG_DOCS_PATH.glob("*.txt")))


@app.get("/rag/status", response_model=RAGStatusResponse, tags=["RAG Agent"])
def rag_status():
    """Check if RAG agent is loaded and ready."""
    doc_count = _count_policy_documents()
    if not HF_API_TOKEN:
        return RAGStatusResponse(
            status="unavailable",
            documents_loaded=doc_count,
            vector_store_ready=False,
            llm_ready=False,
        )
    if _rag_agent is None:
        return RAGStatusResponse(
            status="ready_to_load",
            documents_loaded=doc_count,
            vector_store_ready=False,
            llm_ready=False,
        )
    status = _rag_agent.get_status()
    return RAGStatusResponse(
        status="ok",
        documents_loaded=status["documents_loaded"],
        vector_store_ready=status["vector_store_ready"],
        llm_ready=status["llm_ready"],
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Agent"])
def ask_policy_question(request: AskRequest):
    """
    Ask a question about Medicare claims policy.
    Uses RAG to retrieve relevant context from policy documents
    and generate an answer using LLM.
    In production: would use Azure OpenAI Service for HIPAA compliance.
    """
    started = time.perf_counter()
    agent = _lazy_load_rag_agent()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please provide a question")

    result = agent.ask(request.question, max_chunks=request.max_chunks)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        model_used=RAG_MODEL_ID,
        processing_time_ms=round(elapsed_ms, 2),
    )
