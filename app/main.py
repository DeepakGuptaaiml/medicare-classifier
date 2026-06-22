import asyncio
import json
import logging
import random
import time

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.config import get_hf_api_token, MODEL_PATH, SAMPLE_CLAIMS_PATH
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

# Global singleton — None until first /ask request
rag_agent_instance = None  # MedicareRAGAgent, loaded lazily
rag_agent_lock = asyncio.Lock()

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


async def get_rag_agent():
    """
    Lazy singleton — loads RAG agent on first call only.
    Thread-safe via asyncio lock.
    """
    global rag_agent_instance
    if rag_agent_instance is None:
        async with rag_agent_lock:
            if rag_agent_instance is None:
                token = get_hf_api_token()
                if not token:
                    return None
                try:
                    # Lazy import — only when first /ask request comes in
                    # This prevents startup crash if RAG deps have issues
                    from agent.rag_agent import MedicareRAGAgent

                    rag_agent_instance = MedicareRAGAgent()
                except Exception as e:
                    logging.error(f"RAG agent failed to initialize: {e}")
                    return None
    return rag_agent_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    artifact["data"] = joblib.load(MODEL_PATH)
    artifact["config"] = load_preprocess_config()
    token_ok = bool(get_hf_api_token())
    logging.getLogger("uvicorn.error").info(
        "Medicare Classifier API ready | model=Adaboost | RAG HF_API_TOKEN configured=%s",
        token_ok,
    )
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


def _count_policy_documents() -> int:
    from app.config import RAG_DOCS_PATH

    if not RAG_DOCS_PATH.exists():
        return 0
    return len(list(RAG_DOCS_PATH.glob("*.txt")))


@app.get("/rag/status", response_model=RAGStatusResponse, tags=["RAG Agent"])
async def rag_status():
    """Check if RAG agent is loaded and ready."""
    token = get_hf_api_token()
    token_configured = bool(token and len(token) > 10)
    agent_ready = rag_agent_instance is not None
    doc_count = _count_policy_documents() if token_configured else 0

    if not token_configured:
        status = "unavailable"
    elif agent_ready:
        status = "ready"
    else:
        status = "ready_to_load"

    return RAGStatusResponse(
        status=status,
        documents_loaded=doc_count,
        vector_store_ready=agent_ready,
        llm_ready=agent_ready,
        hf_token_configured=token_configured,
        hf_token_length=len(token) if token else 0,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Agent"])
async def ask_policy_question(request: AskRequest):
    """
    Ask a question about Medicare claims policy.
    Uses RAG to retrieve relevant context from policy documents
    and generate an answer using LLM.
    In production: would use Azure OpenAI Service for HIPAA compliance.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please provide a question")

    agent = await get_rag_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_MSG)

    start_time = time.time()
    result = agent.ask(request.question, max_chunks=request.max_chunks)
    processing_time = (time.time() - start_time) * 1000

    from app.config import RAG_MODEL_ID

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        model_used=RAG_MODEL_ID,
        processing_time_ms=round(processing_time, 2),
    )
