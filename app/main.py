import json
import random

import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.config import MODEL_PATH, SAMPLE_CLAIMS_PATH
from app.preprocess import load_preprocess_config, predict_medicare
from app.schemas import (
    ClaimFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

artifact: dict = {}

LABEL_MAP = {0: "Not Reportable", 1: "Medicare Reportable"}


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
