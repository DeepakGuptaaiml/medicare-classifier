"""API preprocessing — delegates to ml.inference_pipeline (train/serve parity)."""

import pandas as pd

from app.config import PREPROCESS_CONFIG_PATH
from ml.constants import MODEL_FEATURES
from ml.inference_pipeline import InferencePipeline


def load_preprocess_config() -> dict:
    import json

    with open(PREPROCESS_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _pipeline(model, feature_columns: list[str]) -> InferencePipeline:
    return InferencePipeline(
        model=model,
        feature_columns=feature_columns,
        preprocess_config=load_preprocess_config(),
    )


def prepare_features(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    return _pipeline(model=None, feature_columns=feature_columns).encode_raw(payload)


def build_feature_df(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Encoded feature matrix for model input."""
    return prepare_features(payload, feature_columns)


def get_shap_drivers(
    model,
    input_df: pd.DataFrame,
    feature_columns: list,
    top_n: int = 3,
) -> list[dict]:
    return _pipeline(model, feature_columns).get_shap_drivers(
        input_df=input_df,
        top_n=top_n,
    )


def predict_medicare(model, payload: dict, feature_columns: list[str]) -> tuple[int, float]:
    return _pipeline(model, feature_columns).predict_one(payload)


__all__ = [
    "MODEL_FEATURES",
    "build_feature_df",
    "get_shap_drivers",
    "load_preprocess_config",
    "predict_medicare",
    "prepare_features",
]
