import json
import logging

import pandas as pd

from app.config import PREPROCESS_CONFIG_PATH

MODEL_FEATURES = [
    "data_set",
    "pay_cat",
    "pay_code",
    "pay_type",
    "paid_1",
    "paid_3",
    "amount",
    "proc_unit",
    "cont_num",
    "claim_open",
    "date_v1m_xmit_flag",
    "is_us_claimant",
    "orm_threshold_met",
    "tpoc_threshold_met",
    "is_wc",
    "pay_code_bucket",
    "is_excluded_coverage",
    "is_excluded_line",
    "days_open",
    "age_at_event",
]

CAT_COLS = ["data_set", "pay_cat", "pay_type", "pay_code_bucket"]
NUM_COLS = [c for c in MODEL_FEATURES if c not in CAT_COLS]


def load_preprocess_config() -> dict:
    with open(PREPROCESS_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def prepare_features(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    config = load_preprocess_config()
    row = {feature: payload.get(feature) for feature in MODEL_FEATURES}
    df = pd.DataFrame([row])

    for col in CAT_COLS:
        df[col] = df[col].astype(str).replace({"nan": None, "None": None})
        df[col] = df[col].fillna(config["cat_impute"][col])

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(config["num_impute"][col])

    for col in [
        "claim_open",
        "date_v1m_xmit_flag",
        "is_us_claimant",
        "orm_threshold_met",
        "tpoc_threshold_met",
        "is_wc",
        "is_excluded_coverage",
        "is_excluded_line",
    ]:
        df[col] = df[col].astype(int)

    encoded = pd.get_dummies(df[MODEL_FEATURES], columns=CAT_COLS, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.astype(float)


def build_feature_df(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Encoded feature matrix for model input."""
    return prepare_features(payload, feature_columns)


def get_shap_drivers(
    model,
    input_df: pd.DataFrame,
    feature_columns: list,
    top_n: int = 3,
) -> list[dict]:
    """
    Calculate SHAP values for a single prediction.
    Returns top N features driving the prediction.
    In production: explains why claim is/isn't Medicare reportable.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df[feature_columns])

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        drivers = []
        for feat, val in zip(feature_columns, values):
            drivers.append(
                {
                    "feature": feat,
                    "value": float(input_df[feat].iloc[0]),
                    "shap_value": round(float(val), 4),
                    "impact": "increases_reportability"
                    if val > 0
                    else "decreases_reportability",
                }
            )

        drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return drivers[:top_n]

    except Exception as e:
        logging.warning("SHAP calculation failed: %s", e)
        return []


def predict_medicare(model, payload: dict, feature_columns: list[str]) -> tuple[int, float]:
    features = prepare_features(payload, feature_columns)
    proba = float(model.predict_proba(features)[0][1])
    label = 1 if proba >= 0.5 else 0
    return label, proba
