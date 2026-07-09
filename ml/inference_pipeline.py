"""Inference pipeline: raw claim features → encoded matrix → prediction + SHAP."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from ml.constants import CAT_COLS, FEATURES, INT_COLS, MODEL_FEATURES, NUM_COLS


class InferencePipeline:
    def __init__(
        self,
        model,
        feature_columns: list[str],
        preprocess_config: dict,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.preprocess_config = preprocess_config

    @classmethod
    def from_artifact(cls, artifact: dict, preprocess_config: dict) -> InferencePipeline:
        return cls(
            model=artifact["model"],
            feature_columns=artifact["feature_columns"],
            preprocess_config=preprocess_config,
        )

    @classmethod
    def from_paths(cls, model_path: Path, config_path: Path) -> InferencePipeline:
        artifact = joblib.load(model_path)
        import json

        with open(config_path, encoding="utf-8") as f:
            preprocess_config = json.load(f)
        return cls.from_artifact(artifact, preprocess_config)

    def _impute_raw_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        config = self.preprocess_config
        out = df.copy()

        for col in CAT_COLS:
            out[col] = out[col].astype(str).replace({"nan": None, "None": None})
            out[col] = out[col].fillna(config["cat_impute"][col])

        for col in NUM_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(config["num_impute"][col])

        for col in INT_COLS:
            out[col] = out[col].astype(int)

        return out

    def encode_raw(self, payload: dict) -> pd.DataFrame:
        row = {feature: payload.get(feature) for feature in FEATURES}
        df = self._impute_raw_frame(pd.DataFrame([row]))
        encoded = pd.get_dummies(df[MODEL_FEATURES], columns=CAT_COLS, drop_first=True)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
        return encoded.astype(float)

    def predict_one(self, payload: dict) -> tuple[int, float]:
        features = self.encode_raw(payload)
        proba = float(self.model.predict_proba(features)[0][1])
        label = 1 if proba >= 0.5 else 0
        return label, proba

    def get_shap_drivers(
        self,
        input_df: pd.DataFrame | None = None,
        payload: dict | None = None,
        top_n: int = 3,
    ) -> list[dict]:
        """
        Top feature drivers for a single prediction.
        Uses SHAP KernelExplainer; falls back to feature_importances_.
        """
        if input_df is None:
            if payload is None:
                raise ValueError("Provide input_df or payload")
            input_df = self.encode_raw(payload)

        feature_columns = self.feature_columns
        try:
            import shap
            import numpy as np

            x_matrix = input_df[feature_columns]
            background = np.zeros((1, len(feature_columns)))
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background,
                link="logit",
            )
            shap_values = explainer.shap_values(x_matrix, nsamples=32)

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            drivers = []
            for feat, val in zip(feature_columns, values):
                drivers.append(
                    {
                        "feature": feat,
                        "value": float(x_matrix[feat].iloc[0]),
                        "shap_value": round(float(val), 4),
                        "impact": "increases_reportability"
                        if val > 0
                        else "decreases_reportability",
                    }
                )

            drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return drivers[:top_n]

        except Exception as exc:
            logging.warning("SHAP failed, using feature importance: %s", exc)

            try:
                importances = self.model.feature_importances_
                drivers = []
                input_vals = input_df[feature_columns].iloc[0]
                for feat, imp, val in zip(feature_columns, importances, input_vals):
                    drivers.append(
                        {
                            "feature": feat,
                            "value": float(val),
                            "shap_value": round(float(imp), 4),
                            "impact": "increases_reportability"
                            if val > 0
                            else "decreases_reportability",
                        }
                    )
                drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                return drivers[:top_n]
            except Exception as exc2:
                logging.warning("Feature importance fallback failed: %s", exc2)
                return []
