"""Classification metrics shared by training and inference pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def eval_classifier(model, x, y) -> dict:
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else None
    result = {
        "accuracy": accuracy_score(y, pred),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if proba is not None and len(np.unique(y)) > 1:
        result["roc_auc"] = roc_auc_score(y, proba)
    else:
        result["roc_auc"] = float("nan")
    result["confusion_matrix"] = confusion_matrix(y, pred).tolist()
    return result
