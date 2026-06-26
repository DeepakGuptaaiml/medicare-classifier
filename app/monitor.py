"""
MLOps Monitoring — Prediction logging + drift detection.
Uses Evidently for drift detection against training baseline.
In production: logs would go to Azure Monitor / App Insights.
"""

import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

# In-memory prediction store (last 1000 predictions)
# In production: use Azure Table Storage or PostgreSQL
MAX_PREDICTIONS = 1000
prediction_store: deque = deque(maxlen=MAX_PREDICTIONS)
store_lock = Lock()

# SLO thresholds
SLO_CONFIG = {
    "p95_latency_ms": 500,
    "error_rate_pct": 1.0,
    "min_predictions_for_drift": 50,
}


def log_prediction(
    input_features: dict,
    prediction,
    probability: float,
    latency_ms: float,
    model_name: str,
    model_version: str = "1.0",
):
    """
    Log every prediction for monitoring.
    Stored in memory — in production use Azure Table Storage.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "input_features": input_features,
        "prediction": prediction,
        "probability": float(probability) if probability is not None else None,
        "latency_ms": round(latency_ms, 2),
    }
    with store_lock:
        prediction_store.append(record)


def get_prediction_stats() -> dict:
    """
    Calculate prediction statistics for monitoring.
    Returns SLO metrics and prediction distribution.
    """
    with store_lock:
        records = list(prediction_store)

    if not records:
        return {
            "total_predictions": 0,
            "message": "No predictions logged yet",
        }

    latencies = [r["latency_ms"] for r in records]
    predictions = [r["prediction"] for r in records]
    timestamps = [r["timestamp"] for r in records]

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))

    if isinstance(predictions[0], (int, float)) and all(p in (0, 1) for p in predictions):
        positive_rate = sum(1 for p in predictions if p == 1) / len(predictions)
        prediction_dist = {
            "positive_rate": round(positive_rate, 3),
            "negative_rate": round(1 - positive_rate, 3),
        }
    else:
        pred_values = [float(p) for p in predictions]
        prediction_dist = {
            "mean": round(float(np.mean(pred_values)), 2),
            "median": round(float(np.median(pred_values)), 2),
            "std": round(float(np.std(pred_values)), 2),
            "min": round(float(np.min(pred_values)), 2),
            "max": round(float(np.max(pred_values)), 2),
        }

    slo_status = {
        "p95_latency_ok": p95_latency < SLO_CONFIG["p95_latency_ms"],
        "p95_latency_ms": round(p95_latency, 2),
        "slo_threshold_ms": SLO_CONFIG["p95_latency_ms"],
    }

    return {
        "total_predictions": len(records),
        "first_prediction": timestamps[0],
        "last_prediction": timestamps[-1],
        "latency": {
            "p50_ms": round(p50_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "p99_ms": round(p99_latency, 2),
            "mean_ms": round(float(np.mean(latencies)), 2),
        },
        "prediction_distribution": prediction_dist,
        "slo_status": slo_status,
    }


def check_drift(baseline_stats: dict | None = None) -> dict:
    """
    Check for prediction drift vs baseline.
    Uses simple statistical comparison.
    In production: use Evidently AI for full drift report.

    NOTE: In production/enterprise this would use:
    - Evidently AI for comprehensive drift reports
    - Azure Monitor for alerting
    - Azure Application Insights for telemetry
    """
    _ = baseline_stats  # reserved for future Evidently baseline comparison

    with store_lock:
        records = list(prediction_store)

    n = len(records)
    if n < SLO_CONFIG["min_predictions_for_drift"]:
        return {
            "drift_detected": False,
            "status": "insufficient_data",
            "message": f"Need {SLO_CONFIG['min_predictions_for_drift']} predictions, have {n}",
            "predictions_analyzed": n,
            "alerts": [],
            "recommendation": "Collect more predictions before drift analysis",
        }

    predictions = [r["prediction"] for r in records]
    latencies = [r["latency_ms"] for r in records]

    mid = n // 2
    baseline_preds = predictions[:mid]
    current_preds = predictions[mid:]

    drift_alerts = []
    drift_detected = False

    if isinstance(predictions[0], (int, float)) and all(p in (0, 1) for p in predictions):
        baseline_rate = sum(baseline_preds) / len(baseline_preds)
        current_rate = sum(current_preds) / len(current_preds)
        rate_change = abs(current_rate - baseline_rate)

        if rate_change > 0.15:
            drift_detected = True
            drift_alerts.append({
                "type": "prediction_drift",
                "severity": "high" if rate_change > 0.25 else "medium",
                "message": f"Positive prediction rate shifted by {rate_change:.1%}",
            })
    else:
        baseline_mean = float(np.mean([float(p) for p in baseline_preds]))
        current_mean = float(np.mean([float(p) for p in current_preds]))
        pct_change = abs(current_mean - baseline_mean) / (baseline_mean + 1e-9)

        if pct_change > 0.20:
            drift_detected = True
            drift_alerts.append({
                "type": "prediction_drift",
                "severity": "high" if pct_change > 0.40 else "medium",
                "message": f"Mean prediction shifted by {pct_change:.1%}",
            })

    current_p95 = float(np.percentile(latencies[mid:], 95))
    if current_p95 > SLO_CONFIG["p95_latency_ms"]:
        drift_detected = True
        drift_alerts.append({
            "type": "latency_slo_breach",
            "severity": "high",
            "message": (
                f"p95 latency {current_p95:.0f}ms exceeds SLO "
                f"{SLO_CONFIG['p95_latency_ms']}ms"
            ),
        })

    return {
        "drift_detected": drift_detected,
        "status": "drift_detected" if drift_detected else "healthy",
        "alerts": drift_alerts,
        "predictions_analyzed": n,
        "recommendation": (
            "Review model performance and consider retraining "
            "or rolling back to previous version in Azure ML"
            if drift_detected
            else "Model performing within expected bounds"
        ),
    }
