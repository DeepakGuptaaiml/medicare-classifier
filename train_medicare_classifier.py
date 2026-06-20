#!/usr/bin/env python3
"""Train Medicare Classifier and export medicare_classifier.pkl + preprocess_config.json."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RS = 1
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "claims_data.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET = "is_medicare_reportable"
FEATURES = [
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
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


def pay_code_bucket(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "OTHER"
    if code in (113, 120, 135, 137, 153):
        return "SETTLEMENT"
    if 100 <= code <= 199 or code == 390:
        return "TPOC"
    if 300 <= code <= 399:
        return "ORM"
    return "OTHER"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = pd.to_numeric(out["is_v1m_extracted"], errors="coerce").fillna(0).astype(int)

    for col in ["date_event", "date_open", "date_close", "clmnt_dob", "date_v1m_xmit"]:
        out[f"{col}_dt"] = pd.to_datetime(out[col], errors="coerce")

    out["date_v1m_xmit_flag"] = out["date_v1m_xmit_dt"].notna().astype(int)
    out["is_us_claimant"] = (out["clmnt_country"].astype(str) == "USA").astype(int)
    out["orm_threshold_met"] = (pd.to_numeric(out["paid_3"], errors="coerce").fillna(0) > 750).astype(int)
    out["tpoc_threshold_met"] = (pd.to_numeric(out["paid_1"], errors="coerce").fillna(0) > 0).astype(int)
    out["is_wc"] = (out["data_set"].astype(str) == "WC").astype(int)
    out["pay_code_bucket"] = out["pay_code"].apply(pay_code_bucket)
    out["is_excluded_coverage"] = out["coverage_code"].isin(["ZE", "WA", "LT"]).astype(int)
    out["is_excluded_line"] = out["line_code"].isin(["ZE", "GR"]).astype(int)

    out["days_open"] = (out["date_close_dt"] - out["date_open_dt"]).dt.days
    out.loc[out["claim_open"].astype(bool), "days_open"] = 0
    out["days_open"] = out["days_open"].fillna(0)

    out["age_at_event"] = (
        (out["date_event_dt"] - out["clmnt_dob_dt"]).dt.days / 365.25
    ).round(1)

    out["claim_open"] = out["claim_open"].astype(int)
    for col in NUM_COLS:
        if col != "claim_open":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = df[FEATURES].copy()
    for col in CAT_COLS:
        frame[col] = frame[col].astype(str).fillna("MISSING")
    encoded = pd.get_dummies(frame, columns=CAT_COLS, drop_first=True)
    return encoded, list(encoded.columns)


def get_models() -> dict:
    return {
        "dtree": DecisionTreeClassifier(random_state=RS),
        "Bagging": BaggingClassifier(random_state=RS),
        "Random forest": RandomForestClassifier(random_state=RS, n_jobs=1),
        "Adaboost": AdaBoostClassifier(random_state=RS),
        "GBM": GradientBoostingClassifier(random_state=RS),
        "Xgboost": XGBClassifier(
            random_state=RS,
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
        ),
    }


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
        result["roc_auc"] = np.nan
    result["confusion_matrix"] = confusion_matrix(y, pred).tolist()
    return result


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    model_df = engineer_features(raw)

    x_raw, feature_columns = encode_features(model_df)
    y = model_df[TARGET].values

    x_temp, x_test, y_temp, y_test = train_test_split(
        x_raw, y, test_size=0.15, stratify=y, random_state=RS
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.176470588, stratify=y_temp, random_state=RS
    )

    smote = SMOTE(random_state=RS)
    rus = RandomUnderSampler(random_state=RS)
    x_train_over, y_train_over = smote.fit_resample(x_train, y_train)
    x_train_under, y_train_under = rus.fit_resample(x_train, y_train)

    datasets = {
        "Original": (x_train, y_train),
        "Oversampled": (x_train_over, y_train_over),
        "Undersampled": (x_train_under, y_train_under),
    }

    comparison_rows = []
    best_val_recall = -1.0
    best_bundle: dict | None = None

    for sample_name, (x_tr, y_tr) in datasets.items():
        for model_name, estimator in get_models().items():
            model = estimator.__class__(**estimator.get_params())
            model.fit(x_tr, y_tr)
            train_m = eval_classifier(model, x_tr, y_tr)
            val_m = eval_classifier(model, x_val, y_val)
            comparison_rows.append(
                {
                    "Model Name": model_name,
                    "Sampling": sample_name,
                    "Recall_train": train_m["recall"],
                    "Recall_val": val_m["recall"],
                    "F1_val": val_m["f1"],
                    "ROC_AUC_val": val_m["roc_auc"],
                }
            )
            if val_m["recall"] > best_val_recall:
                best_val_recall = val_m["recall"]
                best_bundle = {
                    "model_name": model_name,
                    "sampling_strategy": sample_name,
                    "estimator_class": model.__class__,
                    "params": model.get_params(),
                    "x_train": x_tr,
                    "y_train": y_tr,
                }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["Recall_val", "F1_val"], ascending=False
    )
    print("\n=== Model comparison (validation recall) ===")
    print(comparison_df.head(12).to_string(index=False))

    assert best_bundle is not None
    tuned = best_bundle["estimator_class"](**best_bundle["params"])

    param_grids = {
        DecisionTreeClassifier: {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        },
        BaggingClassifier: {
            "n_estimators": [25, 50, 100],
            "max_samples": [0.6, 0.8, 1.0],
        },
        RandomForestClassifier: {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        },
        AdaBoostClassifier: {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
        },
        GradientBoostingClassifier: {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
        },
        XGBClassifier: {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }

    grid = param_grids.get(best_bundle["estimator_class"], {})
    scorer = metrics.make_scorer(recall_score)
    search = RandomizedSearchCV(
        estimator=tuned,
        param_distributions=grid,
        n_iter=12,
        scoring=scorer,
        cv=5,
        random_state=RS,
        n_jobs=-1,
    )
    search.fit(best_bundle["x_train"], best_bundle["y_train"])
    final_model = search.best_estimator_
    print(f"\nBest base: {best_bundle['model_name']} ({best_bundle['sampling_strategy']})")
    print(f"Tuned params: {search.best_params_}")

    metrics_train = eval_classifier(final_model, x_train, y_train)
    metrics_val = eval_classifier(final_model, x_val, y_val)
    metrics_test = eval_classifier(final_model, x_test, y_test)

    print("\n=== Test set ===")
    for k, v in metrics_test.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    preprocess_config = {
        "cat_impute": {col: model_df[col].astype(str).mode().iloc[0] for col in CAT_COLS},
        "num_impute": {
            col: float(model_df[col].median()) if col in model_df.columns else 0.0
            for col in NUM_COLS
        },
        "categorical_options": {
            col: sorted(model_df[col].astype(str).dropna().unique().tolist()) for col in CAT_COLS
        },
        "feature_columns": feature_columns,
        "raw_features": FEATURES,
        "target": TARGET,
    }

    artifact = {
        "model": final_model,
        "model_name": best_bundle["model_name"],
        "sampling_strategy": best_bundle["sampling_strategy"],
        "target": TARGET,
        "feature_columns": feature_columns,
        "raw_features": FEATURES,
        "best_params": search.best_params_,
        "metrics_train": {k: v for k, v in metrics_train.items() if k != "confusion_matrix"},
        "metrics_val": {k: v for k, v in metrics_val.items() if k != "confusion_matrix"},
        "metrics_test": {k: v for k, v in metrics_test.items() if k != "confusion_matrix"},
        "comparison_top10": comparison_df.head(10).to_dict(orient="records"),
    }

    joblib.dump(artifact, MODELS_DIR / "medicare_classifier.pkl")
    (MODELS_DIR / "preprocess_config.json").write_text(
        json.dumps(preprocess_config, indent=2), encoding="utf-8"
    )
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved {MODELS_DIR / 'medicare_classifier.pkl'}")


if __name__ == "__main__":
    main()
