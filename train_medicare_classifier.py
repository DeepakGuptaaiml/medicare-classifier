#!/usr/bin/env python3
"""Train Medicare Classifier and export medicare_classifier.pkl + preprocess_config.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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
MODELS_DIR = BASE_DIR / "models"  # default for local runs


def resolve_models_dir() -> Path:
    """
    Pick a writable models directory.

    AML code uploads are often read-only; if repo ``models/`` was included in
    the upload bundle, joblib.dump fails with PermissionError.
    """
    candidates: list[Path] = []
    if env_dir := os.getenv("MODELS_DIR"):
        candidates.append(Path(env_dir))
    candidates.extend([Path("/tmp/models"), BASE_DIR / "models"])

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_bytes(b"x")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("No writable models directory found")


def _discover_aml_input_dir() -> Path | None:
    """Return AML-mounted input folder when job declares a datastore input."""
    inputs_root = Path("/mnt/azureml/inputs")
    if not inputs_root.is_dir():
        return None

    preferred = inputs_root / "training_data"
    if preferred.is_dir():
        return preferred

    for child in sorted(inputs_root.iterdir()):
        if child.is_dir() and (any(child.glob("*.parquet")) or any(child.glob("*.csv"))):
            return child
    return None


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires pyarrow. On AML, select environment "
            "'medicare-train-env' (not AzureML-minimal) and rebuild after adding "
            "pyarrow>=14.0 to aml-train-env.yml."
        ) from exc


def load_training_frame(data_path: str | None = None) -> pd.DataFrame:
    """
    Load training data from:
    1. Azure ML input binding path (production)
    2. Local CSV fallback (development)
    """
    if data_path and os.path.exists(data_path):
        path = Path(data_path)
        if path.is_file():
            if path.suffix.lower() == ".parquet":
                print(f"Loading parquet file: {path}")
                return _read_parquet(path)
            print(f"Loading CSV file: {path}")
            return pd.read_csv(path)

        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            print(f"Loading {len(parquet_files)} parquet files from {data_path}")
            return pd.concat([_read_parquet(f) for f in parquet_files], ignore_index=True)

        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            print(f"Loading {len(csv_files)} CSV files from {data_path}")
            return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    local_csv = BASE_DIR / "data" / "claims_data.csv"
    if local_csv.exists():
        print(f"Loading local CSV: {local_csv}")
        return pd.read_csv(local_csv)

    raise FileNotFoundError(
        "No training data found. Provide --data_path argument "
        "(Azure ML input binding, e.g. /mnt/azureml/inputs/training_data) "
        "or place claims_data.csv in data/ folder."
    )


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
    print("=== DEBUG ===")
    print(f"sys.argv: {sys.argv}")
    print(f"Current dir: {os.getcwd()}")
    print(f"Files in current dir: {os.listdir('.')}")
    print(f"MODELS_DIR env: {os.getenv('MODELS_DIR')}")
    print(f"Python executable: {sys.executable}")
    try:
        import pyarrow

        print(f"pyarrow version: {pyarrow.__version__}")
    except ImportError:
        print("pyarrow: NOT INSTALLED — switch job to medicare-train-env")

    data_path = None
    for arg in sys.argv:
        if "--data_path" in arg:
            data_path = arg.split("=")[-1]
    if data_path:
        print(f"data_path: {data_path}")
        print(f"data_path exists: {os.path.exists(data_path)}")
        if os.path.exists(data_path):
            print(f"Files in data_path: {os.listdir(data_path)}")
    print("=== END DEBUG ===")

    parser = argparse.ArgumentParser(description="Train Medicare Classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (Azure ML input binding or local path)",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default=None,
        help="Path to write model artifacts",
    )
    args = parser.parse_args()

    data_path = args.data_path or os.getenv("TRAINING_DATA_URI") or os.getenv("TRAINING_DATA_PATH")
    if not data_path and (mount := _discover_aml_input_dir()):
        data_path = str(mount)

    raw = load_training_frame(data_path=data_path)
    print(f"Loaded {len(raw):,} training records")

    if args.model_output:
        models_dir = Path(args.model_output)
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        models_dir = resolve_models_dir()
    print(f"Writing artifacts to {models_dir}")
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

    joblib.dump(artifact, models_dir / "medicare_classifier.pkl")
    (models_dir / "preprocess_config.json").write_text(
        json.dumps(preprocess_config, indent=2), encoding="utf-8"
    )
    comparison_df.to_csv(models_dir / "model_comparison.csv", index=False)
    print(f"\nSaved {models_dir / 'medicare_classifier.pkl'}")
    if args.model_output:
        return
    _publish_artifacts(models_dir)


def _publish_artifacts(models_dir: Path) -> None:
    """
    Copy trained artifacts to an AML job output folder when configured.

    Set output named ``model_output`` on the command job — AML exposes
    ``AZUREML_OUTPUT_MODEL_OUTPUT``. Or set MODEL_OUTPUT_DIR manually.
    """
    output_dir = os.getenv("MODEL_OUTPUT_DIR") or os.getenv("AZUREML_OUTPUT_MODEL_OUTPUT")
    if not output_dir:
        return

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("medicare_classifier.pkl", "preprocess_config.json", "model_comparison.csv"):
        src = models_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
            print(f"Published {dest / name}")


if __name__ == "__main__":
    main()
