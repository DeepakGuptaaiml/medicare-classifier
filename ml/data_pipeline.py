"""Data pipeline: load raw claims → feature engineering → encode → split → sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from ml.constants import CAT_COLS, FEATURES, NUM_COLS, RS, TARGET, pay_code_bucket

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "claims_data.csv"


@dataclass
class DataBundle:
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_val: pd.DataFrame
    y_val: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray]]
    feature_columns: list[str]
    preprocess_config: dict
    model_df: pd.DataFrame = field(repr=False)


class DataPipeline:
    def __init__(self, random_state: int = RS) -> None:
        self.random_state = random_state

    @staticmethod
    def _discover_aml_input_dir() -> Path | None:
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

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires pyarrow. On AML, select environment "
                "'medicare-train-env' (not AzureML-minimal) with pyarrow>=14.0."
            ) from exc

    @staticmethod
    def _load_path(path: Path) -> pd.DataFrame:
        if path.is_file():
            if path.suffix.lower() == ".parquet":
                print(f"Loading parquet file: {path}")
                return DataPipeline._read_parquet(path)
            print(f"Loading CSV file: {path}")
            return pd.read_csv(path)

        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            print(f"Loading {len(parquet_files)} parquet files from {path}")
            return pd.concat(
                [DataPipeline._read_parquet(f) for f in parquet_files],
                ignore_index=True,
            )

        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            print(f"Loading {len(csv_files)} CSV files from {path}")
            return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        raise FileNotFoundError(f"No parquet or CSV files found in {path}")

    @staticmethod
    def resolve_training_path(data_path: str | Path | None = None) -> Path:
        if data_path and os.path.exists(data_path):
            return Path(data_path)

        for env_name in ("TRAINING_DATA_URI", "TRAINING_DATA_PATH"):
            if env_value := os.getenv(env_name):
                env_path = Path(env_value)
                if env_path.exists():
                    return env_path

        if mount := DataPipeline._discover_aml_input_dir():
            return mount

        if DEFAULT_DATA_PATH.exists():
            return DEFAULT_DATA_PATH

        raise FileNotFoundError(
            "No training data found. Provide --data_path "
            "(Azure ML input binding, e.g. /mnt/azureml/inputs/training_data), "
            "set TRAINING_DATA_URI, or place claims_data.csv in data/."
        )

    @staticmethod
    def load_training_frame(data_path: str | Path | None = None) -> pd.DataFrame:
        resolved = DataPipeline.resolve_training_path(data_path)
        return DataPipeline._load_path(resolved)

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[TARGET] = pd.to_numeric(out["is_v1m_extracted"], errors="coerce").fillna(0).astype(int)

        for col in ["date_event", "date_open", "date_close", "clmnt_dob", "date_v1m_xmit"]:
            out[f"{col}_dt"] = pd.to_datetime(out[col], errors="coerce")

        out["date_v1m_xmit_flag"] = out["date_v1m_xmit_dt"].notna().astype(int)
        out["is_us_claimant"] = (out["clmnt_country"].astype(str) == "USA").astype(int)
        out["orm_threshold_met"] = (
            pd.to_numeric(out["paid_3"], errors="coerce").fillna(0) > 750
        ).astype(int)
        out["tpoc_threshold_met"] = (
            pd.to_numeric(out["paid_1"], errors="coerce").fillna(0) > 0
        ).astype(int)
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

    @staticmethod
    def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        frame = df[FEATURES].copy()
        for col in CAT_COLS:
            frame[col] = frame[col].astype(str).fillna("MISSING")
        encoded = pd.get_dummies(frame, columns=CAT_COLS, drop_first=True)
        return encoded, list(encoded.columns)

    @staticmethod
    def build_preprocess_config(model_df: pd.DataFrame, feature_columns: list[str]) -> dict:
        return {
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

    def fit_transform(self, raw_df: pd.DataFrame) -> DataBundle:
        model_df = self.engineer_features(raw_df)
        x_raw, feature_columns = self.encode_features(model_df)
        y = model_df[TARGET].values

        x_temp, x_test, y_temp, y_test = train_test_split(
            x_raw, y, test_size=0.15, stratify=y, random_state=self.random_state
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=0.176470588, stratify=y_temp, random_state=self.random_state
        )

        smote = SMOTE(random_state=self.random_state)
        rus = RandomUnderSampler(random_state=self.random_state)
        x_train_over, y_train_over = smote.fit_resample(x_train, y_train)
        x_train_under, y_train_under = rus.fit_resample(x_train, y_train)

        datasets = {
            "Original": (x_train, y_train),
            "Oversampled": (x_train_over, y_train_over),
            "Undersampled": (x_train_under, y_train_under),
        }

        preprocess_config = self.build_preprocess_config(model_df, feature_columns)
        print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

        return DataBundle(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            datasets=datasets,
            feature_columns=feature_columns,
            preprocess_config=preprocess_config,
            model_df=model_df,
        )

    def fit_transform_path(self, data_path: str | Path | None = None) -> DataBundle:
        raw_df = self.load_training_frame(data_path)
        print(f"Loaded {len(raw_df):,} training records")
        return self.fit_transform(raw_df)
