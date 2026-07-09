"""Training pipeline: model comparison, recall tuning, artifact export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.constants import FEATURES, RS, TARGET
from ml.data_pipeline import DataBundle
from ml.metrics import eval_classifier

PARAM_GRIDS = {
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


@dataclass
class TrainingResult:
    model: object
    model_name: str
    sampling_strategy: str
    feature_columns: list[str]
    best_params: dict
    metrics_train: dict
    metrics_val: dict
    metrics_test: dict
    comparison_df: pd.DataFrame
    preprocess_config: dict


class TrainingPipeline:
    def __init__(self, random_state: int = RS) -> None:
        self.random_state = random_state

    @staticmethod
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

    def compare_models(self, bundle: DataBundle) -> tuple[pd.DataFrame, dict]:
        comparison_rows = []
        best_val_recall = -1.0
        best_bundle: dict | None = None

        for sample_name, (x_tr, y_tr) in bundle.datasets.items():
            for model_name, estimator in self.get_models().items():
                model = estimator.__class__(**estimator.get_params())
                model.fit(x_tr, y_tr)
                train_m = eval_classifier(model, x_tr, y_tr)
                val_m = eval_classifier(model, bundle.x_val, bundle.y_val)
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

        if best_bundle is None:
            raise RuntimeError("Model comparison did not select a winner")

        return comparison_df, best_bundle

    def tune_best(self, best_bundle: dict) -> tuple[object, dict]:
        tuned = best_bundle["estimator_class"](**best_bundle["params"])
        grid = PARAM_GRIDS.get(best_bundle["estimator_class"], {})
        scorer = metrics.make_scorer(recall_score)
        search = RandomizedSearchCV(
            estimator=tuned,
            param_distributions=grid,
            n_iter=12,
            scoring=scorer,
            cv=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(best_bundle["x_train"], best_bundle["y_train"])
        print(f"\nBest base: {best_bundle['model_name']} ({best_bundle['sampling_strategy']})")
        print(f"Tuned params: {search.best_params_}")
        return search.best_estimator_, search.best_params_

    def fit(self, bundle: DataBundle) -> TrainingResult:
        comparison_df, best_bundle = self.compare_models(bundle)
        final_model, best_params = self.tune_best(best_bundle)

        metrics_train = eval_classifier(final_model, bundle.x_train, bundle.y_train)
        metrics_val = eval_classifier(final_model, bundle.x_val, bundle.y_val)
        metrics_test = eval_classifier(final_model, bundle.x_test, bundle.y_test)

        print("\n=== Test set ===")
        for key, value in metrics_test.items():
            if key != "confusion_matrix":
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        return TrainingResult(
            model=final_model,
            model_name=best_bundle["model_name"],
            sampling_strategy=best_bundle["sampling_strategy"],
            feature_columns=bundle.feature_columns,
            best_params=best_params,
            metrics_train={k: v for k, v in metrics_train.items() if k != "confusion_matrix"},
            metrics_val={k: v for k, v in metrics_val.items() if k != "confusion_matrix"},
            metrics_test={k: v for k, v in metrics_test.items() if k != "confusion_matrix"},
            comparison_df=comparison_df,
            preprocess_config=bundle.preprocess_config,
        )

    def export(self, result: TrainingResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": result.model,
            "model_name": result.model_name,
            "sampling_strategy": result.sampling_strategy,
            "target": TARGET,
            "feature_columns": result.feature_columns,
            "raw_features": FEATURES,
            "best_params": result.best_params,
            "metrics_train": result.metrics_train,
            "metrics_val": result.metrics_val,
            "metrics_test": result.metrics_test,
            "comparison_top10": result.comparison_df.head(10).to_dict(orient="records"),
        }

        model_path = output_dir / "medicare_classifier.pkl"
        config_path = output_dir / "preprocess_config.json"
        comparison_path = output_dir / "model_comparison.csv"

        joblib.dump(artifact, model_path)
        config_path.write_text(json.dumps(result.preprocess_config, indent=2), encoding="utf-8")
        result.comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved {model_path}")
