#!/usr/bin/env python3
"""Train Medicare Classifier and export medicare_classifier.pkl + preprocess_config.json.

Orchestrates ml.data_pipeline → ml.training_pipeline (6 models × 3 sampling, recall tuning).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from ml.data_pipeline import DataPipeline
from ml.training_pipeline import TrainingPipeline

BASE_DIR = Path(__file__).resolve().parent


def resolve_models_dir() -> Path:
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


def resolve_model_output_dir(model_output: str | Path | None) -> Path:
    if model_output:
        path = Path(model_output)
        path.mkdir(parents=True, exist_ok=True)
        return path

    for env_name in ("MODEL_OUTPUT_DIR", "AZUREML_OUTPUT_MODEL_OUTPUT"):
        if env_value := os.getenv(env_name):
            path = Path(env_value)
            path.mkdir(parents=True, exist_ok=True)
            return path

    return resolve_models_dir()


def _publish_artifacts(source_dir: Path) -> None:
    output_dir = os.getenv("MODEL_OUTPUT_DIR") or os.getenv("AZUREML_OUTPUT_MODEL_OUTPUT")
    if not output_dir:
        return

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("medicare_classifier.pkl", "preprocess_config.json", "model_comparison.csv"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
            print(f"Published {dest / name}")


def train(data_path: str | Path | None, output_dir: Path) -> None:
    print(f"Writing artifacts to {output_dir}")
    bundle = DataPipeline().fit_transform_path(data_path)
    result = TrainingPipeline().fit(bundle)
    TrainingPipeline().export(result, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Medicare Classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Training data file or folder (AML mount, parquet, or CSV)",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default=None,
        help="Output directory for medicare_classifier.pkl and preprocess_config.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = args.data_path or os.getenv("TRAINING_DATA_URI") or os.getenv("TRAINING_DATA_PATH")

    explicit_output = args.model_output is not None
    output_dir = resolve_model_output_dir(args.model_output)

    if explicit_output and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train(data_path, output_dir)
        if not explicit_output:
            _publish_artifacts(output_dir)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
