#!/usr/bin/env bash
# Azure ML command-job entrypoint.
# Use $[[inputs.*]] / $[[outputs.*]] in the Studio Command field to call this script,
# or run directly when env vars are already set (local / CI).
#
# Studio Command (recommended):
#   bash scripts/aml_run_training.sh $[[inputs.training_data]] $[[outputs.model_output]]
#
# Inputs/Outputs in Studio:
#   training_data  — uri_folder — azureml://datastores/claims_medicare_training/paths/latest/
#   model_output   — uri_folder — (default output path)

set -euo pipefail

TRAINING_DATA_URI="${1:-${TRAINING_DATA_URI:-}}"
MODEL_OUTPUT_DIR="${2:-${MODEL_OUTPUT_DIR:-${AZUREML_OUTPUT_MODEL_OUTPUT:-}}}"

if [[ -z "$TRAINING_DATA_URI" ]]; then
  echo "ERROR: training data path not set. Pass arg1 or TRAINING_DATA_URI." >&2
  exit 1
fi

export TRAINING_DATA_URI
export MODELS_DIR="${MODELS_DIR:-/tmp/models}"
export MODEL_OUTPUT_DIR

echo "TRAINING_DATA_URI=$TRAINING_DATA_URI"
echo "MODELS_DIR=$MODELS_DIR"
echo "MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR:-<not set — artifacts stay in MODELS_DIR>}"

python train_medicare_classifier.py

if [[ -n "$MODEL_OUTPUT_DIR" ]]; then
  mkdir -p "$MODEL_OUTPUT_DIR"
  for f in medicare_classifier.pkl preprocess_config.json model_comparison.csv; do
    if [[ -f "$MODELS_DIR/$f" ]]; then
      cp "$MODELS_DIR/$f" "$MODEL_OUTPUT_DIR/"
      echo "Copied $f -> $MODEL_OUTPUT_DIR/"
    fi
  done
fi
