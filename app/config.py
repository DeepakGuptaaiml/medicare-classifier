from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "medicare_classifier.pkl"
PREPROCESS_CONFIG_PATH = BASE_DIR / "models" / "preprocess_config.json"
SAMPLE_CLAIMS_PATH = Path(__file__).resolve().parent / "sample_claims.json"
