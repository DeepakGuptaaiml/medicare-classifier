import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "medicare_classifier.pkl"
PREPROCESS_CONFIG_PATH = BASE_DIR / "models" / "preprocess_config.json"
SAMPLE_CLAIMS_PATH = Path(__file__).resolve().parent / "sample_claims.json"

# RAG agent configuration (HF_API_TOKEN from environment only — never hardcode)
def get_hf_api_token() -> str:
    """
    Read HF_API_TOKEN at runtime — not at import time.
    This ensures Azure secret injection works correctly.
    Checks multiple env var names for compatibility.
    """
    token = (
        os.environ.get("HF_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    )
    return token.strip()


RAG_DOCS_PATH: Path = BASE_DIR / "agent" / "docs"
CHROMA_DB_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "agent" / "chroma_db")))
RAG_CHUNK_SIZE: int = 500
RAG_CHUNK_OVERLAP: int = 50
RAG_TOP_K: int = 3
RAG_MODEL_ID: str = "HuggingFaceH4/zephyr-7b-beta"
RAG_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
