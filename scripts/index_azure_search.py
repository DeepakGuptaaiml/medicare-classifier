"""
Upload Medicare policy document chunks to Azure AI Search.

Keyword search works immediately on `content`. Populate `embedding` later when
Azure OpenAI text-embedding-3-small is deployed (1536 dimensions).

Usage:
  export AZURE_SEARCH_ENDPOINT=https://claims-ai-search-dg.search.windows.net
  export AZURE_SEARCH_KEY=<admin-key>
  export AZURE_SEARCH_INDEX=medicare-policy   # optional
  python scripts/index_azure_search.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import requests

API_VERSION = "2024-07-01"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
DOCS_PATH = Path(__file__).resolve().parent.parent / "agent" / "docs"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def doc_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{index}:{content[:64]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-{index}-{digest}"


def load_chunks() -> list[dict]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Policy docs not found: {DOCS_PATH}")

    records: list[dict] = []
    for path in sorted(DOCS_PATH.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, content in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            records.append({
                "id": doc_id(path.name, i, content),
                "content": content,
                "source": path.name,
            })
    return records


def upload_batches(endpoint: str, key: str, index_name: str, records: list[dict]) -> None:
    url = f"{endpoint}/indexes/{index_name}/docs/index?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": key}

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        payload = {"value": [{"@search.action": "upload", **doc} for doc in batch]}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        results = response.json().get("value", [])
        failed = [r for r in results if not r.get("status")]
        if failed:
            raise RuntimeError(f"Upload failed for batch starting at {start}: {failed}")
        print(f"Uploaded {start + len(batch)}/{len(records)} documents")


def main() -> int:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
    key = os.getenv("AZURE_SEARCH_KEY", "")
    index_name = os.getenv("AZURE_SEARCH_INDEX", "medicare-policy")

    if not endpoint or not key:
        print(
            "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY environment variables.",
            file=sys.stderr,
        )
        return 1

    records = load_chunks()
    if not records:
        print("No chunks to upload.", file=sys.stderr)
        return 1

    print(f"Indexing {len(records)} chunks from {DOCS_PATH} → {index_name}")
    upload_batches(endpoint, key, index_name, records)
    print(f"Done. Search: {endpoint}/indexes/{index_name}/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
