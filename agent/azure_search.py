"""Azure AI Search keyword retrieval for Medicare policy RAG."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import requests

API_VERSION = "2024-07-01"


class AzureSearchClient:
    """Simple REST client for keyword search on the medicare-policy index."""

    def __init__(self, endpoint: str, key: str, index_name: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.index_name = index_name

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version={API_VERSION}"
        payload = {
            "search": query,
            "top": top_k,
            "select": "content,source",
        }
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "api-key": self.key},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("value", [])


class AzureSearchRetriever(BaseRetriever):
    """LangChain retriever backed by Azure AI Search keyword search."""

    client: AzureSearchClient = Field(exclude=True)
    k: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str) -> list[Document]:
        hits = self.client.search(query, top_k=self.k)
        return [
            Document(
                page_content=hit.get("content", ""),
                metadata={"source": hit.get("source", "medicare-policy")},
            )
            for hit in hits
            if hit.get("content")
        ]
