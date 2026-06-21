"""
Medicare Policy RAG Agent — LangChain RetrievalQA over MMSEA / MCI / MCRC documents.

Flow: load docs → chunk → embed (HF Inference API) → ChromaDB → retrieve top-k → LLM answer.

NOTE: In production/enterprise this would use Azure OpenAI Service (same GPT-4o model)
for HIPAA compliance and to keep PHI within the Azure tenant. Switch by replacing
HuggingFaceEndpoint with AzureChatOpenAI — all LangChain chain code remains identical.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import (
    CHROMA_DB_PATH,
    HF_API_TOKEN,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DOCS_PATH,
    RAG_EMBEDDING_MODEL,
    RAG_MODEL_ID,
    RAG_TOP_K,
)

# Singleton instance — lazy-loaded on first /ask request
_agent_instance: "MedicareRAGAgent | None" = None
_agent_lock = threading.Lock()

RAG_PROMPT_TEMPLATE = """
You are a Medicare claims policy expert for MMSEA Section 111
reporting. Answer questions based ONLY on the provided context.
If the answer is not in the context, say
"I cannot find this in the policy documents."
Always cite which document section your answer comes from.

Context: {context}
Question: {question}
Answer:
"""


class RAGConfigurationError(Exception):
    """Raised when HF_API_TOKEN or policy documents are missing."""


class MedicareRAGAgent:
    """RAG agent over Medicare policy reference documents."""

    def __init__(self) -> None:
        # Lazy imports — keeps FastAPI startup fast and avoids heavy deps until first /ask
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEndpoint

        # Step 1: Validate Hugging Face API token (required for embeddings + LLM)
        if not HF_API_TOKEN:
            raise RAGConfigurationError(
                "HF_API_TOKEN not set. Export HF_API_TOKEN with a Hugging Face "
                "Inference API token. See .env.example."
            )

        # Step 2: Load all policy documents from agent/docs/
        if not RAG_DOCS_PATH.exists():
            raise RAGConfigurationError(f"Policy documents directory not found: {RAG_DOCS_PATH}")

        txt_files = list(RAG_DOCS_PATH.glob("*.txt"))
        if not txt_files:
            raise RAGConfigurationError("Policy documents not loaded — no .txt files in agent/docs/")

        loader = DirectoryLoader(
            str(RAG_DOCS_PATH),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        self._documents_loaded = len(documents)

        # Step 3: Split documents into overlapping chunks for retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        # Step 4: Create embeddings via Hugging Face Inference API
        # In production use Azure OpenAI embeddings for HIPAA compliance
        self._embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=RAG_EMBEDDING_MODEL,
            api_key=HF_API_TOKEN,
        )

        # Step 5: Persist vectors in ChromaDB (reuses existing index if present)
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        chroma_has_data = any(CHROMA_DB_PATH.iterdir())

        if chroma_has_data:
            self._vectorstore = Chroma(
                collection_name="medicare_policy",
                embedding_function=self._embeddings,
                persist_directory=str(CHROMA_DB_PATH),
            )
        else:
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                collection_name="medicare_policy",
                persist_directory=str(CHROMA_DB_PATH),
            )

        # Step 6: Initialize LLM via Hugging Face Inference Endpoint
        # In production use Azure OpenAI Service (AzureChatOpenAI) for HIPAA compliance
        self._llm = HuggingFaceEndpoint(
            repo_id=RAG_MODEL_ID,
            huggingfacehub_api_token=HF_API_TOKEN,
            temperature=0.1,
            max_new_tokens=512,
        )

        # Step 7: Build RetrievalQA chain — "stuff" packs retrieved chunks into prompt
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        self._default_top_k = RAG_TOP_K

    def get_status(self) -> dict:
        """Return readiness flags without re-initializing the chain."""
        return {
            "documents_loaded": self._documents_loaded,
            "vector_store_ready": self._vectorstore is not None,
            "llm_ready": self._llm is not None,
        }

    def ask(self, question: str, max_chunks: int | None = None) -> dict:
        """
        Answer a policy question using retrieved context + LLM generation.

        Returns dict with question, answer, sources, and chunks_used.
        """
        question = (question or "").strip()
        if not question:
            return {
                "question": question,
                "answer": "Please provide a question",
                "sources": [],
                "chunks_used": [],
            }

        top_k = max_chunks or self._default_top_k

        try:
            self._qa_chain.retriever.search_kwargs["k"] = top_k
            result = self._qa_chain.invoke({"query": question})
            answer = result.get("result", "").strip()
            source_docs = result.get("source_documents", [])

            sources: list[str] = []
            chunks_used: list[str] = []
            for doc in source_docs:
                source_name = Path(doc.metadata.get("source", "unknown")).name
                if source_name not in sources:
                    sources.append(source_name)
                chunk_text = doc.page_content.strip()
                if chunk_text and chunk_text not in chunks_used:
                    chunks_used.append(chunk_text)

            return {
                "question": question,
                "answer": answer or "I cannot find this in the policy documents.",
                "sources": sources,
                "chunks_used": chunks_used,
            }

        except Exception as exc:
            return {
                "question": question,
                "answer": (
                    f"The policy Q&A service is temporarily unavailable: {exc}. "
                    "Please verify HF_API_TOKEN and Hugging Face Inference API status."
                ),
                "sources": [],
                "chunks_used": [],
            }


def get_rag_agent() -> MedicareRAGAgent:
    """Lazy singleton — initialize RAG pipeline once, reuse across requests."""
    global _agent_instance
    if _agent_instance is None:
        with _agent_lock:
            if _agent_instance is None:
                _agent_instance = MedicareRAGAgent()
    return _agent_instance


def reset_rag_agent() -> None:
    """Reset singleton (used in tests)."""
    global _agent_instance
    with _agent_lock:
        _agent_instance = None
