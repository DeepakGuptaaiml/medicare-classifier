"""
Medicare Policy RAG Agent — LangChain RetrievalQA over MMSEA / MCI / MCRC documents.

Flow: load docs → chunk → embed (HF Inference API) → ChromaDB → retrieve top-k → LLM answer.

NOTE: In production/enterprise this would use Azure OpenAI Service (same GPT-4o model)
for HIPAA compliance and to keep PHI within the Azure tenant. Switch by replacing
HuggingFaceEndpoint with AzureChatOpenAI — all LangChain chain code remains identical.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import (
    CHROMA_DB_PATH,
    get_hf_api_token,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DOCS_PATH,
    RAG_EMBEDDING_MODEL,
    RAG_MODEL_ID,
    RAG_TOP_K,
)

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
        import chromadb
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEndpoint

        hf_token = get_hf_api_token()
        if not hf_token:
            raise RAGConfigurationError(
                "HF_API_TOKEN not set. Export HF_API_TOKEN with a Hugging Face "
                "Inference API token. See .env.example."
            )

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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        # Using HuggingFace Inference API for embeddings — no local model download needed.
        # Keeps Docker image lightweight (<500MB). In production: use Azure OpenAI embeddings.
        self._embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=RAG_EMBEDDING_MODEL,
            api_key=hf_token,
        )

        # Lightweight ChromaDB — persistent client (Azure volume at CHROMA_DB_PATH)
        chroma_path = os.getenv("CHROMA_DB_PATH", str(CHROMA_DB_PATH))
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_path)

        collection_exists = False
        if "medicare_policy" in [c.name for c in chroma_client.list_collections()]:
            collection_exists = chroma_client.get_collection("medicare_policy").count() > 0

        if collection_exists:
            self._vectorstore = Chroma(
                client=chroma_client,
                collection_name="medicare_policy",
                embedding_function=self._embeddings,
            )
        else:
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                client=chroma_client,
                collection_name="medicare_policy",
            )

        # LLM via Hugging Face Inference API (no local model weights)
        self._llm = HuggingFaceEndpoint(
            repo_id=RAG_MODEL_ID,
            huggingfacehub_api_token=hf_token,
            temperature=0.1,
            max_new_tokens=512,
        )

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
        """Answer a policy question using retrieved context + LLM generation."""
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
