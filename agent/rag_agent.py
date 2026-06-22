"""
Medicare Policy RAG Agent — LangChain RetrievalQA over MMSEA / MCI / MCRC documents.

Flow: load docs → chunk → TF-IDF retrieve (local, no API) → LLM answer (optional HF).

NOTE: In production/enterprise this would use Azure OpenAI Service for HIPAA compliance.
TF-IDF retrieval works offline; LLM requires outbound access at query time only.
"""

from __future__ import annotations

from pathlib import Path

from app.config import (
    get_hf_api_token,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DOCS_PATH,
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
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_community.retrievers import TFIDFRetriever
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
        self.chunks = splitter.split_documents(documents)

        # TF-IDF retrieval — fully local, no embeddings API or outbound calls at startup
        self.retriever = TFIDFRetriever.from_documents(self.chunks, k=RAG_TOP_K)
        self._default_top_k = RAG_TOP_K

        # LLM via Hugging Face Inference API (outbound only at query time)
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

        self.chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def get_status(self) -> dict:
        """Return readiness flags without re-initializing the chain."""
        return {
            "documents_loaded": self._documents_loaded,
            "vector_store_ready": self.retriever is not None,
            "llm_ready": self._llm is not None,
        }

    def ask(self, question: str, max_chunks: int | None = None) -> dict:
        """Answer a policy question using TF-IDF retrieval + optional LLM generation."""
        question = (question or "").strip()
        if not question:
            return {
                "question": question,
                "answer": "Please provide a question",
                "sources": [],
                "chunks_used": [],
            }

        top_k = max_chunks or self._default_top_k
        self.retriever.k = top_k

        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(
            {
                Path(d.metadata.get("source", "policy_docs")).name
                for d in docs
            }
        )

        try:
            result = self.chain.invoke({"query": question})
            answer = result.get("result", context)
        except Exception:
            answer = f"Based on policy documents:\n\n{context}"

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": [d.page_content[:200] for d in docs],
        }
