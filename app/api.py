"""
app/api.py
----------
Production-facing HTTP API for VeriLex.

This wraps the existing ingestion and retrieval pipeline in a small FastAPI
service so the project can be run as a usable application instead of only a
CLI. The endpoints deliberately stay thin and call the same core pipeline
functions that `main.py` uses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import DEFAULT_PERSIST_DIR
from main import get_pdfs_from_docs, run_ingestion_pipeline, run_query
from pipeline.vector_store import load_vector_store, retrieve_chunks, generate_final_answer
from utils.export_utils import export_chunks_to_json


app = FastAPI(
    title="VeriLex API",
    description="HTTP wrapper around the VeriLex multimodal RAG pipeline",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    """Request payload for ingestion.

    If `pdf_path` is omitted, the service ingests every PDF found in `docs/`.
    """

    pdf_path: Optional[str] = Field(default=None, description="Single PDF to ingest")
    persist_dir: str = Field(default=DEFAULT_PERSIST_DIR, description="ChromaDB persist directory")


class QueryRequest(BaseModel):
    """Request payload for question answering."""

    question: str = Field(..., min_length=1)
    persist_dir: str = Field(default=DEFAULT_PERSIST_DIR, description="ChromaDB persist directory")


class ChatTurnRequest(BaseModel):
    """Stateless chat turn payload.

    Note: this service version does not persist conversation state yet, so
    callers may optionally send a small `history` list which will be forwarded
    to the answer generator.
    """

    question: str = Field(..., min_length=1)
    persist_dir: str = Field(default=DEFAULT_PERSIST_DIR)
    history: list[dict] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    """Basic health probe for deployments and load balancers."""
    # Keep health checks cheap and deterministic so orchestration tools can
    # probe the container without touching the vector store or any LLMs.
    return {
        "status": "ok",
        "persist_dir": DEFAULT_PERSIST_DIR,
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    """Run the ingestion pipeline.

    This endpoint is intentionally synchronous and simple. For large corpora,
    move this work to a background worker queue in a later iteration.
    """
    try:
        if request.pdf_path:
            # Single-file mode is useful for debugging or re-ingesting a known
            # document without scanning the whole `docs/` directory.
            pdf_path = Path(request.pdf_path)
            if not pdf_path.exists():
                raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")
            db = run_ingestion_pipeline(str(pdf_path), persist_dir=request.persist_dir)
            return {"status": "ok", "mode": "single_pdf", "pdf_path": str(pdf_path), "persist_dir": request.persist_dir}

        # Default mode mirrors the CLI: scan the local docs folder and ingest
        # every PDF we find there.
        pdfs = get_pdfs_from_docs()
        last_db = None
        for pdf in pdfs:
            # Each PDF is processed through the same pipeline used by the CLI.
            last_db = run_ingestion_pipeline(str(pdf), persist_dir=request.persist_dir)

        return {
            "status": "ok",
            "mode": "docs_folder",
            "processed_pdfs": len(pdfs),
            "persist_dir": request.persist_dir,
            "collection_ready": last_db is not None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    try:
        # This is a thin wrapper around the existing `run_query` helper so the
        # HTTP API and CLI stay aligned and do not diverge in behavior.
        answer = run_query(request.question, persist_dir=request.persist_dir)
        return {"status": "ok", "question": request.question, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.post("/chat/turn")
def chat_turn(request: ChatTurnRequest) -> dict:
    """Single chat turn with optional caller-provided history.

    The current project has in-memory history in the CLI only. This HTTP
    endpoint exposes a stateless version so the API can already be used in a
    production deployment while session persistence is added later.
    """
    try:
        # Load the persisted Chroma collection once per request. In a more
        # advanced version this could be cached or managed by a worker pool.
        db = load_vector_store(request.persist_dir)

        # Retrieve the most relevant chunks for this question.
        chunks = retrieve_chunks(db, request.question)

        # Generate the final answer using the LLM adapter. Optional `history`
        # is passed through so callers can simulate a multi-turn conversation.
        answer = generate_final_answer(chunks, request.question, history=request.history)

        # Export retrieved context for debugging and traceability.
        export_chunks_to_json(chunks, "rag_results.json")
        return {
            "status": "ok",
            "question": request.question,
            "answer": answer,
            "retrieved_chunks": len(chunks),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat turn failed: {exc}")
