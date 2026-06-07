"""Smoke tests for the FastAPI service.

These tests avoid hitting real LLMs or the vector DB by monkeypatching the
core pipeline functions that the API imports from `main.py` and `pipeline/`.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_query_endpoint_with_monkeypatched_pipeline(monkeypatch):
    # Patch the imported function in the API module so the test stays offline.
    # The goal here is to validate the HTTP wiring, not the retrieval or LLM
    # pipeline, which is covered elsewhere.
    monkeypatch.setattr("app.api.run_query", lambda question, persist_dir: f"answer:{question}:{persist_dir}")
    response = client.post("/query", json={"question": "hello", "persist_dir": "db/chroma_db"})
    assert response.status_code == 200
    assert response.json()["answer"] == "answer:hello:db/chroma_db"


def test_chat_turn_endpoint_with_monkeypatched_pipeline(monkeypatch):
    class DummyDB:
        pass

    # Patch each step the endpoint uses so we can verify the request/response
    # shape without depending on a real vector store or external API call.
    monkeypatch.setattr("app.api.load_vector_store", lambda persist_dir: DummyDB())
    monkeypatch.setattr("app.api.retrieve_chunks", lambda db, question: ["chunk-1", "chunk-2"])
    monkeypatch.setattr("app.api.generate_final_answer", lambda chunks, question, history=None: "final-answer")
    monkeypatch.setattr("app.api.export_chunks_to_json", lambda chunks, filename: chunks)

    response = client.post(
        "/chat/turn",
        json={"question": "what is this?", "persist_dir": "db/chroma_db", "history": []},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "final-answer"
