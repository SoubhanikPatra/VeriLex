"""
pipeline/vector_store.py
------------------------
Step 4 — Embed LangChain Documents and persist them in ChromaDB.
Also provides query utilities for answer generation at retrieval time.
"""

import json
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from utils.llm import get_llm_client

from config.settings import (
    CHAT_MODEL,
    CHROMA_SPACE,
    DEFAULT_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_K,
)

# ── Ingestion ─────────────────────────────────────────────────────────────────

def create_vector_store(
    documents: List[Document],
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Embed documents and persist them in a ChromaDB collection.

    Args:
        documents:          LangChain Documents from the summariser.
        persist_directory:  Directory where ChromaDB will write its files.

    Returns:
        A ready-to-query Chroma instance.
    """
    # Create an embeddings client and encode all documents
    print("🔮 Creating embeddings and storing in ChromaDB...")

    # HuggingFace embeddings convert text to fixed-length vectors for nearest-neighbour search
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("--- Creating vector store ---")
    # Build a Chroma collection from the provided documents and persist to disk
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": CHROMA_SPACE},
    )
    print("--- Finished creating vector store ---")

    print(f"✅ Vector store saved to {persist_directory}")
    return vectorstore


def load_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """
    Load an existing ChromaDB collection from disk.

    Args:
        persist_directory: Directory previously used by create_vector_store.

    Returns:
        A ready-to-query Chroma instance.
    """
    # Recreate the embeddings object to ensure the loaded collection can compute similarities
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": CHROMA_SPACE},
    )


# ── Retrieval + Answer Generation ─────────────────────────────────────────────

def retrieve_chunks(db: Chroma, query: str, k: int = RETRIEVER_K) -> List[Document]:
    """Return the top-k most relevant documents for a query."""
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def generate_final_answer(
    chunks: List[Document], query: str, history: Optional[List[dict]] = None
) -> str:
    """
    Build a multimodal prompt from retrieved chunks (text + tables + images)
    and ask the LLM for a grounded answer. Optionally include prior
    conversation turns (history) so the model can produce multi-turn,
    context-aware answers.

    Args:
        chunks: Documents returned by retrieve_chunks.
        query:  The user's current question.
        history: Optional list of dicts representing past turns. Each dict
                 should contain keys like 'user' and 'assistant' (strings).

    Returns:
        Model response as a plain string.
    """
    try:
        # Use the adapter-backed client which implements retries, backoff,
        # timeouts and fallback logic. This keeps the answer-generation code
        # independent of a particular provider and centralises resilience.
        llm = get_llm_client()

        # Start building the prompt. If history is provided, prepend a short
        # conversation transcript so the model can resolve multi-turn context.
        prompt_text = ""
        if history:
            prompt_text += "Conversation history:\n"
            for turn in history:
                # Support different key names for flexibility
                user_turn = turn.get("user") or turn.get("question") or turn.get("q")
                assistant_turn = (
                    turn.get("assistant") or turn.get("answer") or turn.get("a")
                )
                if user_turn:
                    prompt_text += f"USER: {user_turn}\n"
                if assistant_turn:
                    prompt_text += f"ASSISTANT: {assistant_turn}\n"
            prompt_text += "\n---\n\n"

        prompt_text += (
            f"Based on the following documents, please answer this question: {query}\n\n"
            "CONTENT TO ANALYZE:\n"
        )

        image_count = 0

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"

            # The pipeline stores the original raw text, tables and image base64 in
            # `original_content` so we can reconstruct a multimodal prompt here.
            if "original_content" in chunk.metadata:
                original = json.loads(chunk.metadata["original_content"])

                raw_text = original.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        # Tables are included verbatim to preserve cell structure (as HTML)
                        prompt_text += f"Table {j + 1}:\n{table}\n\n"

                images = original.get("images_base64", [])
                image_count += len(images)
                if images:
                    # We mention images as contextual markers because the LLM cannot
                    # consume raw binary here; downstream answer generation should
                    # note that visual content exists and may influence the answer.
                    prompt_text += f"IMAGES: This document section includes {len(images)} visual element(s) such as diagrams, charts, or illustrations.\n\n"

            prompt_text += "\n"

        # If any images were counted across the retrieved documents, add a top-level
        # note to remind the model to consider visual content when formulating an answer.
        if image_count > 0:
            prompt_text += f"Note: The documents contain {image_count} image(s) with visual content that may contain important information.\n\n"

        prompt_text += (
            "Please provide a clear, comprehensive answer using the text, tables, "
            "and information from the images. If the documents don't contain sufficient information "
            'say "I don\'t have enough information to answer that question based on '
            'the provided documents."\n\nANSWER:'
        )

        # Query the LLM with the assembled prompt and return the raw response
        # text. The adapter will attempt backends in order and raise an
        # informative exception if all backends fail.
        return llm.invoke(prompt_text)

    except Exception as exc:
        print(f"❌ Answer generation failed: {exc}")
        return "Sorry, I encountered an error while generating the answer."
