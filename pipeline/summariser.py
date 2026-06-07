"""
pipeline/summariser.py
----------------------
Step 3 — Enrich chunks that contain tables or images with an AI-generated
         searchable summary, then wrap everything as LangChain Documents.
"""

import json
from typing import List, Optional

from langchain_core.documents import Document
from utils.llm import get_llm_client

from config.settings import CHAT_MODEL, LLM_TEMPERATURE
from utils.content_utils import separate_content_types


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_summary_prompt(
    text: str, tables: List[str], images: Optional[List[str]] = None
) -> str:
    prompt = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
    if tables:
        prompt += "TABLES:\n"
        for i, table in enumerate(tables):
            prompt += f"Table {i + 1}:\n{table}\n\n"
    
    if images:
        prompt += f"IMAGES: This content includes {len(images)} image(s) with visual diagrams, charts, or illustrations.\n"
        prompt += "Analyze these visual elements as part of your comprehensive description.\n\n"

    prompt += """YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable — prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""
    return prompt


def _create_ai_enhanced_summary(
    text: str, tables: List[str], images: List[str]
) -> str:
    """
    Send chunk content (text + tables + images) to Groq and return
    a rich, retrieval-optimised description.

    For images, we include them in the text prompt as image markers
    since Groq's text model processes visual descriptions via prompt context.

    Falls back to a plain text excerpt on any error.
    """
    try:
        # Use the adapter client so backends, retries and timeouts are handled
        # in one place. This decouples the summariser from any specific LLM
        # provider and allows transparent fallback (e.g. Groq -> OpenAI).
        client = get_llm_client()

        # Build the full prompt that contains text, table HTML and image markers
        prompt_text = _build_summary_prompt(text, tables, images)

        # The adapter exposes `invoke(prompt: str) -> str`. The adapter will
        # internally try each configured backend, retry on transient errors,
        # enforce per-call timeouts, and fall back to the next backend when
        # necessary. Summariser code remains simple and focused on prompt
        # construction rather than provider details.
        return client.invoke(prompt_text)

    except Exception as exc:
        print(f"     ❌ AI summary failed: {exc}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


# ── Public API ────────────────────────────────────────────────────────────────

def summarise_chunks(chunks: list) -> list[Document]:
    """
    Convert Unstructured chunks into LangChain Documents.

    For chunks that contain tables or images the page_content is replaced with
    an AI-generated summary optimised for semantic search.  Plain-text chunks
    are stored as-is.  Either way the original content (raw text, table HTML,
    image base64) is preserved in metadata for answer generation.

    Args:
        chunks: List of CompositeElement chunks from the chunker.

    Returns:
        List of LangChain Document objects ready for embedding.
    """
    print("🧠 Processing chunks with AI summaries...")

    documents: list[Document] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # Progress logging per chunk
        print(f"   Processing chunk {i + 1}/{total}")

        # Split the chunk into its component types (text, tables, images)
        content = separate_content_types(chunk)
        print(f"     Types found: {content['types']}")
        print(f"     Tables: {len(content['tables'])}, Images: {len(content['images'])}")

        # If the chunk contains tables or images, prefer an AI-generated summary
        # because these formats often require semantic interpretation beyond raw text.
        if content["tables"] or content["images"]:
            print("     → Creating AI summary for mixed content...")
            try:
                # Create an enhanced, retrieval-optimised description using the LLM
                enhanced = _create_ai_enhanced_summary(
                    content["text"], content["tables"], content["images"]
                )
                # Log a short preview of the AI output (truncated to avoid huge logs)
                print(f"     → AI summary created: {enhanced[:200]}...")
            except Exception as exc:
                # On any error from the AI service, fall back to raw text so ingestion continues
                print(f"     ❌ AI summary failed: {exc}")
                enhanced = content["text"]
        else:
            # No non-text content; keep the original text as the searchable content
            print("     → Using raw text (no tables/images)")
            enhanced = content["text"]

        doc = Document(
            page_content=enhanced,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content["text"],
                        "tables_html": content["tables"],
                        "images_base64": content["images"],
                    }
                )
            },
        )
        documents.append(doc)

    print(f"✅ Processed {len(documents)} chunks")
    return documents
