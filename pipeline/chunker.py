"""
pipeline/chunker.py
-------------------
Step 2 — Split Unstructured elements into semantically coherent chunks.
"""

from unstructured.chunking.title import chunk_by_title

from config.settings import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_NEW_AFTER_N_CHARS,
    CHUNK_COMBINE_UNDER_N_CHARS,
)


def create_chunks_by_title(elements: list) -> list:
    """
    Group Unstructured elements into chunks bounded by document titles/sections.

    Chunk size limits (all configurable in config/settings.py):
      - max_characters          : hard upper limit per chunk
      - new_after_n_chars       : soft target where a new chunk is preferred
      - combine_text_under_n_chars : merge tiny trailing chunks with their predecessor

    Args:
        elements: List of Unstructured elements from the partitioner.

    Returns:
        List of CompositeElement chunks.
    """
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        new_after_n_chars=CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER_N_CHARS,
    )

    print(f"✅ Created {len(chunks)} chunks")
    return chunks
