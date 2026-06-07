"""
utils/export_utils.py
---------------------
Helpers for exporting processed chunks to JSON.
"""

import json
from typing import Any


def export_chunks_to_json(
    chunks: list, filename: str = "chunks_export.json"
) -> list[dict[str, Any]]:
    """
    Serialize a list of LangChain Documents (or Unstructured chunks) to JSON.

    Each entry contains:
      - chunk_id          : 1-based index
      - enhanced_content  : page_content used for embedding/retrieval
      - metadata          : original_content dict (raw_text, tables_html, images_base64)
    """
    export_data = []

    # Build a JSON-serializable representation of each chunk/document so it can
    # be inspected or re-used outside the pipeline (e.g., for debugging or manual review).
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(
                    doc.metadata.get("original_content", "{}")
                )
            },
        }
        export_data.append(chunk_data)

    # Persist to disk using UTF-8 to preserve non-ASCII characters
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data
