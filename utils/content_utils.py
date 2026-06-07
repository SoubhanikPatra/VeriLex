"""
utils/content_utils.py
----------------------
Helpers for inspecting and separating content types inside a chunk.
"""

from typing import TypedDict


class ChunkContent(TypedDict):
    text: str
    tables: list[str]
    images: list[str]   # base64 strings
    types: list[str]


def separate_content_types(chunk) -> ChunkContent:
    """
    Walk a chunk's original elements and split them by content type.

    Returns a dict with:
      - text   : raw text of the whole chunk
      - tables : list of HTML strings for each table found
      - images : list of base64 strings for each image found
      - types  : deduplicated list of content types present
    """
    # Initialize with the chunk's full text. Tables and images will be appended
    # if found in the chunk's original elements metadata.
    content_data: ChunkContent = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    # If the chunk retains `orig_elements` from Unstructured, iterate them and
    # collect tables and images into separate lists. We preserve table HTML when
    # available so downstream prompts can include structured table markup.
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                # Prefer any HTML representation stored in element.metadata
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                # Images are stored as base64 strings in metadata by the partitioner
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    # Deduplicate the types list and return the collected content.
    content_data["types"] = list(set(content_data["types"]))
    return content_data
