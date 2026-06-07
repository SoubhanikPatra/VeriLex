"""
pipeline/partitioner.py
-----------------------
Step 1 — Extract structured elements from a PDF using Unstructured.
"""

from unstructured.partition.pdf import partition_pdf

from config.settings import PARTITION_STRATEGY, PARTITION_IMAGE_BLOCK_TYPES


def partition_document(file_path: str) -> list:
    """
    Partition a PDF into typed Unstructured elements.

    Uses hi-res strategy by default for maximum accuracy.
    Tables are kept as structured HTML; images are stored as base64.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Unstructured elements (Title, NarrativeText, Table, Image, …).
    """
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to source PDF file
        strategy=PARTITION_STRATEGY,  # Partitioning strategy defined in config.settings
        infer_table_structure=True,  # Preserve and return table structure as HTML when possible
        extract_image_block_types=PARTITION_IMAGE_BLOCK_TYPES,  # Image block types to extract
        extract_image_block_to_payload=True,  # Embed extracted images into element payloads (base64)
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements
