"""
main.py
-------
Entry point for the RAG ingestion pipeline.

Usage
-----
# Ingest all PDFs from the docs folder (default)
python main.py

# Query an existing vector store
python main.py query --question "What are the two main components of the Transformer?"

# Ingest and immediately query
python main.py --question "How many attention heads are used?"
"""

import argparse
from pathlib import Path

from pipeline import (
    create_chunks_by_title,
    create_vector_store,
    generate_final_answer,
    load_vector_store,
    partition_document,
    retrieve_chunks,
    summarise_chunks,
)
from utils.export_utils import export_chunks_to_json
from config.settings import DEFAULT_PERSIST_DIR

DOCS_DIR = Path("./docs")


def get_pdfs_from_docs() -> list[Path]:
    """Get all PDF files from the docs folder."""
    # Ensure the docs directory exists and contains PDF files to ingest
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found at {DOCS_DIR}")
    
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}")
    
    return pdfs


def run_ingestion_pipeline(pdf_path: str, persist_dir: str = DEFAULT_PERSIST_DIR):
    """Run all four ingestion steps and return the vector store."""
    # High-level ingestion flow:
    # 1) Partition the PDF into Unstructured elements (text, tables, images)
    # 2) Chunk the elements into semantically-coherent pieces
    # 3) Summarise mixed-content chunks using an LLM for retrieval optimisation
    # 4) Export the processed chunks for inspection
    # 5) Embed the final documents and persist the ChromaDB collection
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)
    export_chunks_to_json(summarised, "chunks_export.json")
    db = create_vector_store(summarised, persist_directory=persist_dir)

    print("🎉 Pipeline completed successfully!")
    return db


def run_query(question: str, persist_dir: str = DEFAULT_PERSIST_DIR) -> str:
    """Load an existing vector store and answer a question."""
    # Load a saved ChromaDB collection, retrieve the most relevant chunks,
    # export the retrieved context for debugging, and ask the LLM to answer.
    db = load_vector_store(persist_dir)
    chunks = retrieve_chunks(db, question)
    export_chunks_to_json(chunks, "rag_results.json")
    return generate_final_answer(chunks, question)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    
    # Add default arguments for when no subcommand is provided
    parser.add_argument(
        "--db", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--question", default=None, help="Optional: query immediately after ingestion"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=False)
    parser.set_defaults(command="ingest")

    # ingest sub-command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest all PDFs from the docs folder into ChromaDB")
    ingest_parser.add_argument(
        "--db", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory"
    )
    ingest_parser.add_argument(
        "--question", default=None, help="Optional: *query immediately after ingestion"
    )

    # query sub-command
    query_parser = subparsers.add_parser("query", help="Query an existing vector store")
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument(
        "--db", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory"
    )

    # chat sub-command: start an interactive multi-turn REPL using the saved DB
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session against the vector store")
    chat_parser.add_argument(
        "--db", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory"
    )

    return parser

def run_conversational_session(persist_dir: str = DEFAULT_PERSIST_DIR):
    """Load the vector store and run an interactive chat loop with history.

    This is a minimal conversational wrapper: it keeps the last turns in
    memory, includes them in the prompt for each new question, and appends
    the model's answer to the history. It's intended as a straightforward
    demo of multi-turn RAG without external session storage.
    """
    db = load_vector_store(persist_dir)
    history: list[dict] = []
    print("🔊 Starting interactive chat session (type 'exit' or 'quit' to stop)")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break

        if not question or question.lower() in ("exit", "quit"):
            print("Exiting chat session.")
            break

        # Retrieve context for the current question
        chunks = retrieve_chunks(db, question)

        # Generate an answer including prior turns for context
        answer = generate_final_answer(chunks, question, history=history)

        # Print and store the turn
        print("\nAssistant:")
        print(answer)
        history.append({"user": question, "assistant": answer})


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "ingest":
        pdfs = get_pdfs_from_docs()
        print(f"📁 Found {len(pdfs)} PDF(s) in {DOCS_DIR}")
        
        # Process each PDF
        db = None
        for pdf_file in pdfs:
            print(f"\n📄 Processing: {pdf_file.name}")
            db = run_ingestion_pipeline(str(pdf_file), persist_dir=args.db)
        
        if args.question and db:
            chunks = retrieve_chunks(db, args.question)
            answer = generate_final_answer(chunks, args.question)
            print("\n" + "=" * 50)
            print("📬 Answer:")
            print(answer)

    elif args.command == "query":
        answer = run_query(args.question, persist_dir=args.db)
        print("\n" + "=" * 50)
        print("📬 Answer:")
        print(answer)

    elif args.command == "chat":
        # Start a conversational REPL that keeps in-memory history across turns
        run_conversational_session(persist_dir=args.db)