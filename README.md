# RAG Ingestion Pipeline

A modular RAG (Retrieval-Augmented Generation) pipeline using Unstructured.io, LangChain, and ChromaDB.

## Project Structure

```
rag_pipeline/
├── config/
│   └── settings.py          # All configuration and constants
├── pipeline/
│   ├── partitioner.py       # PDF partitioning via Unstructured
│   ├── chunker.py           # Title-based chunking
│   ├── summariser.py        # AI-enhanced summarisation
│   └── vector_store.py      # ChromaDB embedding + storage
├── utils/
│   ├── content_utils.py     # Content type separation helpers
│   └── export_utils.py      # JSON export utilities
├── main.py                  # Entry point — run full pipeline or query
└── requirements.txt
```

## Setup

### System Dependencies

```bash
# Linux
apt-get install poppler-utils tesseract-ocr libmagic-dev

# macOS
brew install poppler tesseract libmagic
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

## Usage

### Run the full ingestion pipeline

```bash
python main.py ingest --pdf ./docs/attention-is-all-you-need.pdf
```

### Query the vector store

```bash
python main.py query --question "What are the two main components of the Transformer architecture?"
```

### Run the HTTP API

The project now includes a small FastAPI service so you can deploy it as a
usable application instead of only a CLI.

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — health probe for load balancers and container checks.
- `POST /ingest` — ingest a single PDF or every PDF in `docs/`.
- `POST /query` — ask a question against the persisted vector store.
- `POST /chat/turn` — one stateless chat turn with optional caller-provided history.

Example query request:

```bash
curl -X POST http://localhost:8000/query \
   -H "Content-Type: application/json" \
   -d '{"question":"What is the document about?","persist_dir":"db/chroma_db"}'
```

### Container deployment

Build and run the service locally with Docker:

```bash
docker build -t verilex:latest .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/docs:/app/docs:ro" -v "$PWD/db/chroma_db:/app/db/chroma_db" verilex:latest
```

Or use Docker Compose:

```bash
docker compose up --build
```

The compose file mounts `docs/` and `db/chroma_db/` so your local PDFs and
vector store persist across restarts.

### Quick commands and notes

- `python main.py` — run the default ingestion flow (equivalent to `ingest`). Requires a `docs/` folder containing PDF files.
- `python main.py --question "..."` — ingest then immediately run a query against the newly created DB.
- `python main.py query --question "..." --db db/chroma_db` — query an existing ChromaDB persist directory.
- `python main.py chat --db db/chroma_db` — start an interactive multi-turn chat REPL. History is kept in-memory per session.

Defaults:

- ChromaDB persist directory: `db/chroma_db`

Environment:

- `GROQ_API_KEY` — API key used by the Groq chat LLM client (set in `.env`).

Notes:

- The pipeline preserves tables as HTML and images as base64 in document metadata so the LLM prompt can reference them. Large documents may need chunking to avoid model context limits.
- The `chat` REPL is a minimal demo: it includes prior turns in the prompt but does not persist sessions. For production multi-user chat, add session storage and history truncation.

Deployment notes:

- The API service is intentionally thin and reuses the same pipeline functions as the CLI.
- The current HTTP service is stateless per request; session persistence is a next-step item.
- For GitHub, create a new repository, commit these files, then push your branch with `git push -u origin main`.

## Walkthrough — run the code

Follow these steps to set up and run the pipeline locally.

1. (Optional) Activate the included virtual environment or create a new one.

   Use the bundled venv:

   ```bash
   source verilex/bin/activate
   pip install -r requirements.txt
   ```

   Or create a fresh venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root and add your API key(s):

   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Put the PDF(s) you want to ingest into the `docs/` folder.

4. Run the ingestion pipeline (default):

   ```bash
   python main.py
   ```

   This will partition, chunk, summarise, export `chunks_export.json`, and create a ChromaDB at `db/chroma_db` (or the path passed with `--db`).

5. Run a single query against the saved DB:

   ```bash
   python main.py query --question "What are the two main components of the Transformer?"
   ```

6. Start the interactive multi-turn chat REPL:

   ```bash
   python main.py chat --db db/chroma_db
   ```

7. Outputs and debugging:
   - `chunks_export.json` contains the processed chunks and enhanced content used for embeddings.
   - `rag_results.json` is written when running queries and contains retrieved context.

8. Troubleshooting:
   - If `python main.py` errors with "Docs folder not found", create a `docs/` directory and add PDFs.
   - Ensure `GROQ_API_KEY` is set in `.env` for LLM access.
   - If embeddings or Chroma fail, confirm `requirements.txt` dependencies are installed and your Python version matches the virtualenv (project uses Python 3.10).
