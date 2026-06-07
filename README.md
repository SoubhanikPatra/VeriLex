# VeriLex

VeriLex is a multimodal RAG pipeline for PDF ingestion and grounded question answering.  
It includes both a CLI workflow and a FastAPI service built on top of the same pipeline modules.

## What it does

- Partitions PDF content (text, tables, images)
- Chunks and summarizes content for retrieval
- Stores embeddings in ChromaDB
- Answers questions with retrieved context
- Supports interactive chat via CLI and stateless chat turns via API

## Project structure

```text
.
├── app/                # FastAPI service
├── config/             # Central settings
├── pipeline/           # Ingestion + retrieval pipeline modules
├── utils/              # Export/content helpers
├── main.py             # CLI entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Requirements

### System packages

```bash
# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr libmagic-dev

# macOS
brew install poppler tesseract libmagic
```

### Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the repository root:

```text
GROQ_API_KEY=your_groq_api_key
# Optional (used if OpenAI backend is enabled)
OPENAI_API_KEY=your_openai_api_key
```

Notes:
- Backend order is controlled by `LLM_BACKENDS` in `config/settings.py` (default: `groq,openai`).
- Default vector-store path is `db/chroma_db`.

## CLI usage

Place one or more PDF files in `docs/`, then run:

```bash
# Ingest all PDFs in docs/
python main.py

# Ingest then immediately ask one question
python main.py --question "What is this document about?"

# Query an existing Chroma DB
python main.py query --question "What are the key findings?" --db db/chroma_db

# Interactive multi-turn chat (in-memory history)
python main.py chat --db db/chroma_db
```

## API usage

Start the API server:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /ingest`
- `POST /query`
- `POST /chat/turn`

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the document about?","persist_dir":"db/chroma_db"}'
```

## Docker

```bash
docker compose up --build
```

This mounts `docs/` and `db/chroma_db/` so PDFs and embeddings persist locally.
