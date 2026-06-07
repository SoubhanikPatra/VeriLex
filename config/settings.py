"""
config/settings.py
------------------
Central configuration for the RAG pipeline.
Override any value here or via environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_CHARACTERS = 3000
CHUNK_NEW_AFTER_N_CHARS = 2400
CHUNK_COMBINE_UNDER_N_CHARS = 500

# ── PDF Partitioning ──────────────────────────────────────────────────────────
PARTITION_STRATEGY = "hi_res"
PARTITION_IMAGE_BLOCK_TYPES = ["Image"]

# ── AI Models ─────────────────────────────────────────────────────────────────
CHAT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_TEMPERATURE = 0

# ── Vector Store ──────────────────────────────────────────────────────────────
DEFAULT_PERSIST_DIR = "db/chroma_db"
CHROMA_SPACE = "cosine"
RETRIEVER_K = 3

# ── LLM Adapter / Resilience ─────────────────────────────────────────────────
# Ordered list of backends to try (first successful backend is used)
LLM_BACKENDS = os.getenv("LLM_BACKENDS", "groq,openai").split(",")
# Per-request timeout (seconds) for LLM calls. Adapter enforces this when possible.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "20"))
# Number of retry attempts per-backend before falling back to the next backend.
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))
# Backoff configuration used by tenacity (initial multiplier, multiplier, max)
LLM_BACKOFF = {
	"initial": float(os.getenv("LLM_BACKOFF_INITIAL", "1")),
	"multiplier": float(os.getenv("LLM_BACKOFF_MULTIPLIER", "2")),
	"max": float(os.getenv("LLM_BACKOFF_MAX", "10")),
}

# Optional override for OpenAI model name (used by the OpenAI backend)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
