"""
app/config.py
=============
Loads all environment variables from the .env file located in the project root
(IR INFOTECH/.env — shared across R1 and R2).

Uses pydantic-settings to validate and type-check them automatically.
Adds R2-specific settings for the RAG pipeline: database, vector store,
chunking parameters, embedding model, LLM model, and optional API key auth.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


# ── Locate the shared .env file ────────────────────────────
# This file lives at:  IR INFOTECH/.env
# config.py lives at:  IR INFOTECH/R2/app/config.py
# So we go up 3 levels:  config.py → app → R2 → IR INFOTECH
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    If a variable is missing from .env, it uses the default value defined here.
    The only REQUIRED variable (no default) is ``gemini_api_key``.
    """

    model_config = ConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
    )

    # ── Core App Info ───────────────────────────────────────
    app_name: str = "IR Infotech RAG Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Gemini API Key — REQUIRED, no default ──────────────
    gemini_api_key: str

    # ── Logging ─────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Database ────────────────────────────────────────────
    # SQLite database file.  Stored inside the R2/ directory by default.
    database_url: str = "sqlite:///./rag_assistant.db"

    # ── File Uploads ────────────────────────────────────────
    # Directory where uploaded PDFs are saved (relative to R2/).
    upload_dir: str = "uploads"

    # ── ChromaDB Vector Store ───────────────────────────────
    # Directory where ChromaDB persists its embeddings (relative to R2/).
    chroma_dir: str = "chroma_data"

    # ── Chunking Parameters ─────────────────────────────────
    # chunk_size  — max number of characters per text chunk
    # chunk_overlap — characters shared between consecutive chunks
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ───────────────────────────────────────────
    # top_k — how many chunks to retrieve from the vector store per query
    top_k: int = 5

    # ── AI Models ───────────────────────────────────────────
    # embedding_model — Google model used to create text embeddings
    embedding_model: str = "text-embedding-004"
    # llm_model — Google Gemini model used for answer generation
    llm_model: str = "gemini-2.5-flash"

    # ── Optional API Key Auth ───────────────────────────────
    # If set to a non-empty string, every request must include this key
    # in the header ``X-API-Key``.  Empty string = auth disabled.
    api_key_auth: str = ""


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    ``lru_cache()`` means it only reads the .env file once,
    not on every request — this is a performance best practice.
    """
    return Settings()
