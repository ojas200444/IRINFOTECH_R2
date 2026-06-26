"""
app/services/embedding_service.py
=================================
Generates text embeddings using Google's Gemini text-embedding-004 model.

Embeddings are dense vector representations of text — they capture the
*meaning* of a sentence as a list of floats. Two texts about the same
topic will have similar embeddings (high cosine similarity), which is
how RAG retrieval works.

This service wraps the google-genai SDK:
    client.models.embed_content(model=..., contents=...)

Key design decisions:
- Batch processing: embed many texts in one API call (faster, fewer RPCs).
- Rate limit handling: exponential backoff + retry on 429 errors.
- Singleton instance: one EmbeddingService shared across the app.
"""

import time
import logging
from typing import List

from google import genai
from google.api_core import exceptions as google_exceptions

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Maximum number of texts to embed in a single API call.
# The Gemini API accepts batches, but very large batches
# can time out or hit payload limits.
MAX_BATCH_SIZE = 100

# Retry configuration for rate limit (429) errors
MAX_RETRIES = 5
BASE_RETRY_DELAY = 1.0  # seconds, doubles each retry


# ─────────────────────────────────────────────
# EMBEDDING SERVICE CLASS
# ─────────────────────────────────────────────

class EmbeddingService:
    """
    Generates text embeddings via the Gemini text-embedding-004 model.

    Usage:
        service = EmbeddingService()
        vectors = service.generate_embeddings(["hello", "world"])
        # vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]

    The service is designed to be used as a singleton — create one
    instance and reuse it throughout the app's lifetime.
    """

    def __init__(self) -> None:
        """
        Initialize the embedding service.

        Creates a google-genai Client using the Gemini API key from settings.
        The client is reused for all embedding calls.
        """
        settings = get_settings()
        self._model = settings.embedding_model

        # Initialize the google-genai client with the API key.
        # This client handles authentication and HTTP transport.
        self._client = genai.Client(api_key=settings.gemini_api_key)

        logger.info(
            f"EmbeddingService initialized with model '{self._model}'"
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Handles batching automatically: if texts exceed MAX_BATCH_SIZE,
        they are split into smaller batches and processed sequentially.

        Args:
            texts: List of text strings to embed. Each string should be
                   a chunk of document text (typically 200–1000 chars).

        Returns:
            List of embedding vectors. Each vector is a list of floats.
            The order matches the input texts list.

        Raises:
            ValueError: If the texts list is empty.
            RuntimeError: If the API call fails after all retries.
        """
        if not texts:
            logger.warning("generate_embeddings() called with empty texts list")
            raise ValueError("Cannot generate embeddings for empty text list.")

        logger.info(f"Generating embeddings for {len(texts)} texts")

        all_embeddings: List[List[float]] = []

        # ── Process in batches ────────────────
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(texts))
            batch = texts[batch_start:batch_end]

            logger.debug(
                f"Processing embedding batch {batch_start // MAX_BATCH_SIZE + 1} "
                f"({len(batch)} texts)"
            )

            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info(
            f"Successfully generated {len(all_embeddings)} embeddings "
            f"(dimension: {len(all_embeddings[0]) if all_embeddings else 'N/A'})"
        )
        return all_embeddings

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string.

        Convenience method that wraps generate_embeddings() for
        single-text use cases (e.g. embedding a user's query).

        Args:
            text: The text string to embed.

        Returns:
            A single embedding vector (list of floats).

        Raises:
            ValueError: If the text is empty.
            RuntimeError: If the API call fails after all retries.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        embeddings = self.generate_embeddings([text])
        return embeddings[0]

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Call the Gemini embed_content API with retry logic.

        Retries on rate limit (429) and transient server (5xx) errors
        with exponential backoff. Other errors are raised immediately.

        Args:
            texts: Batch of texts to embed (max MAX_BATCH_SIZE).

        Returns:
            List of embedding vectors for the batch.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                # The google-genai SDK concatenates lists of strings into a single embedding.
                # To get an embedding per chunk, we must call it individually.
                embeddings = []
                for text in texts:
                    response = self._client.models.embed_content(
                        model=self._model,
                        contents=text,
                    )
                    embeddings.extend([
                        embedding.values for embedding in response.embeddings
                    ])
                
                return embeddings

            except google_exceptions.ResourceExhausted as e:
                # 429 Too Many Requests — rate limited
                last_exception = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limited on embedding attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

            except google_exceptions.ServiceUnavailable as e:
                # 503 Service Unavailable — transient error
                last_exception = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Service unavailable on attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

            except Exception as e:
                # Non-retryable error — raise immediately
                logger.error(f"Embedding API call failed with non-retryable error: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        # All retries exhausted
        logger.error(
            f"Embedding API call failed after {MAX_RETRIES} retries. "
            f"Last error: {last_exception}"
        )
        raise RuntimeError(
            f"Embedding generation failed after {MAX_RETRIES} retries: {last_exception}"
        )


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────
# Created once and reused. Import this in other modules:
#     from app.services.embedding_service import embedding_service

_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the singleton EmbeddingService instance.

    Using a function (instead of a module-level instance) avoids
    initialization errors at import time — the service is only
    created when first needed.

    Returns:
        The shared EmbeddingService instance.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
