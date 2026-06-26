"""
app/services/rag_service.py
============================
The RAG (Retrieval-Augmented Generation) orchestration service.

This is the "brain" of the application. It ties together:
1. EmbeddingService — to convert the user's question into a vector
2. VectorStoreService — to find the most relevant document chunks
3. Gemini LLM — to generate a natural language answer from those chunks

The RAG pipeline:
    User Question
        → Embed question (EmbeddingService)
        → Search for similar chunks (VectorStoreService)
        → Build prompt (question + context + chat history)
        → Generate answer (Gemini LLM)
        → Return answer + source citations

Two modes:
- ask():        Returns the complete answer at once.
- ask_stream(): Yields answer tokens as they're generated (for SSE streaming).
"""

import asyncio
import logging
from typing import List, Dict, Optional, AsyncGenerator

from google import genai
from google.api_core import exceptions as google_exceptions

from app.config import get_settings
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Retry configuration for LLM API calls
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0  # seconds

# System prompt that instructs the LLM how to behave
SYSTEM_PROMPT = """You are a helpful, accurate, and detailed document assistant.

Your role is to answer questions based ONLY on the provided document context.
Follow these rules strictly:

1. **Answer from context only**: Use only the information in the provided context chunks to answer the question. Do not use any external knowledge or make assumptions.

2. **Cite your sources**: When providing information, reference the source page number(s). Use the format "(Page X)" or "(Pages X, Y)" after relevant statements.

3. **Be honest about limitations**: If the provided context does not contain enough information to fully answer the question, clearly state: "I don't have enough information in the provided documents to answer this question completely." Then share whatever partial information IS available in the context.

4. **Be detailed and helpful**: Provide thorough, well-structured answers. Use bullet points, numbered lists, or paragraphs as appropriate for clarity.

5. **Handle ambiguity**: If the question is ambiguous, briefly note the ambiguity and answer the most likely interpretation based on the context.

6. **Maintain conversation context**: Consider the chat history to understand follow-up questions and maintain context across the conversation.
"""


# ─────────────────────────────────────────────
# RAG SERVICE CLASS
# ─────────────────────────────────────────────

class RAGService:
    """
    Orchestrates the full RAG pipeline: embed → retrieve → generate.

    Usage:
        service = RAGService()
        result = await service.ask("What is X?", doc_ids=[1, 2])
        # result = {"answer": "...", "sources": [...]}

    For streaming:
        async for chunk in service.ask_stream("What is X?"):
            print(chunk)  # partial answer tokens
    """

    def __init__(
        self,
        embedding_service: Optional[object] = None,
        vector_store: Optional[object] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the RAG service with its dependencies.

        Creates/reuses:
        - EmbeddingService for query embedding
        - VectorStoreService for chunk retrieval
        - google-genai Client for LLM generation
        """
        settings = get_settings()

        # ── Dependencies ──────────────────────
        self._embedding_service = embedding_service or get_embedding_service()
        self._vector_store = vector_store or get_vector_store()

        # ── Gemini LLM client ─────────────────
        # The same genai.Client works for both embeddings and LLM.
        # We create a separate one here for clarity and independence.
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._llm_model = settings.llm_model
        self._top_k = top_k if top_k is not None else settings.top_k

        logger.info(
            f"RAGService initialized (LLM: {self._llm_model}, top_k: {self._top_k})"
        )

    async def ask(
        self,
        question: str,
        doc_ids: Optional[List[int]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Answer a question using the full RAG pipeline (non-streaming).

        Pipeline:
        1. Embed the question → query vector
        2. Search vector store → top-K relevant chunks
        3. Build prompt → question + context + chat history
        4. Call Gemini LLM → generate answer
        5. Return answer + source citations

        Args:
            question:     The user's question string.
            doc_ids:      Optional list of document IDs to search within.
                          If None, searches all documents.
            chat_history: Optional list of previous messages for context.
                          Each dict: {"role": "user"|"assistant", "content": "..."}

        Returns:
            Dict with:
                - answer (str):   The generated answer text.
                - sources (list): List of source citation dicts, each with:
                    - document_id (int)
                    - page_number (int)
                    - chunk_text (str)
                    - relevance_score (float)

        Raises:
            ValueError: If the question is empty.
            RuntimeError: If the LLM call fails after retries.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        chat_history = chat_history or []

        logger.info(f"RAG ask: '{question[:80]}...' (doc_ids={doc_ids})")

        # ── Step 1: Embed the question ────────
        query_embedding = self._embedding_service.generate_embedding(question)
        logger.debug("Query embedding generated")

        # ── Step 2: Retrieve relevant chunks ──
        search_results = self._vector_store.search(
            query_embedding=query_embedding,
            doc_ids=doc_ids,
            top_k=self._top_k,
        )

        if not search_results:
            logger.warning("No relevant chunks found for the question")
            return {
                "answer": (
                    "I couldn't find any relevant information in the uploaded documents "
                    "to answer your question. Please make sure the relevant documents "
                    "have been uploaded and processed."
                ),
                "sources": [],
            }

        logger.info(f"Retrieved {len(search_results)} relevant chunks")

        # ── Step 3: Build the prompt ──────────
        prompt = self._build_prompt(question, search_results, chat_history)

        # ── Step 4: Call the LLM ──────────────
        answer = await self._generate_with_retry(prompt)

        # ── Step 5: Build source citations ────
        sources = [
            {
                "document_id": result["document_id"],
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "relevance_score": round(result["score"], 4),
            }
            for result in search_results
        ]

        logger.info(f"RAG answer generated ({len(answer)} chars, {len(sources)} sources)")

        return {
            "answer": answer,
            "sources": sources,
        }

    async def ask_stream(
        self,
        question: str,
        doc_ids: Optional[List[int]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Answer a question using RAG with streaming response.

        Same pipeline as ask(), but yields answer tokens as they're
        generated by the LLM. This enables Server-Sent Events (SSE)
        for real-time streaming to the frontend.

        Yields dicts in this format:
            {"type": "chunk", "content": "partial text..."}
            {"type": "sources", "content": [...source citations...]}
            {"type": "done", "content": ""}

        Args:
            question:     The user's question string.
            doc_ids:      Optional list of document IDs to search within.
            chat_history: Optional list of previous messages for context.

        Yields:
            Dicts with type and content for SSE streaming.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        chat_history = chat_history or []

        logger.info(f"RAG ask_stream: '{question[:80]}...' (doc_ids={doc_ids})")

        # ── Steps 1–3 are identical to ask() ──
        query_embedding = self._embedding_service.generate_embedding(question)

        search_results = self._vector_store.search(
            query_embedding=query_embedding,
            doc_ids=doc_ids,
            top_k=self._top_k,
        )

        if not search_results:
            yield {
                "type": "chunk",
                "content": (
                    "I couldn't find any relevant information in the uploaded documents "
                    "to answer your question."
                ),
            }
            yield {"type": "sources", "content": []}
            yield {"type": "done", "content": ""}
            return

        prompt = self._build_prompt(question, search_results, chat_history)

        # ── Step 4: Stream LLM response ───────
        try:
            # Use the async streaming API.
            # client.aio is the async interface of the genai Client.
            stream = self._client.aio.models.generate_content_stream(
                model=self._llm_model,
                contents=prompt,
            )

            async for response_chunk in stream:
                # Each chunk may have text content
                if response_chunk.text:
                    yield {
                        "type": "chunk",
                        "content": response_chunk.text,
                    }

        except Exception as e:
            logger.error(f"Streaming LLM call failed: {e}")
            yield {
                "type": "chunk",
                "content": f"Sorry, an error occurred while generating the answer: {str(e)}",
            }

        # ── Step 5: Send source citations ─────
        sources = [
            {
                "document_id": result["document_id"],
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "relevance_score": round(result["score"], 4),
            }
            for result in search_results
        ]

        yield {"type": "sources", "content": sources}
        yield {"type": "done", "content": ""}

        logger.info("Streaming RAG response completed")

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict],
        chat_history: List[Dict],
    ) -> str:
        """
        Build the full prompt for the LLM.

        The prompt structure:
        1. System instructions (how to behave)
        2. Document context (retrieved chunks with page numbers)
        3. Chat history (previous Q&A for follow-ups)
        4. Current question

        Args:
            question:       The user's current question.
            context_chunks: Retrieved chunks from vector search.
            chat_history:   Previous conversation messages.

        Returns:
            The complete prompt string for the LLM.
        """
        # ── Build context section ─────────────
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            page = chunk.get("page_number", "unknown")
            text = chunk.get("text", "")
            doc_id = chunk.get("document_id", "unknown")
            context_parts.append(
                f"[Source {i} | Document {doc_id}, Page {page}]\n{text}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        # ── Build chat history section ────────
        history_text = ""
        if chat_history:
            history_parts = []
            # Only include the last 10 messages to avoid prompt bloat
            recent_history = chat_history[-10:]
            for msg in recent_history:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                history_parts.append(f"{role}: {content}")
            history_text = "\n".join(history_parts)

        # ── Assemble the full prompt ──────────
        prompt = f"""{SYSTEM_PROMPT}

---

## Document Context

The following are relevant excerpts from the uploaded documents:

{context_text}

---"""

        if history_text:
            prompt += f"""

## Previous Conversation

{history_text}

---"""

        prompt += f"""

## Current Question

{question}

Please provide a detailed answer based on the document context above. Remember to cite page numbers."""

        return prompt

    async def _generate_with_retry(self, prompt: str) -> str:
        """
        Call the Gemini LLM with retry logic for rate limits.

        Uses exponential backoff on 429 (rate limit) and 503 (service
        unavailable) errors. Other errors are raised immediately.

        Args:
            prompt: The complete prompt string for the LLM.

        Returns:
            The generated answer text.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                # Use the async interface for non-blocking LLM calls.
                # generate_content returns a GenerateContentResponse.
                response = await self._client.aio.models.generate_content(
                    model=self._llm_model,
                    contents=prompt,
                )

                # Extract the text from the response.
                # response.text is a convenience property that returns
                # the concatenated text of all parts.
                if response.text:
                    return response.text
                else:
                    logger.warning("LLM returned empty response")
                    return "I was unable to generate an answer. Please try rephrasing your question."

            except google_exceptions.ResourceExhausted as e:
                # 429 Too Many Requests
                last_exception = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"LLM rate limited on attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except google_exceptions.ServiceUnavailable as e:
                # 503 Service Unavailable
                last_exception = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"LLM service unavailable on attempt {attempt + 1}/{MAX_RETRIES}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"LLM call failed with non-retryable error: {e}")
                raise RuntimeError(f"Failed to generate answer: {e}") from e

        # All retries exhausted
        logger.error(f"LLM call failed after {MAX_RETRIES} retries: {last_exception}")
        raise RuntimeError(
            f"Failed to generate answer after {MAX_RETRIES} retries: {last_exception}"
        )


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """
    Get or create the singleton RAGService instance.

    Returns:
        The shared RAGService instance.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
