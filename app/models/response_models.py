"""
app/models/response_models.py
==============================
Pydantic models that define the SHAPE of the API responses.

These ensure the API always returns data in a consistent, predictable format.
Each response model maps to one (or more) API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ─────────────────────────────────────────────
# SOURCE REFERENCE (reused in multiple responses)
# ─────────────────────────────────────────────

class SourceReference(BaseModel):
    """
    A single source chunk that was used to generate an answer.

    Returned alongside the AI's answer so the user can verify
    which parts of which documents the answer was based on.
    """

    document_id: int = Field(description="ID of the source document.")
    document_name: str = Field(description="Original filename of the source document.")
    page_number: int = Field(description="Page number in the PDF where this chunk came from.")
    chunk_text: str = Field(description="Snippet of the relevant text chunk.")
    relevance_score: float = Field(description="Similarity score (0–1) from the vector search.")


# ─────────────────────────────────────────────
# ANSWER (POST /ask)
# ─────────────────────────────────────────────

class AnswerResponse(BaseModel):
    """Response body for POST /ask — the AI-generated answer."""

    answer: str = Field(description="The AI-generated answer to the user's question.")
    sources: List[SourceReference] = Field(description="Source chunks that informed the answer.")
    session_id: str = Field(description="Chat session ID (new or existing).")
    question: str = Field(description="The original question that was asked.")


# ─────────────────────────────────────────────
# DOCUMENT
# ─────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Metadata for a single uploaded document."""

    id: int = Field(description="Unique document ID.")
    filename: str = Field(description="Stored filename on disk.")
    original_filename: str = Field(description="Original filename the user uploaded.")
    file_size: int = Field(description="File size in bytes.")
    page_count: Optional[int] = Field(default=None, description="Number of pages in the PDF.")
    chunk_count: Optional[int] = Field(default=None, description="Number of text chunks created.")
    upload_date: str = Field(description="ISO-8601 timestamp of when the file was uploaded.")
    status: str = Field(description="Processing status: 'processing', 'ready', or 'error'.")


class DocumentListResponse(BaseModel):
    """Response body for GET /documents — list of all documents."""

    documents: List[DocumentResponse] = Field(description="List of document metadata objects.")
    total: int = Field(description="Total number of documents.")


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response body for POST /upload — confirmation after uploading files."""

    message: str = Field(description="Human-readable success message.")
    documents: List[DocumentResponse] = Field(
        description="Metadata for each uploaded document (status will be 'processing')."
    )


# ─────────────────────────────────────────────
# CHAT SESSION
# ─────────────────────────────────────────────

class ChatSessionResponse(BaseModel):
    """Metadata for a single chat session."""

    id: str = Field(description="UUID of the chat session.")
    title: str = Field(description="Session title.")
    created_at: str = Field(description="ISO-8601 timestamp of session creation.")
    updated_at: str = Field(description="ISO-8601 timestamp of last update.")
    message_count: int = Field(description="Total number of messages in this session.")


# ─────────────────────────────────────────────
# CHAT MESSAGE / HISTORY
# ─────────────────────────────────────────────

class ChatMessageResponse(BaseModel):
    """A single message within a chat session."""

    id: int = Field(description="Unique message ID.")
    role: str = Field(description="'user' or 'assistant'.")
    content: str = Field(description="The message text.")
    sources: Optional[List[SourceReference]] = Field(
        default=None,
        description="Source references (only present for assistant messages).",
    )
    created_at: str = Field(description="ISO-8601 timestamp of the message.")


class ChatHistoryResponse(BaseModel):
    """Full chat history for a session."""

    session_id: str = Field(description="UUID of the chat session.")
    title: str = Field(description="Session title.")
    messages: List[ChatMessageResponse] = Field(description="Ordered list of messages.")


# ─────────────────────────────────────────────
# ERROR RESPONSE (used by error handler)
# ─────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response format returned by the global error handler."""

    error: str = Field(description="Short error type/name.")
    message: str = Field(description="Human-readable description of what went wrong.")
    status_code: int = Field(description="HTTP status code.")


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health — application health status."""

    status: str = Field(description="Overall status, e.g. 'healthy'.")
    app: str = Field(description="Application name.")
    version: str = Field(description="Application version.")
    documents_count: int = Field(description="Number of documents currently stored.")
    vector_store_status: str = Field(
        description="Status of the ChromaDB vector store, e.g. 'connected' or 'unavailable'."
    )
