"""
app/models/request_models.py
=============================
Pydantic models that define the SHAPE of incoming request bodies.

Think of these as "blueprints" for what the API expects.
FastAPI uses these to:
  - Automatically validate incoming JSON
  - Show correct examples in /docs
  - Return friendly error messages if validation fails

NOTE: File uploads (PDF upload) use ``UploadFile`` from FastAPI directly —
there is no Pydantic request model for multipart form data.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ─────────────────────────────────────────────
# ASK A QUESTION
# ─────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """
    Request body for POST /ask — ask a question against uploaded documents.

    Fields:
        question:     The natural-language question to answer (required).
        document_ids: Optional list of document IDs to restrict the search to.
                      If ``None`` or empty, ALL documents are searched.
        session_id:   Optional existing chat session ID to continue a conversation.
                      If ``None``, a new session is created automatically.
        stream:       If ``True``, the response will be streamed using
                      Server-Sent Events (SSE).  Default is ``False``.
    """

    question: str = Field(
        ...,  # '...' means this field is REQUIRED
        min_length=1,
        description="The question you want to ask about your documents.",
        examples=["What are the key findings in the report?"],
    )

    document_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "List of document IDs to restrict the search to. "
            "Pass None or omit to search ALL documents."
        ),
        examples=[[1, 3]],
    )

    session_id: Optional[str] = Field(
        default=None,
        description=(
            "An existing chat session ID to continue a conversation. "
            "Omit to start a new session."
        ),
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )

    stream: bool = Field(
        default=False,
        description="Set to true to receive a streaming (SSE) response.",
    )

    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top chunks to retrieve from the vector store.",
        examples=[5],
    )

