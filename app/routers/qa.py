"""
app/routers/qa.py
==================
Defines the question-answering API endpoint.

This is the core of the RAG system — it takes a user's question,
retrieves relevant document chunks from ChromaDB, and uses Gemini AI
to generate an answer grounded in the retrieved context.

Supports two response modes:
1. **Standard** — Returns the complete answer as a single JSON response
2. **Streaming** — Returns the answer as Server-Sent Events (SSE) for
   real-time display, with each token streamed as it's generated
"""

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db
from app.models.db_models import ChatSession, ChatMessage
from app.models.request_models import QuestionRequest
from app.models.response_models import AnswerResponse, SourceReference, ErrorResponse
from app.services.rag_service import RAGService
from app.services.auth_service import get_api_key_dependency
from app.logger import setup_logger

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

logger = setup_logger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/qa",
    tags=["Question Answering"],
    dependencies=[Depends(get_api_key_dependency)],
)


# ─────────────────────────────────────────────
# HELPER — Get or create a chat session
# ─────────────────────────────────────────────

def _get_or_create_session(
    session_id: str | None,
    db: Session,
) -> ChatSession:
    """
    Retrieve an existing chat session by ID, or create a new one.
    """
    if session_id:
        result = db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if session:
            logger.info(f"Resuming existing chat session: {session_id}")
            return session

        logger.warning(
            f"Session '{session_id}' not found — creating a new session with this ID"
        )

    new_session_id = session_id or str(uuid.uuid4())
    session = ChatSession(id=new_session_id)
    db.add(session)
    db.flush()

    logger.info(f"Created new chat session: {new_session_id}")
    return session


# ─────────────────────────────────────────────
# HELPER — Load chat history for context
# ─────────────────────────────────────────────

def _load_chat_history(
    session_id: str,
    db: Session,
    max_messages: int = 10,
) -> list[dict[str, str]]:
    """
    Load recent chat messages for a session to provide conversation context.
    """
    result = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(max_messages)
    )
    messages = result.scalars().all()

    # Reverse to get chronological order (we fetched most recent first)
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(messages)
    ]

    logger.info(f"Loaded {len(history)} messages from session {session_id}")
    return history


# ─────────────────────────────────────────────
# HELPER — Save a message to chat history
# ─────────────────────────────────────────────

def _save_message(
    session_id: str,
    role: str,
    content: str,
    db: Session,
    sources: list[dict] | None = None,
) -> ChatMessage:
    """
    Save a single chat message (question or answer) to the database.
    """
    sources_str = json.dumps(sources) if sources else None
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources_str,
    )
    db.add(message)
    return message


# ─────────────────────────────────────────────
# HELPER — SSE streaming generator
# ─────────────────────────────────────────────

async def _stream_sse_response(
    question: str,
    session_id: str,
    chat_history: list[dict[str, str]],
    doc_ids: list[int] | None,
    rag_service: RAGService,
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events (SSE) for streaming responses.
    """
    full_answer = ""
    sources = []

    try:
        # Stream tokens from the RAG service
        async for event in rag_service.ask_stream(
            question=question,
            doc_ids=doc_ids,
            chat_history=chat_history,
        ):
            if event.get("type") == "chunk":
                token = event["content"]
                full_answer += token
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

            elif event.get("type") == "sources":
                sources = event["content"]

            elif event.get("type") == "error":
                yield f"data: {json.dumps({'error': event['content'], 'done': True})}\n\n"
                return

        # Save the question and answer to chat history
        _save_message(session_id, "user", question, db)
        _save_message(session_id, "assistant", full_answer, db, sources=sources)
        db.commit()

        # Final event with the complete answer and sources
        yield f"data: {json.dumps({'token': '', 'done': True, 'answer': full_answer, 'sources': sources})}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# ─────────────────────────────────────────────
# POST /qa/ask — Ask a question
# ─────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a Question",
    description=(
        "Ask a question about your uploaded documents. The system retrieves "
        "relevant chunks from the vector store and generates an AI-powered "
        "answer grounded in your documents."
    ),
    responses={
        200: {"description": "Answer generated successfully"},
        400: {"model": ErrorResponse, "description": "No documents uploaded yet"},
        422: {"description": "Validation error — check your request body"},
        500: {"model": ErrorResponse, "description": "RAG pipeline error"},
    },
)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db),
) -> AnswerResponse | StreamingResponse:
    """
    ## Ask a Question
    """
    logger.info(
        f"POST /qa/ask — Question: '{request.question[:80]}...' "
        f"(stream={request.stream}, session={request.session_id})"
    )

    try:
        # Step 1: Get or create the chat session
        session = _get_or_create_session(request.session_id, db)

        # Step 2: Load chat history for context
        chat_history = _load_chat_history(session.id, db)

        # Step 3: Initialize RAG service with dynamic top_k
        rag_service = RAGService(top_k=request.top_k)

        # Step 4: Handle streaming vs standard response
        if request.stream:
            logger.info("Returning streaming SSE response")
            return StreamingResponse(
                _stream_sse_response(
                    question=request.question,
                    session_id=session.id,
                    chat_history=chat_history,
                    doc_ids=request.document_ids,
                    rag_service=rag_service,
                    db=db,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Step 5: Standard (non-streaming) response
        result = await rag_service.ask(
            question=request.question,
            doc_ids=request.document_ids,
            chat_history=chat_history,
        )

        # Step 6: Save question and answer to chat history
        _save_message(session.id, "user", request.question, db)
        _save_message(
            session.id,
            "assistant",
            result["answer"],
            db,
            sources=result.get("sources"),
        )
        db.commit()

        # Step 7: Build and return the response
        sources = [
            SourceReference(
                document_id=s["document_id"],
                document_name=s.get("document_name", ""),
                page_number=s["page_number"],
                chunk_text=s["chunk_text"],
                relevance_score=s["relevance_score"],
            )
            for s in result.get("sources", [])
        ]

        logger.info(
            f"✅ Answer generated — {len(result['answer'])} chars, "
            f"{len(sources)} sources, session: {session.id}"
        )

        return AnswerResponse(
            answer=result["answer"],
            sources=sources,
            session_id=session.id,
            question=request.question,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ RAG pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}",
        )
