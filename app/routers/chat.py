"""
app/routers/chat.py
====================
Defines chat history management endpoints.

Chat sessions group related questions and answers together. This allows
users to have ongoing conversations about their documents, with the AI
able to reference previous messages for context.
"""

import json
from typing import List

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, delete

from app.database import get_db
from app.models.db_models import ChatSession, ChatMessage
from app.models.response_models import (
    ChatSessionResponse,
    ChatHistoryResponse,
    ChatMessageResponse,
    ErrorResponse,
)
from app.services.auth_service import get_api_key_dependency
from app.logger import setup_logger

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

logger = setup_logger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat History"],
    dependencies=[Depends(get_api_key_dependency)],
)


# ─────────────────────────────────────────────
# GET /chat/sessions — List all sessions
# ─────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=List[ChatSessionResponse],
    summary="List Chat Sessions",
    description=(
        "Retrieve all chat sessions with their message counts. "
        "Sessions are ordered by most recently active first."
    ),
    responses={
        200: {"description": "List of chat sessions returned successfully"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def list_sessions(
    db: Session = Depends(get_db),
) -> List[ChatSessionResponse]:
    """
    ## List Chat Sessions
    """
    logger.info("GET /chat/sessions — Listing all chat sessions")

    try:
        # Query all sessions with their message counts
        message_count_subquery = (
            select(
                ChatMessage.session_id,
                func.count(ChatMessage.id).label("message_count"),
            )
            .group_by(ChatMessage.session_id)
            .subquery()
        )

        result = db.execute(
            select(
                ChatSession,
                func.coalesce(
                    message_count_subquery.c.message_count, 0
                ).label("message_count"),
            )
            .outerjoin(
                message_count_subquery,
                ChatSession.id == message_count_subquery.c.session_id,
            )
            .order_by(ChatSession.updated_at.desc())
        )
        rows = result.all()

        sessions = [
            ChatSessionResponse(
                id=session.id,
                title=session.title,
                message_count=message_count,
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat() if session.updated_at else session.created_at.isoformat(),
            )
            for session, message_count in rows
        ]

        logger.info(f"Returning {len(sessions)} chat session(s)")
        return sessions

    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat sessions: {str(e)}",
        )


# ─────────────────────────────────────────────
# GET /chat/sessions/{session_id} — Get session messages
# ─────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get Chat History",
    description=(
        "Retrieve the complete message history for a specific chat session. "
        "Messages are returned in chronological order."
    ),
    responses={
        200: {"description": "Chat history returned successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db),
) -> ChatHistoryResponse:
    """
    ## Get Chat History
    """
    logger.info(f"GET /chat/sessions/{session_id} — Fetching session history")

    try:
        # Step 1: Verify the session exists
        result = db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if session is None:
            logger.warning(f"Chat session not found: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session '{session_id}' not found.",
            )

        # Step 2: Fetch all messages in chronological order
        result = db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()

        # Step 3: Convert to response models, parsing JSON sources
        message_list = [
            ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=json.loads(msg.sources) if msg.sources else None,
                created_at=msg.created_at.isoformat(),
            )
            for msg in messages
        ]

        logger.info(
            f"Returning {len(message_list)} messages for session {session_id}"
        )

        return ChatHistoryResponse(
            session_id=session.id,
            title=session.title,
            messages=message_list,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Failed to fetch session {session_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}",
        )


# ─────────────────────────────────────────────
# DELETE /chat/sessions/{session_id} — Delete a session
# ─────────────────────────────────────────────

@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Chat Session",
    description=(
        "Delete a chat session and all its associated messages. "
        "This action is irreversible."
    ),
    responses={
        204: {"description": "Session deleted successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Deletion error"},
    },
)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    ## Delete Chat Session
    """
    logger.info(f"DELETE /chat/sessions/{session_id} — Deleting session")

    try:
        # Step 1: Verify the session exists
        result = db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if session is None:
            logger.warning(f"Chat session not found for deletion: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session '{session_id}' not found.",
            )

        # Step 2: Delete all messages in this session
        db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id)
        )

        # Step 3: Delete the session itself
        db.delete(session)
        db.commit()

        logger.info(f"✅ Deleted chat session: {session_id}")
        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Failed to delete session {session_id}: {str(e)}", exc_info=True
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat session: {str(e)}",
        )
