"""
app/services/auth_service.py
=============================
Simple API key authentication for protecting endpoints.

How it works:
- Clients send their API key in the "X-API-Key" HTTP header.
- This service compares it against settings.api_key_auth.
- If settings.api_key_auth is empty (""), auth is DISABLED — all requests pass.
- If set, requests without a valid key get a 401 Unauthorized response.

Usage in routers:
    from app.services.auth_service import verify_api_key

    @router.post("/protected", dependencies=[Depends(verify_api_key)])
    async def protected_endpoint():
        ...

    # Or inject it to get the key value:
    @router.post("/protected")
    async def protected_endpoint(api_key: str = Depends(verify_api_key)):
        ...

Design notes:
- Uses FastAPI's APIKeyHeader scheme, which auto-documents the auth
  in Swagger UI (shows the lock icon and "Authorize" button).
- The auto_error=False on APIKeyHeader means WE handle the missing-key
  error ourselves (for better error messages).
"""

import logging
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# API KEY HEADER SCHEME
# ─────────────────────────────────────────────

# This tells FastAPI to look for the "X-API-Key" header.
# auto_error=False means it returns None instead of raising
# an automatic 403 — we raise our own 401 with a better message.
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authentication. Leave empty if auth is disabled.",
)


# ─────────────────────────────────────────────
# AUTH DEPENDENCY
# ─────────────────────────────────────────────

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency that validates the X-API-Key header.

    Behavior:
    - If settings.api_key_auth is empty → auth is DISABLED, always passes.
    - If settings.api_key_auth is set:
        - Missing header → 401 Unauthorized
        - Wrong key → 401 Unauthorized
        - Correct key → passes, returns the key

    Args:
        api_key: The value from the X-API-Key header (injected by FastAPI).
                 Will be None if the header is missing.

    Returns:
        The API key string if auth passes, or None if auth is disabled.

    Raises:
        HTTPException: 401 Unauthorized if the key is missing or invalid.
    """
    settings = get_settings()

    # ── Auth disabled — let everything through ──
    if not settings.api_key_auth:
        logger.debug("API key auth is disabled (api_key_auth is empty)")
        return None

    # ── Auth enabled — validate the key ──────
    if not api_key:
        logger.warning("Request missing X-API-Key header (auth is enabled)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Unauthorized",
                "message": "Missing API key. Include 'X-API-Key' header in your request.",
            },
        )

    # Constant-time comparison to prevent timing attacks.
    # (import hmac for this if needed, but for simple API keys
    # a basic comparison is fine — this isn't password auth.)
    if api_key != settings.api_key_auth:
        logger.warning(f"Invalid API key provided: '{api_key[:4]}...'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Unauthorized",
                "message": "Invalid API key. Check your X-API-Key header value.",
            },
        )

    logger.debug("API key validation successful")
    return api_key


# Alias for compatibility with router imports
get_api_key_dependency = verify_api_key

