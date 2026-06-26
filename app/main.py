"""
app/main.py
============
The entry point of the FastAPI RAG Document Assistant application.

This file:
1. Creates the FastAPI app instance with comprehensive API documentation
2. Manages the application lifecycle (startup/shutdown)
3. Registers all routers (documents, QA, chat history)
4. Registers global exception handlers for consistent error responses
5. Defines public health-check and root endpoints
6. Configures CORS middleware for development
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import select, func

from app.config import get_settings
from app.database import get_db, create_tables
from app.models.db_models import Document
from app.routers import documents, qa, chat
from app.logger import setup_logger

# ─────────────────────────────────────────────
# SETTINGS & LOGGER
# ─────────────────────────────────────────────

settings = get_settings()
logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# LIFESPAN — Startup & Shutdown events
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle using FastAPI's modern lifespan approach.

    Code BEFORE 'yield' runs at startup:
    - Creates database tables if they don't exist
    - Creates the uploads directory
    - Logs startup information

    Code AFTER 'yield' runs at shutdown:
    - Logs shutdown message
    - Could be used to close DB connections, flush caches, etc.
    """
    # ── STARTUP ──────────────────────────────
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"   Debug mode: {settings.debug}")
    logger.info(f"   Database:   {settings.database_url}")
    logger.info(f"   ChromaDB:   {settings.chroma_dir}")
    logger.info("=" * 60)

    # Create database tables (safe to call multiple times — uses IF NOT EXISTS)
    create_tables()
    logger.info("✅ Database tables created/verified")

    # Ensure the uploads directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"✅ Upload directory ready: {settings.upload_dir}")

    # Log that all services are initialized
    logger.info("✅ All services initialized — API is ready to accept requests!")
    logger.info("=" * 60)

    yield  # ← App runs here (handles requests)

    # ── SHUTDOWN ─────────────────────────────
    logger.info("=" * 60)
    logger.info("🛑 Shutting down RAG Document Assistant...")
    logger.info("   Closing connections and cleaning up resources")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# CREATE FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## IR Infotech RAG Document Assistant 📄🤖

A production-grade **Retrieval-Augmented Generation (RAG)** API that lets you
upload PDF documents and ask questions about them using AI.

### How It Works
1. **Upload** PDF documents → text is extracted, chunked, and embedded
2. **Ask** questions → relevant chunks are retrieved and sent to Gemini AI
3. **Get answers** grounded in your actual documents (not hallucinated!)

| Endpoint | Description |
|----------|-------------|
| `POST /documents/upload` | Upload and process PDF files |
| `GET /documents` | List all uploaded documents |
| `DELETE /documents/{id}` | Remove a document |
| `POST /qa/ask` | Ask a question about your documents |
| `GET /chat/sessions` | View chat history |
| `DELETE /chat/sessions/{id}` | Delete a chat session |

---

### Key Features
- 📄 **PDF Processing** — Automatic text extraction and chunking
- 🧠 **Semantic Search** — ChromaDB vector store for intelligent retrieval
- 🤖 **AI Answers** — Google Gemini generates context-aware responses
- 💬 **Chat Sessions** — Multi-turn conversations with history
- 🔄 **Streaming** — Real-time SSE responses for instant feedback
- 🔑 **API Key Auth** — Simple authentication via `X-API-Key` header

### Authentication
Include your API key in the request header:
```
X-API-Key: your-api-key-here
```

### Error Handling
All errors return a consistent JSON format:
```json
{
    "error": "Error Type",
    "message": "Human-readable description",
    "status_code": 400
}
```
    """,
    contact={
        "name": "Ojas Surana",
        "email": "ojassurana@example.com",
    },
    lifespan=lifespan,
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc UI (alternative docs page)
    openapi_url="/openapi.json",
)


# ─────────────────────────────────────────────
# CORS MIDDLEWARE — Allow cross-origin requests
# ─────────────────────────────────────────────

# In development, we allow all origins so the frontend can talk to the API.
# In production, you'd restrict this to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (restrict in production!)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers (including X-API-Key)
)


# ─────────────────────────────────────────────
# EXCEPTION HANDLERS — Consistent error responses
# ─────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handles validation errors (e.g., missing required fields, wrong data types).

    FastAPI raises RequestValidationError when the request body doesn't
    match the Pydantic model. We format it into a clear error message.
    """
    errors = exc.errors()
    first_error = errors[0] if errors else {}

    # Build a user-friendly error message from the first validation error
    field = " → ".join(str(loc) for loc in first_error.get("loc", []))
    message = first_error.get("msg", "Invalid request data")
    friendly_message = f"Validation error on field '{field}': {message}"

    logger.warning(f"Validation error on {request.url.path}: {friendly_message}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": friendly_message,
            "status_code": 422,
            "details": errors,  # Include all errors for debugging
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handles standard HTTP errors like 404 Not Found, 403 Forbidden, etc.
    Wraps them in our consistent error response format.
    """
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for unexpected errors.

    This is the safety net — if an error slips through all other handlers
    (e.g., ChromaDB is down, Gemini API fails, or there's a bug),
    this ensures the user always gets a proper JSON error response.
    """
    error_message = str(exc)

    logger.error(
        f"Unhandled exception on {request.url.path}: {error_message}",
        exc_info=True,  # Logs the full stack trace
    )

    # Check for Gemini rate limit errors
    if (
        "429" in error_message
        or "RESOURCE_EXHAUSTED" in error_message
        or "quota" in error_message.lower()
    ):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": (
                    "The AI service is temporarily rate-limited. "
                    "Please wait a few seconds and try again."
                ),
                "status_code": 429,
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": f"An unexpected error occurred: {error_message}",
            "status_code": 500,
        },
    )


# ─────────────────────────────────────────────
# REGISTER ROUTERS
# ─────────────────────────────────────────────

# Each router handles a specific group of endpoints.
# Auth is applied per-router (not globally) so health/root stay public.
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(chat.router)


# ─────────────────────────────────────────────
# ROOT & HEALTH CHECK — Public endpoints
# ─────────────────────────────────────────────

@app.get(
    "/",
    tags=["Health"],
    summary="Root",
    description="Welcome endpoint — confirms the API is running.",
)
async def root() -> dict:
    """
    Root endpoint — returns a welcome message and a list of available endpoints.
    Visit this first to make sure everything is working!
    """
    return {
        "message": f"Welcome to {settings.app_name}!",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "documents": {
                "upload": "POST /documents/upload",
                "list": "GET /documents",
                "get": "GET /documents/{document_id}",
                "delete": "DELETE /documents/{document_id}",
            },
            "question_answering": {
                "ask": "POST /qa/ask",
            },
            "chat_history": {
                "list_sessions": "GET /chat/sessions",
                "get_session": "GET /chat/sessions/{session_id}",
                "delete_session": "DELETE /chat/sessions/{session_id}",
            },
        },
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Health check endpoint — used to verify the API and its services are alive.",
)
async def health_check() -> dict:
    """
    Health check endpoint — returns the status of the API and its dependencies.

    Checks:
    - API status (always 'healthy' if this endpoint responds)
    - Document count from the database
    - Vector store connectivity
    """
    logger.info("Health check called")

    health_status = {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "components": {},
    }

    # Check database connectivity and document count
    try:
        from app.database import SessionLocal

        with SessionLocal() as db:
            result = db.execute(select(func.count(Document.id)))
            doc_count = result.scalar() or 0
            health_status["components"]["database"] = {
                "status": "healthy",
                "document_count": doc_count,
            }
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    # Check ChromaDB vector store connectivity
    try:
        from app.services.vector_store import VectorStoreService

        vector_store = VectorStoreService()
        collection_count = vector_store.get_collection_count()
        health_status["components"]["vector_store"] = {
            "status": "healthy",
            "total_vectors": collection_count,
        }
    except Exception as e:
        logger.warning(f"Vector store health check failed: {str(e)}")
        health_status["components"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    return health_status
