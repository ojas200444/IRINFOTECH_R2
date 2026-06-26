"""
app/database.py
===============
Sets up the SQLite database using SQLAlchemy (synchronous).

Why synchronous instead of async?
    SQLite runs in-process — there is no network I/O, so the benefits of
    async are negligible.  Using synchronous SQLAlchemy keeps the code
    simpler and avoids needing the ``aiosqlite`` driver.

This module provides:
    - ``engine``       — the SQLAlchemy engine connected to rag_assistant.db
    - ``SessionLocal`` — a session factory for creating database sessions
    - ``Base``         — the declarative base all ORM models inherit from
    - ``get_db()``     — a FastAPI dependency that yields a session per request
    - ``create_tables()`` — creates all tables on application startup
"""

import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Load settings ──────────────────────────────────────────
settings = get_settings()

# ── Engine ─────────────────────────────────────────────────
# ``check_same_thread=False`` is required for SQLite when using FastAPI,
# because requests may be handled on different threads.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=settings.debug,  # when True, logs every SQL statement (handy for debugging)
)

# ── Session Factory ────────────────────────────────────────
# ``autocommit=False`` — we manage transactions manually via commit()/rollback().
# ``autoflush=False``  — we flush explicitly, avoiding surprise writes.
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ── Declarative Base ──────────────────────────────────────
# All ORM model classes inherit from this so SQLAlchemy knows about them.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage in a route::

        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...

    The session is automatically closed after the request finishes,
    even if an exception occurs (thanks to the ``finally`` block).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """
    Create all tables defined by ORM models that inherit from ``Base``.

    Called once during application startup (in ``main.py``).
    If the tables already exist, this is a no-op — SQLAlchemy's
    ``create_all`` uses ``CREATE TABLE IF NOT EXISTS`` under the hood.

    NOTE: The ORM models module must be imported BEFORE calling this
    function so that ``Base.metadata`` knows about all the tables.
    """
    # Import models here so Base.metadata is populated before create_all()
    import app.models.db_models  # noqa: F401 — import for side-effects

    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")
