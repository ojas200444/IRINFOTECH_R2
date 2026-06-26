"""
needful/logger.py
=================
Shared logging configuration for all IR Infotech projects (R1, R2, etc.).

Provides a consistent log format across every module:
    timestamp | LEVEL    | module.name | message

Logs are written to:
  - Console (stdout)  — always, for terminal visibility
  - File (app.log)    — optional, for persistent records

The file handler is wrapped in try/except because serverless platforms
(e.g. Vercel, AWS Lambda) often have read-only filesystems.

Usage:
    from needful.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Something happened")
"""

import logging
import os
import sys


def setup_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the given name.

    Each logger gets two handlers:
      1. Console (stdout) — so you can see logs in the terminal
      2. File (app.log)   — so you have a persistent record on disk

    The log level is read from the LOG_LEVEL environment variable.
    If not set, it defaults to INFO.

    Args:
        name: Usually __name__ of the calling module, e.g. "app.services.gemini"

    Returns:
        A configured logging.Logger instance ready to use.
    """
    logger = logging.getLogger(name)

    # ── Guard: avoid adding duplicate handlers ─────────────────
    # If this logger was already set up (e.g. another import triggered it),
    # just return the existing one. Without this check, every import
    # would add MORE handlers, causing duplicate log lines.
    if logger.handlers:
        return logger

    # ── Log Level ──────────────────────────────────────────────
    # Read from environment variable so it can be changed without code edits.
    # Supports: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # ── Formatter ──────────────────────────────────────────────
    # Consistent format: timestamp | level | module | message
    # Example: 2026-06-25 12:30:00 | INFO     | app.main | Server started
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ────────────────────────────────────────
    # Always present — writes to stdout so logs show in the terminal.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (optional) ────────────────────────────────
    # Saves logs to app.log for persistent storage.
    # Wrapped in try/except because serverless platforms (e.g. Vercel)
    # have a read-only filesystem and will throw an error here.
    try:
        file_handler = logging.FileHandler("app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        # Can't write to filesystem — console-only logging is perfectly fine.
        pass

    return logger
