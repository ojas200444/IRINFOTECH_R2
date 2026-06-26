"""
needful/config.py
=================
Shared configuration loader for all IR Infotech projects (R1, R2, etc.).

Loads environment variables from the project root .env file using
pydantic-settings for automatic validation and type-checking.

The .env file lives at: IR INFOTECH/.env  (two levels up from this file)
This is resolved dynamically using Path(__file__) so it works regardless
of where the app is launched from.

Usage:
    from needful.config import get_settings
    settings = get_settings()
    print(settings.gemini_api_key)
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


# Resolve the path to the project root .env file.
# This file lives at: IR INFOTECH/needful/config.py
# So parent.parent gives us: IR INFOTECH/
# And we append .env to get: IR INFOTECH/.env
_ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings are read from IR INFOTECH/.env at startup.
    If a variable is missing from .env, the default value is used —
    except for gemini_api_key which is REQUIRED and has no default.

    Attributes:
        gemini_api_key: Google Gemini API key (required, no default).
        app_name:       Display name for the application.
        app_version:    Semantic version string (e.g. "1.0.0").
        debug:          Enables debug mode (verbose errors, etc.).
        log_level:      Python logging level (DEBUG, INFO, WARNING, ERROR).
        host:           Server bind address (0.0.0.0 for all interfaces).
        port:           Server port number.
    """

    model_config = ConfigDict(
        # Point pydantic-settings to the project root .env file
        env_file=str(_ENV_FILE_PATH),
        env_file_encoding="utf-8",
    )

    # ── Gemini API Key ─────────────────────────────────────────
    # REQUIRED — the app will refuse to start without this.
    # Get yours free at: https://aistudio.google.com/app/apikey
    gemini_api_key: str

    # ── App Info ───────────────────────────────────────────────
    app_name: str = "IR Infotech AI API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Logging ────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Server ─────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance (singleton pattern).

    lru_cache() ensures the .env file is read only ONCE during the
    entire lifetime of the application — not on every import or request.
    This is both a performance and consistency best practice.

    Returns:
        A fully validated Settings object with all config values.
    """
    return Settings()
