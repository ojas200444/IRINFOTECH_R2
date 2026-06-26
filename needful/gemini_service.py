"""
needful/gemini_service.py
=========================
Shared Gemini AI wrapper for all IR Infotech projects (R1, R2, etc.).

This is the heart of AI communication — it handles all interaction with
the Google Gemini API using the official google-genai SDK.

Think of it as a "wrapper" around Gemini that makes it easy for any
service (summarize, translate, RAG, etc.) to call without worrying
about the API details, retry logic, or rate-limit handling.

Usage:
    from needful.gemini_service import gemini_service

    response = await gemini_service.generate("Explain quantum computing")
    print(response)
"""

import asyncio
from google import genai
from google.genai import types
from needful.config import get_settings
from needful.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class GeminiService:
    """
    Handles all interactions with the Google Gemini API.

    This class is initialized once and reused across all requests
    (singleton pattern). It wraps the google-genai SDK with:
      - Automatic retry on rate-limit (429) and unavailable (503) errors
      - Exponential backoff to respect API quotas
      - Consistent generation parameters across the entire application
    """

    def __init__(self) -> None:
        """
        Set up the Gemini client with our API key and default config.

        The client is created using the API key from settings (.env),
        and the generation config controls how the AI produces text.
        """
        logger.info("Initializing Gemini AI service...")

        # Create the Gemini client with our API key
        self.client = genai.Client(api_key=settings.gemini_api_key)

        # We use gemini-2.5-flash — latest, fast, capable, and FREE
        self.model_name = "gemini-2.5-flash"

        # Generation config — controls how the AI produces text
        # temperature: Controls creativity (0 = robotic, 1 = very creative)
        # top_p:       Controls diversity of word choices
        # max_output_tokens: Maximum length of the response
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048,
        )

        logger.info(f"Gemini AI service initialized. Model: {self.model_name}")

    async def generate(self, prompt: str, retries: int = 5) -> str:
        """
        Sends a prompt to Gemini and returns the text response.

        Includes automatic retry logic with exponential backoff for
        rate-limit (429) and service-unavailable (503) errors. This is
        important because Gemini's free tier has strict rate limits.

        Args:
            prompt:  The instruction or question to send to the AI.
            retries: Maximum number of retry attempts on transient errors.
                     Defaults to 5, which gives ~62 seconds of total wait.

        Returns:
            The AI's response as a plain string, stripped of whitespace.

        Raises:
            ValueError: If the AI returns an empty or blocked response.
            Exception:  For any unrecoverable Gemini API errors.
        """
        logger.debug(f"Sending prompt to Gemini (length: {len(prompt)} chars)")

        for attempt in range(1, retries + 1):
            try:
                # ── Call Gemini API asynchronously ─────────────────
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )

                # ── Validate response ─────────────────────────────
                # Sometimes Gemini returns empty responses (safety filters,
                # content policy, etc.) — we treat this as an error.
                if not response.text:
                    logger.warning("Gemini returned an empty response.")
                    raise ValueError(
                        "The AI returned an empty response. Please try again."
                    )

                logger.debug(
                    f"Received response from Gemini (length: {len(response.text)} chars)"
                )
                return response.text.strip()

            except ValueError:
                # Re-raise our custom ValueError immediately (no retry).
                # These are "permanent" errors — retrying won't help.
                raise

            except Exception as e:
                error_str = str(e)

                # ── Check if this is a retryable error ────────────
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str

                if (is_rate_limit or is_unavailable) and attempt < retries:
                    # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                    wait = 2 ** attempt
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{retries}). "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue

                # ── Non-retryable error or exhausted retries ──────
                logger.error(f"Gemini API error: {error_str}", exc_info=True)
                raise Exception(f"AI service error: {error_str}")


# ── Singleton Instance ─────────────────────────────────────────────
# Create a single shared instance of GeminiService.
# This is the "singleton pattern" — one instance, shared everywhere.
# Every module imports this instead of creating its own.
gemini_service = GeminiService()
