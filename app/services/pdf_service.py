"""
app/services/pdf_service.py
===========================
Handles all PDF text extraction using PyMuPDF (fitz).

This service provides two ways to extract text from PDFs:
1. From a file path on disk  → extract_text()
2. From raw bytes in memory  → extract_text_from_bytes()

Both return a list of dicts, one per page:
    [{"page_number": 1, "text": "..."}, ...]

Page numbers are 1-indexed (human-friendly) and preserved
throughout the pipeline so the final answer can cite sources
like "Page 3" instead of "Page 2 (0-indexed)".
"""

import fitz  # PyMuPDF — imported as 'fitz' by convention
import logging
from typing import List, Dict

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def extract_text(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file on disk.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts, each containing:
            - page_number (int): 1-indexed page number
            - text (str):        The extracted text for that page

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a valid PDF or contains no text.
    """
    logger.info(f"Extracting text from PDF file: {file_path}")

    try:
        # fitz.open() handles file-not-found and corrupted PDFs internally,
        # but we wrap it for consistent error reporting.
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF file '{file_path}': {e}")
        raise ValueError(f"Could not open PDF file: {e}") from e

    # Delegate to the shared extraction logic
    pages = _extract_pages(doc, source_label=file_path)

    # Close the document to free memory
    doc.close()

    return pages


def extract_text_from_bytes(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from raw PDF bytes (e.g. from an upload).

    This avoids writing to disk — useful when the file is already
    in memory from a FastAPI UploadFile.read().

    Args:
        pdf_bytes: The raw bytes of the PDF file.

    Returns:
        List of dicts with page_number and text (same format as extract_text).

    Raises:
        ValueError: If the bytes don't represent a valid PDF or contain no text.
    """
    logger.info(f"Extracting text from PDF bytes ({len(pdf_bytes):,} bytes)")

    try:
        # stream=pdf_bytes tells PyMuPDF to read from memory, not disk.
        # filetype="pdf" explicitly tells it to treat the bytes as a PDF.
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF from bytes: {e}")
        raise ValueError(f"Could not read PDF from bytes: {e}") from e

    # Delegate to the shared extraction logic
    pages = _extract_pages(doc, source_label="in-memory PDF")

    # Close the document to free memory
    doc.close()

    return pages


def count_pages(file_path: str) -> int:
    """
    Count the number of pages in a PDF without extracting text.

    This is much faster than extract_text() when you only need the page count
    (e.g. for validation or metadata).

    Args:
        file_path: Path to the PDF file.

    Returns:
        Number of pages in the PDF.

    Raises:
        ValueError: If the file cannot be opened as a PDF.
    """
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        logger.info(f"PDF '{file_path}' has {page_count} pages")
        return page_count
    except Exception as e:
        logger.error(f"Failed to count pages in '{file_path}': {e}")
        raise ValueError(f"Could not count PDF pages: {e}") from e


def count_pages_from_bytes(pdf_bytes: bytes) -> int:
    """
    Count the number of pages in a PDF from raw bytes.

    Args:
        pdf_bytes: The raw bytes of the PDF file.

    Returns:
        Number of pages in the PDF.

    Raises:
        ValueError: If the bytes cannot be read as a PDF.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()
        logger.info(f"In-memory PDF has {page_count} pages")
        return page_count
    except Exception as e:
        logger.error(f"Failed to count pages from bytes: {e}")
        raise ValueError(f"Could not count PDF pages: {e}") from e


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _extract_pages(doc: fitz.Document, source_label: str) -> List[Dict]:
    """
    Shared logic to iterate over a PyMuPDF Document and extract
    page-level text.

    Args:
        doc:          An opened fitz.Document instance.
        source_label: A human-readable label for logging (file path or "in-memory PDF").

    Returns:
        List of dicts with page_number and text.

    Raises:
        ValueError: If no text could be extracted from any page.
    """
    total_pages = len(doc)
    logger.info(f"Processing {total_pages} pages from {source_label}")

    if total_pages == 0:
        logger.warning(f"PDF has 0 pages: {source_label}")
        raise ValueError("PDF file contains no pages.")

    pages: List[Dict] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]

        # get_text("text") extracts plain text preserving reading order.
        # Other options: "html", "dict", "blocks" — but plain text is
        # best for RAG since we feed it to an LLM anyway.
        text = page.get_text("text")

        # Clean up the text: strip whitespace, collapse excessive newlines
        text = _clean_text(text)

        if text:
            pages.append({
                "page_number": page_idx + 1,  # 1-indexed for human readability
                "text": text,
            })
        else:
            # Log but don't fail — some pages are images-only or blank.
            logger.debug(
                f"Page {page_idx + 1} of {source_label} has no extractable text "
                f"(may be an image or blank page)"
            )

    # If we got zero pages with text, the PDF is likely scanned images
    if not pages:
        logger.error(f"No text found in any page of: {source_label}")
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may contain only scanned images. "
            "OCR is not currently supported."
        )

    logger.info(
        f"Successfully extracted text from {len(pages)}/{total_pages} pages "
        f"of {source_label}"
    )
    return pages


def _clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace while
    preserving paragraph structure.

    Args:
        text: Raw text from PyMuPDF page extraction.

    Returns:
        Cleaned text string, or empty string if input was only whitespace.
    """
    if not text:
        return ""

    # Strip leading/trailing whitespace from the entire text
    text = text.strip()

    # Replace multiple consecutive newlines with double newline (paragraph break)
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple consecutive spaces with single space
    text = re.sub(r" {2,}", " ", text)

    return text
