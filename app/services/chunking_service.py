"""
app/services/chunking_service.py
================================
Splits extracted PDF text into overlapping chunks for embedding.

Why chunking matters:
- Embedding models have token limits (text-embedding-004 handles ~2048 tokens).
- Smaller chunks = more precise retrieval (a whole page may dilute relevance).
- Overlapping chunks prevent information from being split at boundaries.

Splitting strategy (recursive, from coarsest to finest):
1. Paragraphs  (split on "\\n\\n")
2. Sentences   (split on ". ", "? ", "! ")
3. Words       (split on " ")

Each chunk carries metadata:
    {
        "text":        "The actual chunk text...",
        "page_number": 3,
        "chunk_index": 7,
        "metadata": {
            "page_number": 3,
            "chunk_index": 7,
            "char_count":  487,
        }
    }
"""

import logging
from typing import List, Dict, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Separators for recursive splitting, ordered
# from coarsest to finest granularity.
# ─────────────────────────────────────────────
SEPARATORS = [
    "\n\n",   # Paragraph breaks
    "\n",     # Line breaks
    ". ",     # Sentence endings (period + space)
    "? ",     # Question marks
    "! ",     # Exclamation marks
    "; ",     # Semicolons
    ", ",     # Commas
    " ",      # Individual words (last resort)
]


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def chunk_text(
    pages: List[Dict],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict]:
    """
    Split page-level text into overlapping chunks for embedding.

    Args:
        pages:         List of dicts from pdf_service, each with 'page_number' and 'text'.
        chunk_size:    Maximum characters per chunk. Defaults to settings.chunk_size (1000).
        chunk_overlap: Number of overlapping characters between consecutive chunks.
                       Defaults to settings.chunk_overlap (200).

    Returns:
        List of chunk dicts, each containing:
            - text (str):         The chunk text
            - page_number (int):  Source page number (1-indexed)
            - chunk_index (int):  Global chunk index (0-indexed across all pages)
            - metadata (dict):    Additional metadata for storage

    Raises:
        ValueError: If pages list is empty or contains no text.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # ── Validation ────────────────────────────
    if not pages:
        logger.error("chunk_text() received an empty pages list")
        raise ValueError("Cannot chunk empty pages list.")

    if chunk_overlap >= chunk_size:
        logger.warning(
            f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). "
            f"Reducing overlap to chunk_size // 4 = {chunk_size // 4}"
        )
        chunk_overlap = chunk_size // 4

    logger.info(
        f"Chunking {len(pages)} pages with chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    all_chunks: List[Dict] = []
    global_chunk_index = 0

    for page in pages:
        page_number = page.get("page_number", 0)
        page_text = page.get("text", "").strip()

        # ── Skip empty pages ──────────────────
        if not page_text:
            logger.debug(f"Skipping empty page {page_number}")
            continue

        # ── If the entire page fits in one chunk, no splitting needed ──
        if len(page_text) <= chunk_size:
            chunk = _build_chunk(
                text=page_text,
                page_number=page_number,
                chunk_index=global_chunk_index,
            )
            all_chunks.append(chunk)
            global_chunk_index += 1
            continue

        # ── Recursive split for pages longer than chunk_size ──
        page_chunks = _recursive_split(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text_str in page_chunks:
            chunk = _build_chunk(
                text=chunk_text_str,
                page_number=page_number,
                chunk_index=global_chunk_index,
            )
            all_chunks.append(chunk)
            global_chunk_index += 1

    # ── Final validation ──────────────────────
    if not all_chunks:
        logger.error("Chunking produced zero chunks from the given pages")
        raise ValueError("No chunks could be created from the provided pages.")

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(pages)} pages "
        f"(avg {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars/chunk)"
    )
    return all_chunks


# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _build_chunk(text: str, page_number: int, chunk_index: int) -> Dict:
    """
    Create a standardized chunk dict with metadata.

    Args:
        text:        The chunk text content.
        page_number: Source page number (1-indexed).
        chunk_index: Global chunk index (0-indexed).

    Returns:
        Dict with text, page_number, chunk_index, and metadata sub-dict.
    """
    return {
        "text": text,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "metadata": {
            "page_number": page_number,
            "chunk_index": chunk_index,
            "char_count": len(text),
        },
    }


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separator_index: int = 0,
) -> List[str]:
    """
    Recursively split text into chunks, trying coarser separators first.

    The algorithm:
    1. Try splitting on the current separator (e.g. "\\n\\n" for paragraphs).
    2. Merge the resulting pieces into chunks that fit within chunk_size.
    3. If any merged chunk is still too large, recurse with the next
       finer separator (e.g. "\\n" for lines, then ". " for sentences).

    Args:
        text:            The text to split.
        chunk_size:      Maximum characters per chunk.
        chunk_overlap:   Overlap between consecutive chunks.
        separator_index: Current index into the SEPARATORS list.

    Returns:
        List of text strings, each within chunk_size (best effort).
    """
    # ── Base case: no more separators to try ──
    # Just hard-cut the text at chunk_size boundaries.
    if separator_index >= len(SEPARATORS):
        return _hard_split(text, chunk_size, chunk_overlap)

    separator = SEPARATORS[separator_index]

    # ── If this separator doesn't appear in the text, try the next one ──
    if separator not in text:
        return _recursive_split(text, chunk_size, chunk_overlap, separator_index + 1)

    # ── Split on this separator ───────────────
    pieces = text.split(separator)

    # ── Merge pieces into chunks that fit ─────
    chunks: List[str] = []
    current_chunk = ""

    for piece in pieces:
        # What the chunk would look like if we added this piece
        if current_chunk:
            candidate = current_chunk + separator + piece
        else:
            candidate = piece

        if len(candidate) <= chunk_size:
            # Still fits — keep building the chunk
            current_chunk = candidate
        else:
            # Adding this piece would exceed chunk_size.
            # Save the current chunk (if non-empty).
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If the piece itself is too long, recurse with a finer separator
            if len(piece) > chunk_size:
                sub_chunks = _recursive_split(
                    piece, chunk_size, chunk_overlap, separator_index + 1
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = piece

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # ── Apply overlap ─────────────────────────
    # Overlap means the end of chunk N appears at the start of chunk N+1.
    # This prevents information loss at chunk boundaries.
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    Add overlapping text between consecutive chunks.

    For each chunk after the first, we prepend the last `overlap` characters
    of the previous chunk. This creates a sliding window effect.

    Args:
        chunks:  List of chunk strings (no overlap yet).
        overlap: Number of characters to overlap.

    Returns:
        New list of chunks with overlap applied.
    """
    if len(chunks) <= 1:
        return chunks

    overlapped: List[str] = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        # Take the last `overlap` characters from the previous chunk
        overlap_text = prev_chunk[-overlap:]

        # Try to start the overlap at a word boundary to avoid
        # cutting words in half. Find the first space in the overlap.
        space_idx = overlap_text.find(" ")
        if space_idx != -1:
            overlap_text = overlap_text[space_idx + 1:]

        # Prepend the overlap to the current chunk
        overlapped_chunk = overlap_text + " " + current_chunk
        overlapped.append(overlapped_chunk.strip())

    return overlapped


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Last-resort splitting: cut text at exact character boundaries.

    Used when no separator can break the text small enough.
    Tries to split at word boundaries when possible.

    Args:
        text:       The text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Overlap between chunks.

    Returns:
        List of text strings.
    """
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunks.append(text[start:].strip())
            break

        # Try to break at a word boundary (look backwards from `end`)
        # Search for the last space within the chunk
        last_space = text.rfind(" ", start, end)
        if last_space > start:
            end = last_space

        chunks.append(text[start:end].strip())

        # Move start forward, accounting for overlap
        start = end - overlap if overlap > 0 else end

    return [c for c in chunks if c]  # Filter out empty strings
