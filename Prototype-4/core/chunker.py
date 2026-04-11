"""
Core chunking module.

Design principles:
  - Pure functions only, no global state
  - Input: bytes (PDF) → Output: list of dicts
  - No framework dependencies
  - Each function does exactly one thing
"""

import io
import logging
import re
import warnings
from collections import Counter
from typing import TypedDict

import pdfplumber
from chonkie import TokenChunker
from langdetect import LangDetectException, detect

logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 50
_TOKENIZER = "cl100k_base"
_MIN_CHUNK_LENGTH = 50


class ChunkDict(TypedDict):
    article_id: str
    chunk_index: int
    content: str


class ChunkError(Exception):
    pass


def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Extracted plain text.

    Raises:
        ChunkError: If extraction fails.
    """
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        raise ChunkError(f"PDF to text failed: {e}") from e


def is_english(text: str) -> bool:
    """
    Detect if text is English.

    Args:
        text: Text sample (first 500 chars is sufficient).

    Returns:
        True if English, False otherwise.
    """
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def clean_text(text: str) -> str:
    """
    Remove low-signal content from extracted PDF text.

    Removes:
      - References / bibliography sections
      - Lines with fewer than 5 meaningful characters
      - Repeated short lines (headers / footers)

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    text = re.sub(r"\nReferences\n.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\nBibliography\n.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    lines = [
        line
        for line in text.split("\n")
        if len(re.sub(r"[\d\s\.\,\[\]\(\)\-\+\=\/\*]+", "", line)) > 5
    ]

    line_counts = Counter(line.strip() for line in lines if len(line.strip()) < 80)
    lines = [
        line
        for line in lines
        if line_counts.get(line.strip(), 0) < 3 or len(line.strip()) >= 80
    ]

    return "\n".join(lines).strip()


def is_valid_chunk(text: str) -> bool:
    """
    Return True if chunk is worth embedding.

    Filters out:
      - Chunks shorter than minimum length
      - Chunks with too many non-ASCII characters (formulas, symbols)
      - Non-English chunks

    Args:
        text: Chunk text to validate.

    Returns:
        True if chunk should be kept, False otherwise.
    """
    if len(text.strip()) < _MIN_CHUNK_LENGTH:
        return False

    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.15:
        return False

    if not is_english(text[:200]):
        return False

    return True


def chunk_text(text: str, article_id: str) -> list[ChunkDict]:
    """
    Split text into overlapping token-based chunks.
    Filters out invalid chunks (non-English, too short, symbol-heavy).

    Args:
        text: Clean plain text.
        article_id: Source article identifier.

    Returns:
        List of valid ChunkDict objects.
    """
    chunker = TokenChunker(
        tokenizer=_TOKENIZER,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    return [
        ChunkDict(
            article_id=article_id,
            chunk_index=i,
            content=c.text,
        )
        for i, c in enumerate(chunker.chunk(text))
        if is_valid_chunk(c.text)
    ]


def process_pdf(pdf_bytes: bytes, article_id: str) -> list[ChunkDict]:
    """
    Full pipeline: PDF bytes → validated chunks.

    Args:
        pdf_bytes: Raw PDF file bytes.
        article_id: Source article identifier.

    Returns:
        List of valid English text chunks ready for embedding.

    Raises:
        ChunkError: If any step fails.
    """
    text = pdf_to_text(pdf_bytes)

    if not is_english(text[:500]):
        raise ChunkError(f"{article_id} is not in English")

    text = clean_text(text)
    return chunk_text(text, article_id)
