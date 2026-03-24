"""
PDF text extraction and chunking module.

Pipeline per article:
    PDF bytes → plain text (pdfplumber)
             → language check (English only — article level)
             → clean text (remove references, noise)
             → token-based chunks (chonkie TokenChunker)
             → language check (English only — chunk level)
"""

import io
import logging
import re
import warnings
from collections import Counter

import pdfplumber
from chonkie import TokenChunker
from langdetect import LangDetectException, detect
from pydantic import BaseModel

from logger import setup_logger

# Suppress pdfplumber font warnings
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

logger = setup_logger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 50
_TOKENIZER = "cl100k_base"
_MIN_CHUNK_LENGTH = 50  # discard chunks shorter than this


class Chunk(BaseModel):
    article_id: str
    chunk_index: int
    content: str


class ChunkResult(BaseModel):
    article_id: str
    chunks: list[Chunk]


class ChunkError(Exception):
    pass


def _is_english(text: str) -> bool:
    """Return True if the text is detected as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def pdf_to_text(content: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        raise ChunkError(f"PDF to text conversion failed: {e}") from e


def clean_text(text: str) -> str:
    """
    Remove low-signal content from extracted PDF text.

    - Strips reference / bibliography sections.
    - Drops lines that are purely numeric or symbolic.
    - Removes repeated short lines (headers / footers).
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


def _is_valid_chunk(text: str) -> bool:
    """
    Return True if chunk is worth embedding.

    Filters out:
      - Chunks shorter than minimum length
      - Non-English chunks
      - Chunks with too many non-ASCII characters (formulas, symbols)
    """
    if len(text.strip()) < _MIN_CHUNK_LENGTH:
        return False

    # Non-ASCII ratio check — too many symbols = formula/noise chunk
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.15:
        return False

    # Language check on chunk level
    if not _is_english(text[:200]):
        return False

    return True


def chunk_article(article) -> ChunkResult:
    """
    Convert a single Article (PDF bytes) into a list of text chunks.

    Args:
        article: An Article object with .id and .content (bytes).

    Returns:
        ChunkResult containing all valid English chunks for the article.

    Raises:
        ChunkError: If extraction or language check fails.
    """
    logger.info("Chunking started", extra={"article_id": article.id})

    try:
        text = pdf_to_text(article.content)

        # Article-level language check
        if not _is_english(text[:500]):
            raise ChunkError(f"{article.id} is not in English, skipping")

        text = clean_text(text)

        chunker = TokenChunker(
            tokenizer=_TOKENIZER,
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )

        raw_chunks = chunker.chunk(text)

        # Chunk-level validation
        chunks = []
        skipped = 0
        for i, c in enumerate(raw_chunks):
            if _is_valid_chunk(c.text):
                chunks.append(
                    Chunk(article_id=article.id, chunk_index=i, content=c.text)
                )
            else:
                skipped += 1

        logger.info(
            "Chunking completed",
            extra={
                "article_id": article.id,
                "total_chunks": len(chunks),
                "skipped_chunks": skipped,
            },
        )
        return ChunkResult(article_id=article.id, chunks=chunks)

    except Exception as e:
        logger.error(
            "Chunking failed",
            extra={"article_id": article.id, "error": str(e)},
        )
        raise ChunkError(f"Failed to chunk {article.id}: {e}") from e


def chunk_multiple(articles: list) -> list[ChunkResult]:
    """
    Chunk multiple articles. Failed and non-English articles are skipped.

    Args:
        articles: List of Article objects.

    Returns:
        List of ChunkResult for successfully processed English articles.
    """
    results = []
    for article in articles:
        try:
            results.append(chunk_article(article))
        except ChunkError:
            continue
    return results
