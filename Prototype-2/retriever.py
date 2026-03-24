"""
Retrieval module.

Provides two retrieval functions:
  - retrieve_abstracts : finds the most relevant paper abstracts using
                         the arXiv-style reformulated query
  - retrieve_chunks    : finds the most relevant full-text chunks,
                         filtered by minimum score threshold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from chunker import Chunk
from embedder import embed_chunks
from logger import setup_logger
from vectorstore import ABSTRACTS_COLLECTION, PAPERS_COLLECTION

load_dotenv()
logger = setup_logger(__name__)

_TOP_ABSTRACTS = 3
_TOP_CHUNKS = 5
_MIN_CHUNK_SCORE = 0.45  # chunks below this score are discarded
_MIN_ABSTRACT_SCORE = 0.50  # abstracts below this score are discarded


def _embed_query(query: str) -> list[float]:
    """Embed a query string and return its vector."""
    embedded = embed_chunks([Chunk(article_id="query", chunk_index=0, content=query)])
    return embedded[0].embedding


def retrieve_abstracts(arxiv_query: str, client: QdrantClient) -> list[dict]:
    """
    Find the top-3 most relevant paper abstracts.
    Filters out abstracts below minimum score threshold.

    Args:
        arxiv_query: Academic-style reformulation of the user's question.
        client: Active QdrantClient instance.

    Returns:
        List of dicts with keys: article_id, content, score.
    """
    logger.info("Abstract retrieval started", extra={"arxiv_query": arxiv_query[:80]})

    query_vector = _embed_query(arxiv_query)
    results = client.query_points(
        collection_name=ABSTRACTS_COLLECTION,
        query=query_vector,
        limit=_TOP_ABSTRACTS,
    ).points

    abstracts = [
        {
            "article_id": r.payload["article_id"],
            "content": r.payload["content"],
            "score": r.score,
        }
        for r in results
        if r.score >= _MIN_ABSTRACT_SCORE
    ]

    if not abstracts:
        logger.warning(
            "No abstracts above score threshold",
            extra={"threshold": _MIN_ABSTRACT_SCORE},
        )
    else:
        logger.info("Abstract retrieval completed", extra={"total": len(abstracts)})

    return abstracts


def retrieve_chunks(query: str, client: QdrantClient) -> list[dict]:
    """
    Find the top-5 most relevant full-text chunks.
    Filters out chunks below minimum score threshold.

    Args:
        query: User's natural language question.
        client: Active QdrantClient instance.

    Returns:
        List of dicts with keys: article_id, chunk_index, content, score.
        Empty list if no chunks meet the minimum score threshold.
    """
    logger.info("Chunk retrieval started", extra={"query": query})

    query_vector = _embed_query(query)
    results = client.query_points(
        collection_name=PAPERS_COLLECTION,
        query=query_vector,
        limit=_TOP_CHUNKS,
    ).points

    chunks = [
        {
            "article_id": r.payload["article_id"],
            "chunk_index": r.payload["chunk_index"],
            "content": r.payload["content"],
            "score": r.score,
        }
        for r in results
        if r.score >= _MIN_CHUNK_SCORE
    ]

    if not chunks:
        logger.warning(
            "No chunks above score threshold",
            extra={"threshold": _MIN_CHUNK_SCORE},
        )
    else:
        logger.info("Chunk retrieval completed", extra={"total": len(chunks)})

    return chunks
