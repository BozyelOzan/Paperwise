"""
Core retrieval module.

Design principles:
  - Pure query function: vector in → results out
  - No global state, client injected
  - Returns plain dicts (TypedDict), not Pydantic models
  - Score threshold filtering built-in
"""

from typing import TypedDict

from qdrant_client import QdrantClient

_MIN_CHUNK_SCORE = 0.45
_MIN_ABSTRACT_SCORE = 0.50


class RetrievalResult(TypedDict):
    article_id: str
    chunk_index: int
    content: str
    score: float


class RetrievalError(Exception):
    pass


def retrieve(
    query_vector: list[float],
    collection: str,
    client: QdrantClient,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievalResult]:
    """
    Search a Qdrant collection for the most similar vectors.
    Filters results below the minimum score threshold.

    Args:
        query_vector: Embedded query vector.
        collection: Qdrant collection name.
        client: QdrantClient instance (injected).
        top_k: Number of results to return.
        min_score: Minimum similarity score threshold (default: 0.0).

    Returns:
        List of RetrievalResult sorted by score descending,
        filtered by min_score.

    Raises:
        RetrievalError: If the query fails.
    """
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RetrievalResult(
                article_id=r.payload["article_id"],
                chunk_index=r.payload.get("chunk_index", 0),
                content=r.payload["content"],
                score=r.score,
            )
            for r in results
            if r.score >= min_score
        ]
    except Exception as e:
        raise RetrievalError(f"Retrieval failed: {e}") from e
