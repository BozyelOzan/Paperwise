"""
Retrieval module.

Embeds the user query and searches the Qdrant collection
for the most semantically similar chunks.
"""

from qdrant_client import QdrantClient

from chunker import Chunk
from embedder import embed_chunks
from logger import setup_logger

logger = setup_logger(__name__)

_TOP_K = 5


def retrieve(query: str, client: QdrantClient, collection: str) -> list[dict]:
    """
    Find the top-K most relevant chunks for a given query.

    Args:
        query: User's natural language question.
        client: Active QdrantClient instance.
        collection: Name of the Qdrant collection to search.

    Returns:
        List of dicts with keys: article_id, chunk_index, content, score.
    """
    logger.info("Retrieval started", extra={"query": query})

    embedded = embed_chunks([Chunk(article_id="query", chunk_index=0, content=query)])
    query_vector = embedded[0].embedding

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=_TOP_K,
    ).points

    chunks = [
        {
            "article_id": r.payload["article_id"],
            "chunk_index": r.payload["chunk_index"],
            "content": r.payload["content"],
            "score": r.score,
        }
        for r in results
    ]

    logger.info("Retrieval completed", extra={"total": len(chunks)})
    return chunks
