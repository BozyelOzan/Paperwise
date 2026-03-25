"""Qdrant vector store service."""

import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from logger import setup_logger

logger = setup_logger(__name__)

ABSTRACTS_COLLECTION = "abstracts"
PAPERS_COLLECTION = "papers"
_VECTOR_SIZE = 1536


def get_client() -> QdrantClient:
    """Connect to Docker-based Qdrant instance."""
    return QdrantClient(host="localhost", port=6333)


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection created", extra={"collection": name})


def init_collections(client: QdrantClient) -> None:
    _ensure_collection(client, ABSTRACTS_COLLECTION)
    _ensure_collection(client, PAPERS_COLLECTION)


def _make_point_id(article_id: str, chunk_index: int = 0) -> str:
    """
    Derive a stable UUID from article_id + chunk_index.

    The same inputs always produce the same UUID, so concurrent workers
    indexing the same article generate identical point IDs. Qdrant's upsert
    is idempotent for identical IDs, so there is no collision.
    """
    raw = f"{article_id}:{chunk_index}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return str(uuid.UUID(digest))


def _article_already_indexed(
    client: QdrantClient, collection: str, article_id: str
) -> bool:
    """
    Return True if at least one point with this article_id exists.

    Uses a payload filter + scroll with limit=1 — O(1) regardless of
    collection size, unlike fetching all IDs.
    """
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def save_abstracts(client: QdrantClient, embedded_abstracts: list) -> int:
    new = [
        e
        for e in embedded_abstracts
        if not _article_already_indexed(client, ABSTRACTS_COLLECTION, e["article_id"])
    ]
    if not new:
        return 0

    points = [
        PointStruct(
            id=_make_point_id(e["article_id"]),
            vector=e["vector"],
            payload={"article_id": e["article_id"], "content": e["content"]},
        )
        for e in new
    ]
    client.upsert(collection_name=ABSTRACTS_COLLECTION, points=points)
    logger.info("Abstracts saved", extra={"new": len(new)})
    return len(new)


def save_chunks(client: QdrantClient, embedded_chunks: list) -> int:
    # Group by article_id to perform one existence check per article
    article_ids = {e["article_id"] for e in embedded_chunks}
    already_indexed = {
        aid
        for aid in article_ids
        if _article_already_indexed(client, PAPERS_COLLECTION, aid)
    }
    new = [e for e in embedded_chunks if e["article_id"] not in already_indexed]
    if not new:
        return 0

    points = [
        PointStruct(
            id=_make_point_id(e["article_id"], e["chunk_index"]),
            vector=e["vector"],
            payload={
                "article_id": e["article_id"],
                "chunk_index": e["chunk_index"],
                "content": e["content"],
            },
        )
        for e in new
    ]
    client.upsert(collection_name=PAPERS_COLLECTION, points=points)
    logger.info("Chunks saved", extra={"new": len(new)})
    return len(new)
