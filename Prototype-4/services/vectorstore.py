"""Qdrant vector store service.

Connects to Qdrant Cloud when QDRANT_URL + QDRANT_API_KEY are set,
falls back to localhost for local development.
"""

import hashlib
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from logger import setup_logger

logger = setup_logger(__name__)

ABSTRACTS_COLLECTION = "abstracts"
PAPERS_COLLECTION = "papers"
_VECTOR_SIZE = 1536


def get_client() -> QdrantClient:
    """Connect to Qdrant Cloud or local Docker instance."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        return QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection created", extra={"collection": name})

    client.create_payload_index(
        collection_name=name,
        field_name="article_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def init_collections(client: QdrantClient) -> None:
    _ensure_collection(client, ABSTRACTS_COLLECTION)
    _ensure_collection(client, PAPERS_COLLECTION)


def _make_point_id(article_id: str, chunk_index: int = 0) -> str:
    raw = f"{article_id}:{chunk_index}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return str(uuid.UUID(digest))


def _article_already_indexed(
    client: QdrantClient, collection: str, article_id: str
) -> bool:
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
