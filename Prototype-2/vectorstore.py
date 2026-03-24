"""
Qdrant vector store module.

Manages two collections:
  - abstracts : paper abstracts, used for semantic paper selection
  - papers    : full-text chunks, used for answer retrieval

Both collections accumulate across queries within a session.
Data is lost when the process ends (in-memory).
"""

from logger import setup_logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = setup_logger(__name__)

ABSTRACTS_COLLECTION = "abstracts"
PAPERS_COLLECTION = "papers"
_VECTOR_SIZE = 1536  # text-embedding-3-small


def get_client() -> QdrantClient:
    """
    Return an in-memory Qdrant client.
    Production: replace with QdrantClient(host="localhost", port=6333).
    """
    return QdrantClient(":memory:")


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Create collection if it does not exist."""
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection created", extra={"collection": name})


def init_collections(client: QdrantClient) -> None:
    """Initialize both abstracts and papers collections."""
    _ensure_collection(client, ABSTRACTS_COLLECTION)
    _ensure_collection(client, PAPERS_COLLECTION)


def _get_existing_ids(client: QdrantClient, collection: str) -> set[str]:
    """Return the set of article IDs already stored in a collection."""
    results, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        results.extend(batch)
        if offset is None:
            break
    return {r.payload["article_id"] for r in results}


def save_abstracts(client: QdrantClient, embedded_abstracts: list) -> int:
    """
    Save embedded abstracts to the abstracts collection.
    Skips abstracts whose article_id is already stored.

    Args:
        client: Active QdrantClient instance.
        embedded_abstracts: List of EmbeddedChunk objects representing abstracts.

    Returns:
        Number of newly saved abstracts.
    """
    existing = _get_existing_ids(client, ABSTRACTS_COLLECTION)
    new = [e for e in embedded_abstracts if e.article_id not in existing]

    if not new:
        logger.info("No new abstracts to save")
        return 0

    # Use a stable offset based on collection size
    count = client.count(collection_name=ABSTRACTS_COLLECTION).count
    points = [
        PointStruct(
            id=count + i,
            vector=e.embedding,
            payload={
                "article_id": e.article_id,
                "content": e.content,
            },
        )
        for i, e in enumerate(new)
    ]
    client.upsert(collection_name=ABSTRACTS_COLLECTION, points=points)
    logger.info(
        "Abstracts saved",
        extra={"new": len(new), "skipped": len(embedded_abstracts) - len(new)},
    )
    return len(new)


def save_chunks(client: QdrantClient, embedded_chunks: list) -> int:
    """
    Save embedded paper chunks to the papers collection.
    Skips chunks whose article_id is already fully stored.

    Args:
        client: Active QdrantClient instance.
        embedded_chunks: List of EmbeddedChunk objects.

    Returns:
        Number of newly saved chunks.
    """
    existing_article_ids = _get_existing_ids(client, PAPERS_COLLECTION)
    new = [e for e in embedded_chunks if e.article_id not in existing_article_ids]

    if not new:
        logger.info("No new chunks to save")
        return 0

    count = client.count(collection_name=PAPERS_COLLECTION).count
    points = [
        PointStruct(
            id=count + i,
            vector=e.embedding,
            payload={
                "article_id": e.article_id,
                "chunk_index": e.chunk_index,
                "content": e.content,
            },
        )
        for i, e in enumerate(new)
    ]
    client.upsert(collection_name=PAPERS_COLLECTION, points=points)
    logger.info(
        "Chunks saved",
        extra={"new": len(new), "skipped": len(embedded_chunks) - len(new)},
    )
    return len(new)
