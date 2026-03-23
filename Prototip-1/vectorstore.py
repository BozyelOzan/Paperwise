"""
Qdrant vector store module.

Currently runs in-memory for prototype purposes.
To switch to a persistent Docker-based instance, replace get_client() with:
    QdrantClient(host="localhost", port=6333)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from logger import setup_logger

logger = setup_logger(__name__)

COLLECTION = "arxiv_papers"
_VECTOR_SIZE = 1536  # Dimensionality of text-embedding-3-small


def get_client() -> QdrantClient:
    """
    Return a Qdrant client.

    Prototype: in-memory (data is lost when the process ends).
    Production: replace with QdrantClient(host="localhost", port=6333).
    """
    return QdrantClient(":memory:")


def init_collection(client: QdrantClient) -> None:
    """Create the collection if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection created", extra={"collection": COLLECTION})


def save_embeddings(client: QdrantClient, embedded_chunks: list) -> None:
    """
    Upsert embedded chunks into the Qdrant collection.

    Args:
        client: Active QdrantClient instance.
        embedded_chunks: List of EmbeddedChunk objects.
    """
    points = [
        PointStruct(
            id=i,
            vector=chunk.embedding,
            payload={
                "article_id": chunk.article_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            },
        )
        for i, chunk in enumerate(embedded_chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    logger.info("Embeddings saved", extra={"total": len(points)})
