"""
OpenAI embedding module.

Uses text-embedding-3-small to convert text chunks into dense vectors.
API key is loaded from the OPENAI_API_KEY environment variable (.env supported).
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddedChunk(BaseModel):
    article_id: str
    chunk_index: int
    content: str
    embedding: list[float]


class EmbedError(Exception):
    pass


def embed_chunks(chunks: list) -> list[EmbeddedChunk]:
    """
    Embed a list of Chunk objects using OpenAI's embedding API.

    Args:
        chunks: List of Chunk objects with .content, .article_id, .chunk_index.

    Returns:
        List of EmbeddedChunk objects with embedding vectors.

    Raises:
        EmbedError: If the API call fails.
    """
    if not chunks:
        return []

    texts = [c.content for c in chunks]
    logger.info("Embedding started", extra={"total": len(texts)})

    try:
        response = _client.embeddings.create(model=_EMBED_MODEL, input=texts)
        logger.info("Embedding completed", extra={"total": len(texts)})

        return [
            EmbeddedChunk(
                article_id=chunks[i].article_id,
                chunk_index=chunks[i].chunk_index,
                content=chunks[i].content,
                embedding=item.embedding,
            )
            for i, item in enumerate(response.data)
        ]
    except Exception as e:
        logger.error("Embedding failed", extra={"error": str(e)})
        raise EmbedError(f"Embedding failed: {e}") from e


def embed_multiple(chunk_results: list) -> list[EmbeddedChunk]:
    """
    Flatten multiple ChunkResult objects and embed all chunks in a single API call.

    Args:
        chunk_results: List of ChunkResult objects.

    Returns:
        List of EmbeddedChunk objects.
    """
    all_chunks = [chunk for result in chunk_results for chunk in result.chunks]
    return embed_chunks(all_chunks)
