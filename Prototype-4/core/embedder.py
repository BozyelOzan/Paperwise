"""
Core embedding module.

Design principles:
  - Single function: texts in → vectors out
  - No Pydantic, no global client state (client injected)
  - Token usage returned explicitly for cost tracking
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_EMBED_MODEL = "text-embedding-3-small"


class EmbedResult(TypedDict):
    vectors: list[list[float]]
    token_usage: int


class EmbedError(Exception):
    pass


def embed_texts(
    texts: list[str],
    client: OpenAI,
) -> EmbedResult:
    """
    Embed a list of texts using OpenAI's embedding API.

    Args:
        texts: List of text strings to embed.
        client: OpenAI client instance (injected, not global).

    Returns:
        EmbedResult with vectors and token usage count.

    Raises:
        EmbedError: If the API call fails.
    """
    if not texts:
        return EmbedResult(vectors=[], token_usage=0)

    try:
        response = client.embeddings.create(
            model=_EMBED_MODEL,
            input=texts,
        )
        return EmbedResult(
            vectors=[item.embedding for item in response.data],
            token_usage=response.usage.total_tokens,
        )
    except Exception as e:
        raise EmbedError(f"Embedding failed: {e}") from e
