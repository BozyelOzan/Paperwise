"""
Answer generation service.

Supports both streaming and non-streaming modes.
Streaming is used by Chainlit UI for real-time token delivery.
"""

import os
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "You are a scientific assistant specializing in research papers. "
    "Answer the user's question based strictly on the provided paper excerpts enclosed in <context> tags. "
    "If the answer cannot be found in the context, state that clearly. "
    "Be concise, precise, and cite the article ID when referring to specific findings."
)


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[{c['article_id']} | chunk {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def generate(query: str, chunks: list[dict]) -> str:
    """
    Generate a complete answer (non-streaming).
    Used by worker for RabbitMQ responses.
    """
    logger.info("Generation started", extra={"query": query, "num_chunks": len(chunks)})
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=_build_messages(query, chunks),
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        logger.info("Generation completed", extra={"query": query})
        return answer
    except Exception as e:
        logger.error("Generation failed", extra={"error": str(e)})
        raise


def generate_stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """
    Generate answer as a stream of tokens.
    Used by Chainlit UI for real-time display.

    Yields:
        Token strings as they are generated.
    """
    logger.info("Stream generation started", extra={"query": query})
    try:
        stream = _client.chat.completions.create(
            model=_MODEL,
            messages=_build_messages(query, chunks),
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
        logger.info("Stream generation completed", extra={"query": query})
    except Exception as e:
        logger.error("Stream generation failed", extra={"error": str(e)})
        raise
