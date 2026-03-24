"""
Answer generation module.

Sends retrieved chunks as context to GPT-4o-mini
and returns a grounded answer to the user's question.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "You are a scientific assistant specializing in research papers. "
    "Answer the user's question based strictly on the provided paper excerpts. "
    "If the answer cannot be found in the context, state that clearly. "
    "Be concise, precise, and cite the article ID when referring to specific findings."
)


def generate(query: str, chunks: list[dict]) -> str:
    """
    Generate an answer to the query using retrieved chunks as context.

    Args:
        query: User's natural language question.
        chunks: List of retrieved chunk dicts (article_id, chunk_index, content, score).

    Returns:
        Generated answer string.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    logger.info("Generation started", extra={"query": query, "num_chunks": len(chunks)})

    context = "\n\n---\n\n".join(
        f"[{c['article_id']} | chunk {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        logger.info("Generation completed", extra={"query": query})
        return answer
    except Exception as e:
        logger.error("Generation failed", extra={"error": str(e)})
        raise
