"""
Redis cache client.

Two cache namespaces:
  - pdf:{arxiv_id}     → raw PDF bytes (persistent across sessions)
  - session:{user_id}  → session state: indexed article IDs (TTL: 24h)

Railway injects REDIS_URL automatically when the Redis plugin is added.
Falls back to localhost for local development.
"""

import json
import os

import redis
from redis import ConnectionPool

_PDF_TTL = 60 * 60 * 24 * 7  # 7 days
_SESSION_TTL = 60 * 60 * 24  # 24 hours

_pool: ConnectionPool | None = None


def _get_client() -> redis.Redis:
    """Return a Redis client backed by a shared connection pool."""
    global _pool
    if _pool is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            _pool = ConnectionPool.from_url(
                redis_url,
                decode_responses=False,
                max_connections=20,
            )
        else:
            _pool = ConnectionPool(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=False,
                max_connections=20,
            )
    return redis.Redis(connection_pool=_pool)


def get_pdf(arxiv_id: str) -> bytes | None:
    """Return cached PDF bytes or None if not cached."""
    return _get_client().get(f"pdf:{arxiv_id}")


def set_pdf(arxiv_id: str, content: bytes) -> None:
    """Cache PDF bytes with 7-day TTL."""
    _get_client().setex(f"pdf:{arxiv_id}", _PDF_TTL, content)


def get_session(user_id: str) -> dict:
    """
    Return session state for a user.
    Contains: indexed_article_ids (list of already processed articles).
    """
    raw = _get_client().get(f"session:{user_id}")
    if raw:
        return json.loads(raw)
    return {"indexed_article_ids": []}


def update_session(user_id: str, article_ids: list[str]) -> None:
    """Add new article IDs to session state and reset TTL."""
    session = get_session(user_id)
    existing = set(session["indexed_article_ids"])
    existing.update(article_ids)
    session["indexed_article_ids"] = list(existing)
    _get_client().setex(f"session:{user_id}", _SESSION_TTL, json.dumps(session))


def clear_session(user_id: str) -> None:
    """Delete session state for a user."""
    _get_client().delete(f"session:{user_id}")
