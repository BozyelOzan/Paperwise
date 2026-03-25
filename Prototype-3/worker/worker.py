"""
RabbitMQ worker — session-based pipeline consumer.

Consumes query messages from RabbitMQ, runs the full P2 pipeline,
and publishes results back to the per-request reply queue specified
in the message's reply_to property (RPC pattern).

Message format (incoming):
  {
    "user_id": str,
    "query_id": str,
    "question": str
  }

Message format (outgoing):
  {
    "query_id": str,
    "status": "ok" | "no_results" | "error",
    "answer": str,
    "chunks": list[RetrievalResult],
    "step_log": list[str]
  }
"""

import json
import os
import sys
from pathlib import Path

import pika
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.redis_client import get_pdf, get_session, set_pdf, update_session
from core.chunker import ChunkError, process_pdf
from core.embedder import EmbedError, embed_texts
from core.retriever import (
    _MIN_ABSTRACT_SCORE,
    _MIN_CHUNK_SCORE,
    RetrievalError,
    retrieve,
)
from logger import setup_logger
from services.fetcher import FetchError, fetch_pdf
from services.generator import generate
from services.query_analyzer import QueryAnalysisError, analyze_query
from services.search import SearchError, search_arxiv
from services.vectorstore import (
    ABSTRACTS_COLLECTION,
    PAPERS_COLLECTION,
    init_collections,
    save_abstracts,
    save_chunks,
)

load_dotenv()
logger = setup_logger(__name__)

_RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://paperwise:paperwise@localhost:5672/")
_QUEUE_IN = "paperwise.query"
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_qdrant_client = QdrantClient(host="localhost", port=6333)

init_collections(_qdrant_client)


def _build_abstract_embed_input(papers: list) -> tuple[list[str], list[str]]:
    """Return (texts, article_ids) for abstract embedding."""
    texts = [f"{p.title}\n\n{p.summary}" for p in papers]
    ids = [p.id for p in papers]
    return texts, ids


def run_pipeline(user_id: str, question: str) -> dict:
    step_log = []
    session = get_session(user_id)
    already_indexed = set(session["indexed_article_ids"])

    try:
        # 1. Analyze query
        step_log.append("Analyzing query...")
        try:
            analysis = analyze_query(question)
            terms = analysis.terms
            arxiv_query = analysis.arxiv_query
        except QueryAnalysisError:
            terms = [question]
            arxiv_query = question

        step_log.append(f"Search terms: {terms}")

        # 2. Search arXiv
        step_log.append("Searching arXiv...")
        seen_ids: set[str] = set()
        all_papers = []
        for term in terms:
            try:
                result = search_arxiv(term, max_results=20)
                for p in result.papers:
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        all_papers.append(p)
            except SearchError:
                continue

        step_log.append(f"Found {len(all_papers)} unique papers")

        if not all_papers:
            return {
                "status": "no_results",
                "answer": "No papers found. Try rephrasing your question.",
                "chunks": [],
                "step_log": step_log,
            }

        # 3. Embed abstracts
        step_log.append("Indexing abstracts...")
        texts, ids = _build_abstract_embed_input(all_papers)
        embed_result = embed_texts(texts, _openai_client)
        embedded_abstracts = [
            {
                "article_id": ids[i],
                "content": texts[i],
                "vector": embed_result["vectors"][i],
            }
            for i in range(len(ids))
        ]
        save_abstracts(_qdrant_client, embedded_abstracts)

        # 4. Retrieve top-3 abstracts
        step_log.append("Selecting most relevant papers...")
        query_embed = embed_texts([arxiv_query], _openai_client)
        query_vector = query_embed["vectors"][0]

        top_abstracts = retrieve(
            query_vector=query_vector,
            collection=ABSTRACTS_COLLECTION,
            client=_qdrant_client,
            top_k=3,
            min_score=_MIN_ABSTRACT_SCORE,
        )

        if not top_abstracts:
            return {
                "status": "no_results",
                "answer": (
                    "No relevant papers found above similarity threshold. "
                    "The question may be too vague or off-topic for arXiv. "
                    "Try asking about a more specific academic topic."
                ),
                "chunks": [],
                "step_log": step_log,
            }

        selected_ids = [a["article_id"] for a in top_abstracts]
        step_log.append(f"Selected: {selected_ids}")

        # 5. Fetch PDFs (with Redis cache)
        step_log.append("Fetching papers...")
        new_ids = [aid for aid in selected_ids if aid not in already_indexed]

        for arxiv_id in new_ids:
            cached = get_pdf(arxiv_id)
            if cached:
                pdf_bytes = cached
                logger.info("PDF from cache", extra={"arxiv_id": arxiv_id})
            else:
                try:
                    article = fetch_pdf(arxiv_id)
                    pdf_bytes = article.content
                    set_pdf(arxiv_id, pdf_bytes)
                except FetchError as e:
                    logger.error(
                        "Fetch failed",
                        extra={"arxiv_id": arxiv_id, "error": str(e)},
                    )
                    continue

            try:
                chunks = process_pdf(pdf_bytes, arxiv_id)
                if not chunks:
                    continue

                chunk_texts = [c["content"] for c in chunks]
                chunk_embed = embed_texts(chunk_texts, _openai_client)

                embedded_chunks = [
                    {
                        "article_id": arxiv_id,
                        "chunk_index": chunks[i]["chunk_index"],
                        "content": chunks[i]["content"],
                        "vector": chunk_embed["vectors"][i],
                    }
                    for i in range(len(chunks))
                ]
                save_chunks(_qdrant_client, embedded_chunks)

            except (ChunkError, EmbedError) as e:
                logger.error(
                    "Processing failed",
                    extra={"arxiv_id": arxiv_id, "error": str(e)},
                )
                continue

        # Update session
        update_session(user_id, new_ids)

        # 6. Retrieve chunks
        step_log.append("Retrieving relevant chunks...")
        question_embed = embed_texts([question], _openai_client)
        question_vector = question_embed["vectors"][0]

        result_chunks = retrieve(
            query_vector=question_vector,
            collection=PAPERS_COLLECTION,
            client=_qdrant_client,
            top_k=5,
            min_score=_MIN_CHUNK_SCORE,
        )

        if not result_chunks:
            return {
                "status": "no_results",
                "answer": (
                    "No relevant content found above similarity threshold. "
                    "The retrieved papers may not directly answer this question. "
                    "Try rephrasing or asking about a more specific topic."
                ),
                "chunks": [],
                "step_log": step_log,
            }

        # 7. Generate answer
        step_log.append("Generating answer...")
        answer = generate(question, result_chunks)
        step_log.append("Done.")

        return {
            "status": "ok",
            "answer": answer,
            "chunks": result_chunks,
            "step_log": step_log,
        }

    except Exception as e:
        logger.error("Pipeline error", extra={"error": str(e)})
        return {
            "status": "error",
            "answer": f"Pipeline error: {e}",
            "chunks": [],
            "step_log": step_log,
        }


def on_message(ch, method, properties, body):
    msg = json.loads(body)
    user_id = msg["user_id"]
    query_id = msg["query_id"]
    question = msg["question"]

    # reply_to is set by main.py to the exclusive per-request queue name.
    # If missing (e.g. direct CLI test), fall back to the default result queue.
    reply_to = properties.reply_to or "paperwise.result"

    logger.info("Query received", extra={"user_id": user_id, "query_id": query_id})

    result = run_pipeline(user_id, question)
    result["query_id"] = query_id

    ch.basic_publish(
        exchange="",
        routing_key=reply_to,
        body=json.dumps(result),
        properties=pika.BasicProperties(
            correlation_id=properties.correlation_id,
        ),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)
    logger.info(
        "Result published",
        extra={"query_id": query_id, "status": result["status"], "reply_to": reply_to},
    )


def main():
    connection = pika.BlockingConnection(pika.URLParameters(_RABBITMQ_URL))
    channel = connection.channel()

    channel.queue_declare(queue=_QUEUE_IN, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=_QUEUE_IN, on_message_callback=on_message)

    logger.info("Worker started, waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
