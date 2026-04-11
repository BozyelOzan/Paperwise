"""
FastAPI gateway.

Endpoints:
  POST /query/pipeline  — runs the full pipeline (search, fetch, chunk, embed)
                          via RabbitMQ worker, returns chunks + step_log.
  POST /query/stream    — given question + chunks, streams the LLM answer
                          token by token via SSE (no worker involved).
  GET  /health
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Iterator

import pika
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger
from services.generator import generate_stream

load_dotenv(Path(__file__).parent.parent / ".env")

logger = setup_logger(__name__)

app = FastAPI(title="Paperwise API", version="0.4.0")

_RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://paperwise:paperwise@localhost:5672/")
_QUEUE_IN = "paperwise.query"
_TIMEOUT = 180  # seconds


class PipelineRequest(BaseModel):
    user_id: str
    question: str


class PipelineResponse(BaseModel):
    query_id: str
    status: str
    answer: str
    chunks: list[dict]
    step_log: list[str]


class StreamRequest(BaseModel):
    question: str
    chunks: list[dict]


@app.post("/query/pipeline", response_model=PipelineResponse)
def query_pipeline(request: PipelineRequest):
    """
    Run the full pipeline (search, fetch, chunk, embed, retrieve) via RabbitMQ.
    Uses the RabbitMQ RPC pattern with an exclusive per-request reply queue.
    """
    query_id = str(uuid.uuid4())
    logger.info(
        "Pipeline request received",
        extra={"user_id": request.user_id, "query_id": query_id},
    )

    try:
        connection = pika.BlockingConnection(pika.URLParameters(_RABBITMQ_URL))
        channel = connection.channel()

        channel.queue_declare(queue=_QUEUE_IN, durable=True)

        result_queue = channel.queue_declare(queue="", exclusive=True, auto_delete=True)
        reply_queue_name = result_queue.method.queue

        channel.basic_publish(
            exchange="",
            routing_key=_QUEUE_IN,
            body=json.dumps(
                {
                    "user_id": request.user_id,
                    "query_id": query_id,
                    "question": request.question,
                }
            ),
            properties=pika.BasicProperties(
                delivery_mode=2,
                correlation_id=query_id,
                reply_to=reply_queue_name,
            ),
        )

        result = None
        for method, properties, body in channel.consume(
            queue=reply_queue_name,
            auto_ack=True,
            inactivity_timeout=_TIMEOUT,
        ):
            if method is None:
                break
            msg = json.loads(body)
            if msg.get("query_id") == query_id:
                result = msg
                break

        connection.close()

        if result is None:
            raise HTTPException(status_code=504, detail="Worker timeout")

        return PipelineResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pipeline API error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def query_stream(request: StreamRequest):
    """
    Stream the LLM-generated answer token by token (Server-Sent Events).
    Accepts question + retrieved chunks directly — no worker involved.
    """
    logger.info(
        "Stream request received",
        extra={"question": request.question, "num_chunks": len(request.chunks)},
    )

    def token_generator() -> Iterator[str]:
        try:
            for token in generate_stream(request.question, request.chunks):
                yield token
        except Exception as e:
            logger.error("Stream generation error", extra={"error": str(e)})
            yield f"\n\n[Generation error: {e}]"

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4.0"}
