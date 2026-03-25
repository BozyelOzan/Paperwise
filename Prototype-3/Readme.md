# Paperwise — Prototype 3

Production-ready service architecture with RabbitMQ, Redis, persistent Qdrant and real-time streaming UI.

## What it does

1. Analyze user question with LLM → extract search terms + arXiv-style query
2. Search arXiv with each term → up to 80 unique English papers
3. Embed abstracts → store in `abstracts` collection
4. Semantic search → select top-3 most relevant papers
5. Download PDFs → extract, clean, and chunk text (Redis cache)
6. Embed chunks → store in persistent `papers` collection
7. Retrieve top-5 chunks → stream grounded answer token by token

## Improvements over P2

|                 | P2                        | P3                                  |
| --------------- | ------------------------- | ----------------------------------- |
| Architecture    | Single process            | FastAPI + RabbitMQ + Worker         |
| Concurrency     | Single user               | Multi-user (horizontal scaling)     |
| Reply routing   | —                         | Per-request exclusive queue (RPC)   |
| Vector store    | In-memory Qdrant          | Docker Qdrant (persistent)          |
| PDF cache       | ✗                         | Redis (7-day TTL)                   |
| Session state   | ✗                         | Redis per user (24h TTL)            |
| Answer delivery | Blocking, complete string | Token-by-token SSE stream           |
| Point IDs       | —                         | Deterministic UUIDs (parallel-safe) |
| JSON from LLM   | No enforcement            | `response_format: json_object`      |

## Limitations

- PDF text quality depends on pdfplumber extraction
- No reranking
- Abstract similarity threshold may filter out valid papers on broad queries

## Setup

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY

# 3. Start worker (terminal 1)
python worker/worker.py

# 4. Start API (terminal 2)
uvicorn api.main:app --port 8000

# 5. Start UI (terminal 3)
chainlit run ui/app.py --port 8080
```

Open `http://localhost:8080` in your browser.

To handle concurrent users, start additional worker processes — RabbitMQ distributes messages automatically:

```bash
python worker/worker.py  # terminal 1
python worker/worker.py  # terminal 2
python worker/worker.py  # terminal 3
```

## Stack

| Component          | Technology                     |
| ------------------ | ------------------------------ |
| Embedding          | text-embedding-3-small         |
| LLM                | gpt-4o-mini                    |
| Vector store       | Qdrant (Docker, persistent)    |
| Message queue      | RabbitMQ                       |
| Cache              | Redis                          |
| API                | FastAPI                        |
| UI                 | Chainlit                       |
| PDF extraction     | pdfplumber                     |
| Chunking           | chonkie TokenChunker           |
| Query analysis     | gpt-4o-mini (json_object mode) |
| Language detection | langdetect                     |

## Environment Variables

```
OPENAI_API_KEY=sk-...
RABBITMQ_URL=amqp://paperwise:paperwise@localhost:5672/
```

## Multi-user simulation

```bash
python simulate_users.py              # 2 concurrent users (default)
python simulate_users.py --users 3   # 3 concurrent users
```
