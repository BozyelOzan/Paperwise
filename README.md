# Paperwise

An iterative RAG system that answers scientific questions by retrieving and reasoning over arXiv papers in real time.

## Overview

Paperwise takes a natural language question, analyzes it to extract relevant search terms, retrieves matching papers from arXiv, and generates a grounded answer using a RAG pipeline. The project was developed across three prototypes, each addressing limitations of the previous one — evolving from a single-script baseline to a production-style distributed service.

```
User Question
     │
     ▼
┌─────────────────────┐
│   Query Analysis    │  LLM extracts search terms + arXiv-style query
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   arXiv Search      │  Up to 80 unique English papers retrieved
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Abstract Ranking   │  Embed abstracts → semantic top-3 selection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   PDF Processing    │  Extract, clean, chunk full text (Redis cache)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Chunk Retrieval    │  Top-5 chunks from vector store
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Answer Generation  │  Grounded response via gpt-4o-mini
└─────────────────────┘
```

## Prototypes

The system was built iteratively. Each prototype is self-contained with its own README.

|                      | [Prototype 1](./prototype_1/) | [Prototype 2](./prototype_2/)      | [Prototype 3](./prototype_3/)   |
| -------------------- | ----------------------------- | ---------------------------------- | ------------------------------- |
| **Focus**            | Baseline pipeline             | Semantic paper selection           | Production architecture         |
| **Paper selection**  | Random                        | Semantic (abstract embeddings)     | Semantic                        |
| **Candidate papers** | 20                            | Up to 80                           | Up to 80                        |
| **Vector store**     | Qdrant in-memory              | Qdrant in-memory (dual collection) | Qdrant persistent (Docker)      |
| **PDF cache**        | —                             | —                                  | Redis (7-day TTL)               |
| **Architecture**     | Single process                | Single process                     | FastAPI + RabbitMQ + Worker     |
| **Concurrency**      | Single user                   | Single user                        | Multi-user (horizontal scaling) |
| **UI**               | CLI                           | CLI                                | Chainlit (streaming)            |
| **Mean chunk score** | 0.421                         | 0.610                              | —                               |
| **Judge score**      | 4.4 / 5                       | 5.0 / 5                            | —                               |

## Stack

| Component          | Technology                     |
| ------------------ | ------------------------------ |
| LLM                | gpt-4o-mini                    |
| Embeddings         | text-embedding-3-small         |
| Vector store       | Qdrant                         |
| Message queue      | RabbitMQ                       |
| Cache              | Redis                          |
| API                | FastAPI                        |
| UI                 | Chainlit                       |
| PDF extraction     | pdfplumber                     |
| Chunking           | chonkie TokenChunker           |
| Query analysis     | gpt-4o-mini (json_object mode) |
| Language detection | langdetect                     |

## Status

Active prototype. Prototype 3 is the most complete iteration. Known limitations across prototypes:

- PDF text quality depends on pdfplumber extraction
- No reranker — retrieval relies solely on embedding similarity
- Abstract similarity threshold may filter out valid papers on broad queries

## Getting Started

See the README inside each prototype directory for setup instructions. Prototype 3 is recommended as the starting point.

```bash
cd prototype_3/
# Follow README instructions to start Docker services, worker, API, and UI
```

## Requirements

- Python 3.10+
- Docker (Prototype 3)
- OpenAI API key
