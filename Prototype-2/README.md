# Paperwise — Prototype 2

Semantic paper selection via LLM query analysis and dual vector store collections.

## What it does

1. Analyze user question with LLM → extract search terms + arXiv-style query
2. Search arXiv with each term → up to 80 unique English papers
3. Embed abstracts → store in `abstracts` collection
4. Semantic search → select top-3 most relevant papers
5. Download PDFs → extract, clean, and chunk text
6. Embed chunks → store in `papers` collection
7. Retrieve top-5 chunks → generate grounded answer

## Improvements over P1

|                    | P1         | P2                     |
| ------------------ | ---------- | ---------------------- |
| Paper selection    | Random     | Semantic               |
| Search terms       | 1 (manual) | 2-4 (LLM-generated)    |
| Candidate papers   | 20         | up to 80               |
| Qdrant collections | 1          | 2 (abstracts + papers) |
| Language filter    | ✗          | ✓                      |
| Mean chunk score   | 0.421      | 0.610                  |
| Judge score        | 4.4/5      | 5.0/5                  |

## Limitations

- In-memory storage — data lost on exit
- No reranking
- PDF text quality depends on pdfplumber extraction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python main.py
```

## Stack

| Component          | Technology                          |
| ------------------ | ----------------------------------- |
| Embedding          | text-embedding-3-small              |
| LLM                | gpt-4o-mini                         |
| Vector store       | Qdrant (in-memory, dual collection) |
| PDF extraction     | pdfplumber                          |
| Chunking           | chonkie TokenChunker                |
| Query analysis     | gpt-4o-mini (JSON output)           |
| Language detection | langdetect                          |
