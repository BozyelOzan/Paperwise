# Paperwise — Prototype 1

Baseline RAG pipeline for arXiv papers.

## What it does

1. Search arXiv by topic (top 20 results)
2. Randomly select 3 papers and download PDFs
3. Extract, clean, and chunk text
4. Embed chunks with OpenAI `text-embedding-3-small`
5. Store vectors in Qdrant (in-memory)
6. Interactive query loop: retrieve → generate → answer

## Limitations

- Paper selection is **random** — irrelevant papers likely
- Single Qdrant collection
- In-memory storage — data lost on exit

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python main.py
```

## Stack

| Component      | Technology             |
| -------------- | ---------------------- |
| Embedding      | text-embedding-3-small |
| LLM            | gpt-4o-mini            |
| Vector store   | Qdrant (in-memory)     |
| PDF extraction | pdfplumber             |
| Chunking       | chonkie TokenChunker   |
