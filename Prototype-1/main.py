"""
Paperwise — Prototype 1

End-to-end RAG pipeline for arXiv papers:
    1. Search arXiv by topic (top 20 results)
    2. Randomly select 3 papers and download PDFs
    3. Extract, clean, and chunk text
    4. Embed chunks with OpenAI text-embedding-3-small
    5. Store vectors in Qdrant (in-memory)
    6. Interactive query loop: retrieve → generate → answer
"""

import random

from chunker import chunk_multiple
from embedder import embed_multiple
from fetcher import fetch_multiple
from generator import generate
from retriever import retrieve
from search import SearchError, search_arxiv
from vectorstore import COLLECTION, get_client, init_collection, save_embeddings


def main() -> None:
    topic = input("Enter topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    try:
        # 1. Search
        search_result = search_arxiv(topic)
        print(f"\nQuery  : {search_result.query}")
        print(f"Found  : {search_result.total} papers")

        # 2. Select 3 random papers
        k = min(3, len(search_result.papers))
        selected = random.sample(search_result.papers, k=k)
        print("\nSelected papers:")
        for p in selected:
            print(f"  #{p.rank} [{p.id}] {p.title[:70]}")

        # 3. Fetch PDFs
        fetch_result = fetch_multiple([p.id for p in selected])
        print(
            f"\nFetch  → success: {fetch_result.success} | "
            f"failed: {fetch_result.failed or 'none'}"
        )

        if not fetch_result.articles:
            print("No articles fetched. Exiting.")
            return

        # 4. Chunk
        chunk_results = chunk_multiple(fetch_result.articles)
        total_chunks = sum(len(r.chunks) for r in chunk_results)
        print(f"Chunk  → {len(chunk_results)} articles | {total_chunks} chunks")

        # 5. Embed
        embedded = embed_multiple(chunk_results)
        print(f"Embed  → {len(embedded)} vectors")

        # 6. Store
        qdrant = get_client()
        init_collection(qdrant)
        save_embeddings(qdrant, embedded)
        print(f"Store  → {len(embedded)} vectors saved to Qdrant")

        # 7. Query loop
        print("\n" + "═" * 60)
        print("Query mode. Type 'q' to exit.")
        print("═" * 60)

        while True:
            query = input("\nQuestion: ").strip()
            if not query or query.lower() == "q":
                break

            chunks = retrieve(query, qdrant, COLLECTION)
            print(f"\nRetrieved {len(chunks)} chunks:")
            for c in chunks:
                print(
                    f"  [{c['article_id']}] "
                    f"score: {c['score']:.3f} | "
                    f"{c['content'][:80].strip()}..."
                )

            answer = generate(query, chunks)
            print(f"\nAnswer:\n{answer}")

    except SearchError as e:
        print(f"Search error: {e}")


if __name__ == "__main__":
    main()
