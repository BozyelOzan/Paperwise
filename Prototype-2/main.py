"""
Paperwise — Prototype 2
"""

from chunker import Chunk, chunk_multiple
from embedder import embed_chunks, embed_multiple
from fetcher import fetch_multiple
from generator import generate
from logger import setup_logger
from query_analyzer import QueryAnalysisError, analyze_query
from retriever import (
    _MIN_ABSTRACT_SCORE,
    _MIN_CHUNK_SCORE,
    retrieve_abstracts,
    retrieve_chunks,
)
from search import SearchError, search_arxiv
from vectorstore import (
    get_client,
    init_collections,
    save_abstracts,
    save_chunks,
)

logger = setup_logger(__name__)


def build_abstract_chunks(papers: list) -> list:
    return [
        Chunk(
            article_id=p.id,
            chunk_index=0,
            content=f"{p.title}\n\n{p.summary}",
        )
        for p in papers
    ]


def main() -> None:
    print("=" * 60)
    print("Paperwise — Prototype 2")
    print("=" * 60)

    qdrant = get_client()
    init_collections(qdrant)

    while True:
        question = input("\nQuestion (q to quit): ").strip()
        if not question or question.lower() == "q":
            break

        try:
            # 1. Analyze query
            print("\n[1/6] Analyzing query...")
            analysis = analyze_query(question)
            print(f"  Search terms  : {analysis.terms}")
            print(f"  arXiv query   : {analysis.arxiv_query[:100]}...")

            # 2. Search arXiv
            print("\n[2/6] Searching arXiv...")
            seen_ids: set[str] = set()
            all_papers = []
            for term in analysis.terms:
                try:
                    result = search_arxiv(term, max_results=20)
                    for p in result.papers:
                        if p.id not in seen_ids:
                            seen_ids.add(p.id)
                            all_papers.append(p)
                except SearchError as e:
                    logger.warning(
                        "Search failed for term",
                        extra={"term": term, "error": str(e)},
                    )

            print(
                f"  Found {len(all_papers)} unique papers "
                f"across {len(analysis.terms)} terms"
            )

            if not all_papers:
                print("  No papers found. Try a different question.")
                continue

            # 3. Embed abstracts → save
            print("\n[3/6] Indexing abstracts...")
            abstract_chunks = build_abstract_chunks(all_papers)
            embedded_abstracts = embed_chunks(abstract_chunks)
            saved = save_abstracts(qdrant, embedded_abstracts)
            print(
                f"  {saved} new abstracts indexed "
                f"({len(embedded_abstracts) - saved} already known)"
            )

            # 4. Retrieve top-3 abstracts
            print("\n[4/6] Selecting most relevant papers...")
            top_abstracts = retrieve_abstracts(analysis.arxiv_query, qdrant)

            if not top_abstracts:
                print(
                    f"  ⚠️  No papers found above score threshold "
                    f"({_MIN_ABSTRACT_SCORE}). "
                    f"The question may be too vague or off-topic for arXiv."
                )
                continue

            selected_ids = [a["article_id"] for a in top_abstracts]
            print("  Selected papers:")
            for a in top_abstracts:
                print(
                    f"    [{a['article_id']}] score: {a['score']:.3f} | "
                    f"{a['content'][:80].strip()}..."
                )

            # 5. Fetch PDFs → chunk → embed → save
            print("\n[5/6] Fetching and indexing full papers...")
            fetch_result = fetch_multiple(selected_ids)
            if fetch_result.failed:
                print(f"  Warning: failed to fetch {fetch_result.failed}")

            if not fetch_result.articles:
                print("  No papers could be fetched.")
                continue

            chunk_results = chunk_multiple(fetch_result.articles)
            embedded_chunks_list = embed_multiple(chunk_results)
            saved_chunks = save_chunks(qdrant, embedded_chunks_list)
            total_chunks = sum(len(r.chunks) for r in chunk_results)
            print(f"  {total_chunks} chunks | {saved_chunks} newly indexed")

            # 6. Retrieve top-5 chunks → generate answer
            print("\n[6/6] Retrieving and generating answer...")
            chunks = retrieve_chunks(question, qdrant)

            if not chunks:
                print(
                    f"  ⚠️  No relevant chunks found above score threshold "
                    f"({_MIN_CHUNK_SCORE}). "
                    f"The retrieved papers may not directly answer this question."
                )
                continue

            print(f"  Retrieved {len(chunks)} chunks:")
            for c in chunks:
                print(
                    f"    [{c['article_id']}] "
                    f"score: {c['score']:.3f} | "
                    f"{c['content'][:80].strip()}..."
                )

            answer = generate(question, chunks)
            print(f"\n{'=' * 60}")
            print(f"Answer:\n{answer}")
            print("=" * 60)

        except QueryAnalysisError as e:
            print(f"Query analysis error: {e}")
        except Exception as e:
            logger.error("Unexpected error", extra={"error": str(e)})
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
