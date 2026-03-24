"""
arXiv search module.

Queries the arXiv Atom API by title and abstract,
returns the top N results as Paper objects.
Only English-language papers are returned.
"""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from langdetect import LangDetectException, detect
from pydantic import BaseModel

from logger import setup_logger

logger = setup_logger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"


class Paper(BaseModel):
    rank: int
    id: str
    title: str
    summary: str


class SearchResult(BaseModel):
    query: str
    total: int
    papers: list[Paper]


class SearchError(Exception):
    pass


def _is_english(text: str) -> bool:
    """Return True if the text is detected as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True  # uncertain → keep


def search_arxiv(topic: str, max_results: int = 20) -> SearchResult:
    """
    Search arXiv for English papers matching the topic in title or abstract.

    Args:
        topic: Search query string.
        max_results: Maximum number of results to return (default: 20).

    Returns:
        SearchResult containing matched English papers.

    Raises:
        SearchError: If the request or parsing fails.
    """
    params = urllib.parse.urlencode(
        {
            "search_query": f"ti:{topic} OR abs:{topic}",
            "sortBy": "relevance",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        }
    )

    try:
        logger.info(
            "Search started", extra={"topic": topic, "max_results": max_results}
        )

        with urllib.request.urlopen(f"{_BASE_URL}?{params}") as r:
            xml_data = r.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        skipped = 0
        for i, entry in enumerate(root.findall("atom:entry", ns)):
            arxiv_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

            if not _is_english(summary):
                skipped += 1
                logger.debug(
                    "Non-English paper skipped",
                    extra={"arxiv_id": arxiv_id, "title": title[:60]},
                )
                continue

            papers.append(
                Paper(rank=len(papers) + 1, id=arxiv_id, title=title, summary=summary)
            )

        logger.info(
            "Search completed",
            extra={"topic": topic, "total": len(papers), "skipped": skipped},
        )
        return SearchResult(query=topic, total=len(papers), papers=papers)

    except Exception as e:
        logger.error("Search failed", extra={"topic": topic, "error": str(e)})
        raise SearchError(f"Search failed: {e}") from e
