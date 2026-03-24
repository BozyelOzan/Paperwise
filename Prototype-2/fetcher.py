"""
arXiv PDF fetcher.

Downloads PDF files from arxiv.org in parallel.
Validates that the response is a real PDF (starts with %PDF magic bytes).
"""

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import setup_logger
from pydantic import BaseModel

logger = setup_logger(__name__)

_PDF_MAGIC = b"%PDF"
_USER_AGENT = "Mozilla/5.0 (compatible; paperwise/1.0)"


class Article(BaseModel):
    id: str
    content: bytes


class FetchResult(BaseModel):
    total: int
    success: int
    failed: list[str]
    articles: list[Article]


class FetchError(Exception):
    pass


def fetch_pdf(arxiv_id: str, max_retries: int = 3) -> Article:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    logger.info("Fetching PDF", extra={"arxiv_id": arxiv_id})

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                content = r.read()

            if not content.startswith(_PDF_MAGIC):
                raise FetchError(f"Response for {arxiv_id} is not a valid PDF")

            logger.info(
                "PDF fetched", extra={"arxiv_id": arxiv_id, "size_bytes": len(content)}
            )
            return Article(id=arxiv_id, content=content)

        except FetchError:
            raise
        except Exception as e:
            logger.warning(
                "Fetch attempt failed",
                extra={"arxiv_id": arxiv_id, "attempt": attempt, "error": str(e)},
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)  # 2s, 4s
            else:
                raise FetchError(
                    f"Failed to fetch {arxiv_id} after {max_retries} attempts: {e}"
                ) from e


def fetch_multiple(arxiv_ids: list[str], max_workers: int = 3) -> FetchResult:
    """
    Download multiple PDFs in parallel.

    Args:
        arxiv_ids: List of arXiv paper identifiers.
        max_workers: Thread pool size (default: 3).

    Returns:
        FetchResult with successfully fetched articles and failed IDs.
    """
    logger.info("Batch fetch started", extra={"total": len(arxiv_ids)})
    articles: list[Article] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_pdf, aid): aid for aid in arxiv_ids}
        for future in as_completed(futures):
            aid = futures[future]
            try:
                articles.append(future.result())
            except FetchError:
                failed.append(aid)

    logger.info(
        "Batch fetch completed",
        extra={"success": len(articles), "failed": len(failed)},
    )
    return FetchResult(
        total=len(arxiv_ids),
        success=len(articles),
        failed=failed,
        articles=articles,
    )
