"""
PDF fetcher service.

Downloads PDF files from arxiv.org with retry mechanism.
Results are cached in Redis to avoid redundant downloads.
"""

import time
import urllib.error
import urllib.request

from logger import setup_logger

logger = setup_logger(__name__)

_PDF_MAGIC = b"%PDF"
_USER_AGENT = "Mozilla/5.0 (compatible; paperwise/1.0)"


class Article:
    def __init__(self, id: str, content: bytes):
        self.id = id
        self.content = content


class FetchError(Exception):
    pass


def fetch_pdf(arxiv_id: str, max_retries: int = 3) -> Article:
    """
    Download PDF for a given arXiv ID with retry mechanism.

    Args:
        arxiv_id: arXiv paper identifier.
        max_retries: Number of retry attempts (default: 3).

    Returns:
        Article with raw PDF bytes.

    Raises:
        FetchError: If all attempts fail or response is not a valid PDF.
    """
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
                "PDF fetched",
                extra={"arxiv_id": arxiv_id, "size_bytes": len(content)},
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
                time.sleep(2 * attempt)
            else:
                raise FetchError(
                    f"Failed to fetch {arxiv_id} after {max_retries} attempts: {e}"
                ) from e
