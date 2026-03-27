"""
Query analysis service.

Sends the user's question to GPT-4o-mini and extracts:
  - core search terms explicitly present in the question (1-3 max)
  - an arXiv-style reformulation for semantic abstract search
"""

import json
import os

from dotenv import load_dotenv
from logger import setup_logger
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
logger = setup_logger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = """You are a scientific search assistant specializing in academic paper retrieval.

Your task is to analyze the user's question and extract the core topics to construct a precise search query for the arXiv API.

Core Rules:
1. Extract ONLY the exact technical topics explicitly mentioned in the text.
2. Use the exact technical term as it would appear in an academic paper title or abstract.
3. Formulate a logical boolean query for arXiv based strictly on the extracted terms.

Respond with a JSON object containing exactly these keys:
- "question": the original question (string)
- "terms": list of extracted technical terms (list of strings, 1-3 items)
- "arxiv_query": arXiv-style boolean query string (string)

Do not include any text outside the JSON object."""


class QueryAnalysis(BaseModel):
    question: str
    terms: list[str]
    arxiv_query: str


class QueryAnalysisError(Exception):
    pass


def analyze_query(question: str) -> QueryAnalysis:
    """
    Extract core arXiv search terms and semantic query from a natural language question.

    Args:
        question: User's natural language question.

    Returns:
        QueryAnalysis with search terms and arXiv-style reformulation.

    Raises:
        QueryAnalysisError: If the LLM call fails or returns invalid JSON.
    """
    logger.info("Query analysis started", extra={"question": question})

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise QueryAnalysisError(f"LLM returned invalid JSON: {raw}") from e

        if "terms" not in data or "arxiv_query" not in data:
            raise QueryAnalysisError(f"Missing keys in LLM response: {data}")

        if not isinstance(data["terms"], list) or not all(
            isinstance(t, str) for t in data["terms"]
        ):
            raise QueryAnalysisError(f"Invalid terms format: {data['terms']}")

        if not isinstance(data["arxiv_query"], str):
            raise QueryAnalysisError(
                f"Invalid arxiv_query format: {data['arxiv_query']}"
            )

        result = QueryAnalysis(
            question=question,
            terms=data["terms"],
            arxiv_query=data["arxiv_query"],
        )

        logger.info(
            "Query analysis completed",
            extra={
                "question": question,
                "terms": result.terms,
                "arxiv_query": result.arxiv_query,
            },
        )
        return result

    except QueryAnalysisError:
        raise
    except Exception as e:
        logger.error("Query analysis failed", extra={"error": str(e)})
        raise QueryAnalysisError(f"Query analysis failed: {e}") from e
