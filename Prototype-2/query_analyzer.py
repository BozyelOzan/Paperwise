"""
Query analysis module.

Sends the user's question to GPT-4o-mini and extracts:
  - core search terms explicitly present in the question (1-3 max)
  - an arXiv-style reformulation for semantic abstract search
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = """You are a scientific search assistant specializing in academic paper retrieval.

Your task is to extract the CORE TOPIC(S) from the user's question for querying arXiv.

Rules:
- Extract ONLY the main topic(s) explicitly mentioned in the question.
- Do NOT generate related terms, synonyms, or broader concepts.
- If the question mentions 1 topic → return 1 term.
- If the question mentions 2 topics → return 2 terms.
- Maximum 3 terms, only if clearly present in the question.
- Use the exact technical term as it would appear in a paper title.

Return ONLY valid JSON. No markdown, no explanation.

Examples:
  Question: "What is LLMs?"
  Output: {"terms": ["large language models"], "arxiv_query": "Large language models (LLMs) are neural network architectures trained on large text corpora to understand and generate human language."}

  Question: "How does U-Net differ from V-Net?"
  Output: {"terms": ["U-Net", "V-Net"], "arxiv_query": "Comparison of U-Net and V-Net architectures for medical image segmentation tasks."}

  Question: "How do skip connections in U-Net improve segmentation?"
  Output: {"terms": ["U-Net"], "arxiv_query": "Skip connections in U-Net encoder-decoder architecture improve segmentation accuracy by preserving spatial information."}

  Question: "What is the role of self-attention in transformers?"
  Output: {"terms": ["transformer self-attention"], "arxiv_query": "Self-attention mechanism in transformer models captures long-range dependencies in sequences."}"""


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
