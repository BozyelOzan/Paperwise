"""
Multi-user simulation script.

Simulates multiple concurrent users via asyncio.
Each user sends an independent question with its own user_id;
responses are printed to the console in real time.

Usage:
    python simulate_users.py
    python simulate_users.py --users 3 --api http://localhost:8000
"""

# Standard library imports
import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass, field

import httpx

# ── Terminal color codes ──────────────────────────────────────────────────────
COLORS = ["\033[94m", "\033[92m", "\033[93m", "\033[95m", "\033[96m"]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# ── Default questions ─────────────────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    "What are the latest advances in large language model alignment?",
    "How does retrieval-augmented generation improve factual accuracy?",
    "What are the key challenges in federated learning for privacy?",
    "Explain the transformer architecture and its attention mechanism.",
    "What is the role of reinforcement learning from human feedback in LLMs?",
]


@dataclass
class UserResult:
    user_index: int
    user_id: str
    question: str
    status: str = "pending"
    answer: str = ""
    step_log: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""


def cprint(user_index: int, msg: str, dim: bool = False) -> None:
    """Print a color-coded, user-tagged line to the console."""
    color = COLORS[user_index % len(COLORS)]
    tag = f"[User {user_index + 1}]"
    style = DIM if dim else ""
    print(f"{color}{BOLD}{tag}{RESET}{style} {msg}{RESET}")


async def run_user(
    user_index: int,
    question: str,
    api_url: str,
    timeout: int,
) -> UserResult:
    """
    Run the full flow for a single simulated user:
      1. /query/pipeline  → pipeline (arXiv search, PDF fetch, embedding, retrieval)
      2. /query/stream    → LLM token stream (server-sent events)
    """
    user_id = str(uuid.uuid4())
    result = UserResult(user_index=user_index, user_id=user_id, question=question)
    start = time.perf_counter()

    cprint(
        user_index,
        f'Sending question: {BOLD}"{question[:70]}..."'
        if len(question) > 70
        else f'Sending question: {BOLD}"{question}"',
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        # ── 1. Pipeline request — arXiv search, PDF fetch, embedding, retrieval ─
        try:
            cprint(user_index, "Pipeline started...", dim=True)
            resp = await client.post(
                f"{api_url}/query/pipeline",
                json={"user_id": user_id, "question": question},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            result.status = "timeout"
            result.error = "Pipeline timeout"
            result.elapsed = time.perf_counter() - start
            cprint(user_index, f"❌ Pipeline timeout ({result.elapsed:.1f}s)")
            return result
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            result.elapsed = time.perf_counter() - start
            cprint(user_index, f"❌ Pipeline error: {e}")
            return result

        result.status = data.get("status", "unknown")
        result.step_log = data.get("step_log", [])
        result.chunks = data.get("chunks", [])

        # Print each pipeline step as it arrives
        for step in result.step_log:
            cprint(user_index, f"  ✓ {step}", dim=True)

        if result.status != "ok":
            result.answer = data.get("answer", "")
            result.elapsed = time.perf_counter() - start
            cprint(user_index, f"⚠️  Status: {result.status} — {result.answer[:80]}")
            return result

        # ── 2. Answer stream — tokens received in real time ───────────────────
        cprint(user_index, "Starting answer stream...", dim=True)
        tokens = []
        try:
            async with client.stream(
                "POST",
                f"{api_url}/query/stream",
                json={"question": question, "chunks": result.chunks},
            ) as stream:
                async for token in stream.aiter_text():
                    tokens.append(token)
                    # Notify once the first token arrives
                    if len(tokens) == 1:
                        cprint(user_index, "First token received, answer incoming...")
        except Exception as e:
            result.error = f"Stream error: {e}"
            result.elapsed = time.perf_counter() - start
            cprint(user_index, f"❌ Stream error: {e}")
            return result

        result.answer = "".join(tokens)
        result.elapsed = time.perf_counter() - start

    cprint(
        user_index,
        f"✅ Done ({result.elapsed:.1f}s) — {len(result.chunks)} chunks, {len(result.answer)} chars",
    )
    return result


def print_summary(results: list[UserResult], total_elapsed: float) -> None:
    """Print a formatted summary table of all user results to the console."""
    print("\n" + "═" * 70)
    print(f"{BOLD}  RESULTS{RESET}")
    print("═" * 70)

    for r in results:
        color = COLORS[r.user_index % len(COLORS)]
        status_icon = {
            "ok": "✅",
            "no_results": "⚠️",
            "error": "❌",
            "timeout": "⏱️",
        }.get(r.status, "❓")

        print(
            f"\n{color}{BOLD}── User {r.user_index + 1}{RESET}  {DIM}(user_id: {r.user_id[:8]}...){RESET}"
        )
        print(f"   Question : {r.question[:80]}{'...' if len(r.question) > 80 else ''}")
        print(
            f"   Status   : {status_icon} {r.status}  |  Time: {r.elapsed:.1f}s  |  Chunks: {len(r.chunks)}"
        )

        if r.error:
            print(f"   Error    : {r.error}")

        if r.answer:
            # Show the first 300 characters of the answer as a preview
            preview = r.answer[:300].replace("\n", " ")
            ellipsis = "..." if len(r.answer) > 300 else ""
            print(f"   Answer   : {preview}{ellipsis}")

        if r.chunks:
            unique_articles = {c["article_id"] for c in r.chunks}
            print(f"   Articles : {', '.join(sorted(unique_articles))}")

    print("\n" + "─" * 70)
    ok_count = sum(1 for r in results if r.status == "ok")
    print(
        f"{BOLD}  Total time: {total_elapsed:.1f}s  |  Successful: {ok_count}/{len(results)}{RESET}"
    )
    print("─" * 70 + "\n")


async def main(num_users: int, api_url: str, timeout: int) -> None:
    questions = DEFAULT_QUESTIONS[:num_users]

    print("\n" + "═" * 70)
    print(f"{BOLD}  PAPERWISE — Multi-User Simulation{RESET}")
    print(f"  {num_users} users  |  API: {api_url}")
    print("═" * 70 + "\n")

    # Launch all user tasks concurrently
    wall_start = time.perf_counter()
    tasks = [
        run_user(
            user_index=i,
            question=questions[i],
            api_url=api_url,
            timeout=timeout,
        )
        for i in range(num_users)
    ]
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_start

    print_summary(list(results), wall_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paperwise multi-user simulation")
    parser.add_argument(
        "--users", type=int, default=2, help="Number of concurrent users (default: 2)"
    )
    parser.add_argument(
        "--api", type=str, default="http://localhost:8000", help="FastAPI base URL"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    if args.users > len(DEFAULT_QUESTIONS):
        print(
            f"⚠️  Maximum {len(DEFAULT_QUESTIONS)} users supported. Setting to {len(DEFAULT_QUESTIONS)}."
        )
        args.users = len(DEFAULT_QUESTIONS)

    asyncio.run(main(args.users, args.api, args.timeout))
