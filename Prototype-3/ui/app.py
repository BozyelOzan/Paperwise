"""Chainlit UI for Paperwise."""

import os
import sys
import uuid
from pathlib import Path

import chainlit as cl
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

_API_URL = os.getenv("API_URL", "http://localhost:8000")
_TIMEOUT = 300

_STEP_LABELS = {
    "Analyzing query...": "🔍 Analyzing query",
    "Searching arXiv...": "📚 Searching arXiv",
    "Indexing abstracts...": "🗂 Indexing abstracts",
    "Selecting most relevant papers...": "🎯 Selecting papers",
    "Fetching papers...": "⬇️ Fetching papers",
    "Retrieving relevant chunks...": "🔎 Retrieving chunks",
    "Generating answer...": "✍️ Generating answer",
}


def _summarize_step(raw: str) -> str | None:
    for key, label in _STEP_LABELS.items():
        if raw.startswith(key):
            return label
    if raw.startswith("Found "):
        return f"📄 {raw}"
    return None


def _arxiv_url(article_id: str) -> str:
    base_id = article_id.split("v")[0] if "v" in article_id else article_id
    return f"https://arxiv.org/abs/{base_id}"


@cl.on_chat_start
async def on_start():
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    await cl.Message(
        content=(
            "👋 Welcome to **Paperwise**!\n\n"
            "Ask me anything about academic research. "
            "I'll search arXiv, find the most relevant papers, "
            "and answer your question."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    question = message.content.strip()

    if not question:
        return

    pipeline_data = None

    # Step 1 — run the heavy pipeline (search, fetch, chunk, embed)
    async with cl.Step(name="🔬 Paperwise Pipeline") as pipeline_step:
        pipeline_step.output = "⏳ Processing..."

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{_API_URL}/query/pipeline",
                    json={"user_id": user_id, "question": question},
                )
                response.raise_for_status()
                pipeline_data = response.json()

        except httpx.TimeoutException:
            pipeline_step.output = "⚠️ Request timed out."
            await cl.Message(content="⚠️ Request timed out.").send()
            return
        except Exception as e:
            pipeline_step.output = f"⚠️ Error: {e}"
            await cl.Message(content=f"⚠️ Error: {e}").send()
            return

        step_lines = []
        for raw in pipeline_data.get("step_log", []):
            label = _summarize_step(raw)
            if label:
                step_lines.append(f"✅ {label}")

        pipeline_step.output = "\n".join(step_lines) if step_lines else "✅ Done."

    if pipeline_data is None:
        return

    status = pipeline_data.get("status")
    if status != "ok":
        await cl.Message(content=pipeline_data.get("answer", "No answer.")).send()
        return

    # Step 2 — stream the generated answer token by token
    chunks = pipeline_data.get("chunks", [])
    seen = set()
    elements = []
    for c in chunks:
        aid = c["article_id"]
        if aid not in seen:
            seen.add(aid)
            elements.append(
                cl.Text(
                    name=aid,
                    content=_arxiv_url(aid),
                    display="side",
                    url=_arxiv_url(aid),
                )
            )

    answer_msg = cl.Message(content="", elements=elements)
    await answer_msg.send()

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{_API_URL}/query/stream",
                json={"question": question, "chunks": chunks},
            ) as stream:
                async for token in stream.aiter_text():
                    answer_msg.content += token
                    await answer_msg.update()
    except Exception as e:
        answer_msg.content += f"\n\n⚠️ Stream error: {e}"
        await answer_msg.update()
