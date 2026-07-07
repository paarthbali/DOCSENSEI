"""
utils/prompt.py

The strict RAG prompt template. Forces the LLM to answer only from
retrieved context, cite sources inline, and explicitly refuse when the
answer isn't present — this is what prevents hallucination.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

REFUSAL_MESSAGE = "I do not know based on the provided document."

_TEMPLATE = """You are DocSensei, a precise document Q&A assistant.

STRICT RULES:
1. Answer the question using ONLY the context provided below.
2. Never use outside knowledge, even if you are confident about the answer.
3. Every factual claim in your answer MUST end with an inline citation in
   the exact format shown in the context, e.g. (DocumentName, Page 4) or
   (DocumentName, Chunk 12).
4. If the answer cannot be found in the context, reply with EXACTLY this
   sentence and nothing else: "{refusal}"
5. Be concise and factual. Do not pad the answer with filler.

Context:
{context}

Chat history (for follow-up questions, use only to resolve references like "it" or "that"):
{chat_history}

Question: {question}

Answer (with inline citations):"""


def build_prompt(context: str, question: str, chat_history: str) -> str:
    """Fill the strict QA template with the given values."""
    return _TEMPLATE.format(
        refusal=REFUSAL_MESSAGE,
        context=context,
        chat_history=chat_history,
        question=question,
    )


def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into a citation-labeled context block."""
    parts = []
    for doc in chunks:
        citation = doc.metadata.get("citation", "Unknown source")
        parts.append(f"[Source: {citation}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_chat_history(history: List[dict], max_turns: int = 3) -> str:
    """Format the last few Q&A turns as plain text for the prompt."""
    if not history:
        return "(no previous questions)"
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"Q: {turn.get('question', '')}")
        lines.append(f"A: {turn.get('answer', '')}")
    return "\n".join(lines)
