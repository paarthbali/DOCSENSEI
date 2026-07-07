"""
utils/splitter.py

Implements the two required chunking strategies:
1. RecursiveCharacterTextSplitter (LangChain's standard character-based splitter)
2. SentenceChunker (custom sentence-aware splitter, groups whole sentences
   together instead of cutting mid-sentence)

Both expose the same `.split_documents(docs) -> List[Document]` interface
so the rest of the app can swap between them transparently.
"""

from __future__ import annotations

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

STRATEGY_RECURSIVE = "Recursive Character Splitter"
STRATEGY_SENTENCE = "Sentence-Based Splitter"
ALL_STRATEGIES = (STRATEGY_RECURSIVE, STRATEGY_SENTENCE)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_into_sentences(text: str) -> List[str]:
    """Lightweight, dependency-free sentence splitter.

    Splits on '.', '!', '?' followed by whitespace and a capital letter
    or digit. Not perfect for edge cases like abbreviations (e.g. "Dr.")
    but performs well for typical study notes / reports / articles.
    """
    text = text.strip().replace("\n", " ")
    if not text:
        return []
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


class SentenceChunker:
    """Groups whole sentences into chunks close to a target character size,
    with a configurable number of overlapping sentences between chunks so
    context isn't lost at chunk boundaries.
    """

    def __init__(self, target_chunk_size: int = 800, overlap_sentences: int = 1):
        self.target_chunk_size = target_chunk_size
        self.overlap_sentences = max(0, overlap_sentences)

    def _chunk_text(self, text: str) -> List[str]:
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            current.append(sentence)
            current_len += len(sentence) + 1
            if current_len >= self.target_chunk_size:
                chunks.append(" ".join(current))
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current_len = sum(len(s) + 1 for s in current)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        result: List[Document] = []
        for doc in docs:
            for text in self._chunk_text(doc.page_content):
                result.append(Document(page_content=text, metadata=dict(doc.metadata)))
        return result


def get_splitter(strategy: str, chunk_size: int, chunk_overlap: int):
    """Factory that returns a splitter object matching the chosen strategy.

    Args:
        strategy: one of ALL_STRATEGIES.
        chunk_size: target character size per chunk.
        chunk_overlap: for the recursive splitter, overlapping characters.
            For the sentence splitter, this is converted into a number of
            overlapping sentences (roughly 1 sentence per 100 overlap chars,
            minimum 0, maximum 3).
    """
    if strategy == STRATEGY_RECURSIVE:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == STRATEGY_SENTENCE:
        overlap_sentences = max(0, min(3, chunk_overlap // 100))
        return SentenceChunker(target_chunk_size=chunk_size, overlap_sentences=overlap_sentences)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def attach_chunk_metadata(chunks: List[Document], strategy: str) -> List[Document]:
    """Attach chunk_id and human-readable citation strings to each chunk."""
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", None)
        source = chunk.metadata.get("source", "document")
        if isinstance(page, int):
            citation = f"{source}, Page {page + 1}, Chunk {i}"
        else:
            citation = f"{source}, Chunk {i}"
        chunk.metadata["chunk_id"] = i
        chunk.metadata["citation"] = citation
        chunk.metadata["strategy"] = strategy
    return chunks
