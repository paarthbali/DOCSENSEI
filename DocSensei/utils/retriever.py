"""
utils/retriever.py

Thin helpers around Chroma's retriever / similarity search so the rest
of the app (and the comparison module) shares one code path.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def get_retriever(vectordb: Chroma, top_k: int = 4):
    """Return a LangChain retriever configured for top-k similarity search."""
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def similarity_search_with_scores(
    vectordb: Chroma, query: str, top_k: int = 4
) -> List[Tuple[Document, float]]:
    """Run a similarity search and return (document, relevance_score) pairs.

    Relevance score is normalized so higher = more relevant (0 to 1 range,
    approximately, depending on the underlying distance metric).
    """
    return vectordb.similarity_search_with_relevance_scores(query, k=top_k)
