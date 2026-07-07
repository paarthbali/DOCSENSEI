"""
utils/comparison.py

Runs both chunking strategies on the same document set and produces a
side-by-side comparison report: chunk counts, average chunk size,
retrieval latency, top similarity score, and generated answer quality
(via the same question asked against both indexes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from utils.splitter import get_splitter, attach_chunk_metadata, ALL_STRATEGIES
from utils.vectorstore import VectorStoreManager
from utils.retriever import similarity_search_with_scores
from utils.rag import RAGPipeline
from utils.helpers import Timer, format_timestamp


@dataclass
class StrategyResult:
    strategy: str
    num_chunks: int
    avg_chunk_size: float
    retrieval_latency_seconds: float
    top_similarity_score: float
    answer: str
    retrieved_previews: List[str] = field(default_factory=list)


class ChunkingComparison:
    """Coordinates building two temporary vector stores (one per strategy)
    and gathering comparable metrics from each.
    """

    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager

    def _run_single_strategy(
        self,
        strategy: str,
        raw_docs: List[Document],
        embeddings,
        chunk_size: int,
        chunk_overlap: int,
        question: str,
        top_k: int,
        llm,
        collection_prefix: str,
    ) -> StrategyResult:
        splitter = get_splitter(strategy, chunk_size, chunk_overlap)
        chunks = splitter.split_documents(raw_docs)
        chunks = attach_chunk_metadata(chunks, strategy)

        avg_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0

        collection_name = f"{collection_prefix}_{strategy.replace(' ', '_').lower()}"
        vectordb = self.vector_manager.build(chunks, embeddings, collection_name)

        with Timer() as t:
            results = similarity_search_with_scores(vectordb, question, top_k=top_k)

        top_score = max((score for _, score in results), default=0.0)
        previews = [doc.page_content[:200] for doc, _ in results]

        rag = RAGPipeline(llm)
        rag_answer = rag.answer(vectordb, question, top_k=top_k)

        # Clean up the temporary comparison collection so it doesn't
        # permanently clutter the knowledge base.
        self.vector_manager.clear_collection(collection_name)

        return StrategyResult(
            strategy=strategy,
            num_chunks=len(chunks),
            avg_chunk_size=round(avg_size, 1),
            retrieval_latency_seconds=round(t.elapsed, 4),
            top_similarity_score=round(top_score, 4),
            answer=rag_answer.answer,
            retrieved_previews=previews,
        )

    def run(
        self,
        raw_docs: List[Document],
        embeddings,
        llm,
        question: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 4,
        collection_prefix: str = "comparison",
    ) -> List[StrategyResult]:
        """Run both chunking strategies and return their results."""
        results = []
        for strategy in ALL_STRATEGIES:
            result = self._run_single_strategy(
                strategy=strategy,
                raw_docs=raw_docs,
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                question=question,
                top_k=top_k,
                llm=llm,
                collection_prefix=collection_prefix,
            )
            results.append(result)
        return results

    @staticmethod
    def recommend(results: List[StrategyResult]) -> str:
        """Pick a winner using a simple weighted score: similarity matters
        most, latency matters a little. Returns the recommended strategy name.
        """
        if not results:
            return "N/A"

        def score(r: StrategyResult) -> float:
            return r.top_similarity_score - (r.retrieval_latency_seconds * 0.05)

        best = max(results, key=score)
        return best.strategy

    @staticmethod
    def to_markdown_report(results: List[StrategyResult], question: str) -> str:
        """Render a downloadable Markdown comparison report."""
        lines = [
            "# DocSensei — Chunking Strategy Comparison Report",
            f"_Generated: {format_timestamp()}_",
            "",
            f"**Test question:** {question}",
            "",
            "| Metric | " + " | ".join(r.strategy for r in results) + " |",
            "|---|" + "---|" * len(results),
            "| Number of chunks | " + " | ".join(str(r.num_chunks) for r in results) + " |",
            "| Avg. chunk size (chars) | " + " | ".join(str(r.avg_chunk_size) for r in results) + " |",
            "| Retrieval latency (s) | " + " | ".join(str(r.retrieval_latency_seconds) for r in results) + " |",
            "| Top similarity score | " + " | ".join(str(r.top_similarity_score) for r in results) + " |",
            "",
        ]
        for r in results:
            lines.append(f"## {r.strategy} — Generated Answer")
            lines.append(r.answer)
            lines.append("")

        winner = ChunkingComparison.recommend(results)
        lines.append(f"## Recommendation\n**{winner}** performed better for this document and question, "
                      f"based on retrieval relevance (weighted higher) and latency (weighted lower).")
        return "\n".join(lines)
