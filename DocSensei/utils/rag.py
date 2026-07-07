"""
utils/rag.py

The core RAG pipeline: question -> embed -> retrieve -> strict prompt ->
LLM -> answer + sources. Supports Google Gemini, OpenAI, and Groq as
configurable LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from utils.prompt import build_prompt, format_context, format_chat_history, REFUSAL_MESSAGE
from utils.retriever import get_retriever
from utils.helpers import Timer

PROVIDER_GEMINI = "Google Gemini"
PROVIDER_OPENAI = "OpenAI"
PROVIDER_GROQ = "Groq"
ALL_PROVIDERS = (PROVIDER_GEMINI, PROVIDER_OPENAI, PROVIDER_GROQ)

DEFAULT_MODELS = {
    PROVIDER_GEMINI: "gemini-2.0-flash",
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_GROQ: "llama-3.1-8b-instant",
}

ENV_VAR_NAMES = {
    PROVIDER_GEMINI: "GOOGLE_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
    PROVIDER_GROQ: "GROQ_API_KEY",
}


class MissingAPIKeyError(Exception):
    """Raised when the selected provider has no API key configured."""


@dataclass
class RAGAnswer:
    answer: str
    sources: List[Document] = field(default_factory=list)
    latency_seconds: float = 0.0
    refused: bool = False


def get_llm(provider: str, model_name: str, api_key: str, temperature: float = 0.2):
    """Factory that returns a LangChain chat model for the chosen provider.

    Raises:
        MissingAPIKeyError: if api_key is empty.
        ValueError: if provider is not recognized.
    """
    if not api_key:
        raise MissingAPIKeyError(
            f"No API key provided for {provider}. "
            f"Set it in the sidebar or as the {ENV_VAR_NAMES.get(provider, 'API_KEY')} environment variable."
        )

    if provider == PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

    elif provider == PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)

    elif provider == PROVIDER_GROQ:
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"Unknown provider: {provider}")


class RAGPipeline:
    """Ties retrieval and generation together for a single vector store."""

    def __init__(self, llm):
        self.llm = llm

    def answer(
        self,
        vectordb: Chroma,
        question: str,
        top_k: int = 4,
        chat_history: Optional[List[dict]] = None,
    ) -> RAGAnswer:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        with Timer() as t:
            retriever = get_retriever(vectordb, top_k=top_k)
            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                return RAGAnswer(answer=REFUSAL_MESSAGE, sources=[], latency_seconds=t.elapsed, refused=True)

            context = format_context(retrieved_docs)
            history_text = format_chat_history(chat_history or [])
            prompt_text = build_prompt(context=context, question=question, chat_history=history_text)

            response = self.llm.invoke(prompt_text)
            answer_text = response.content if hasattr(response, "content") else str(response)

        refused = REFUSAL_MESSAGE.strip().lower() in answer_text.strip().lower()
        return RAGAnswer(
            answer=answer_text.strip(),
            sources=retrieved_docs,
            latency_seconds=t.elapsed,
            refused=refused,
        )

    def stream_answer(self, vectordb: Chroma, question: str, top_k: int = 4, chat_history: Optional[List[dict]] = None):
        """Generator version for Streamlit's st.write_stream. Yields text chunks.
        Falls back to a single yield if the underlying model doesn't support streaming.
        """
        retriever = get_retriever(vectordb, top_k=top_k)
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            yield REFUSAL_MESSAGE
            return

        context = format_context(retrieved_docs)
        history_text = format_chat_history(chat_history or [])
        prompt_text = build_prompt(context=context, question=question, chat_history=history_text)

        try:
            for chunk in self.llm.stream(prompt_text):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text:
                    yield text
        except Exception:  # noqa: BLE001
            response = self.llm.invoke(prompt_text)
            yield response.content if hasattr(response, "content") else str(response)
