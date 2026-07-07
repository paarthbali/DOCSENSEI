"""
utils/embeddings.py

Wraps the Sentence Transformers embedding model. Cached at the Streamlit
resource level (by the app) so the model is only loaded into memory once
per session, regardless of how many documents are processed.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A short list of good, well-tested alternatives users can switch between.
AVAILABLE_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",   # fast, 384-dim, great default
    "sentence-transformers/all-mpnet-base-v2",  # slower, 768-dim, higher quality
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]


class EmbeddingManager:
    """Thin wrapper so the rest of the app depends on this class, not
    directly on langchain_community, making it easy to swap embedding
    backends later (e.g. to OpenAI embeddings) without touching other files.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(model_name=model_name)

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    def embed_query(self, text: str):
        return self._embeddings.embed_query(text)
