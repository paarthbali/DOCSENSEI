"""
utils/vectorstore.py

Manages creation, persistence, and clearing of ChromaDB collections.
Each (document set + chunking strategy) combination gets its own named
collection so switching strategies doesn't require re-uploading files.
"""

from __future__ import annotations

import os
import shutil
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DEFAULT_PERSIST_DIR = "chroma_db"


class VectorStoreManager:
    """Builds and manages local, persistent Chroma vector stores."""

    def __init__(self, persist_directory: str = DEFAULT_PERSIST_DIR):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

    def build(self, chunks: List[Document], embeddings, collection_name: str) -> Chroma:
        """Build (or rebuild) a Chroma collection from document chunks."""
        collection_path = os.path.join(self.persist_directory, collection_name)
        os.makedirs(collection_path, exist_ok=True)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=collection_path,
        )
        # Newer versions of langchain-chroma auto-persist; older ones need
        # an explicit call. Try both gracefully.
        try:
            vectordb.persist()
        except Exception:  # noqa: BLE001
            pass
        return vectordb

    def load(self, embeddings, collection_name: str) -> Chroma | None:
        """Load an existing persisted collection, if present on disk."""
        collection_path = os.path.join(self.persist_directory, collection_name)
        if not os.path.exists(collection_path):
            return None
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=collection_path,
        )

    def clear_all(self) -> None:
        """Delete the entire knowledge base (all collections) from disk."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def clear_collection(self, collection_name: str) -> None:
        """Delete a single named collection from disk."""
        collection_path = os.path.join(self.persist_directory, collection_name)
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path, ignore_errors=True)
