"""
utils/loader.py

Handles reading PDF and DOCX files and turning them into LangChain
Document objects with consistent metadata (source filename, page number).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

from utils.helpers import validate_upload


@dataclass
class LoadedDocument:
    """Container for a loaded file's parsed content and metadata."""

    filename: str
    documents: List[Document]
    num_pages: int = field(default=0)

    @property
    def full_text(self) -> str:
        return "\n\n".join(d.page_content for d in self.documents)


class DocumentLoader:
    """Loads PDF/DOCX files (from Streamlit's UploadedFile objects) into
    LangChain Document objects, with page/source metadata attached.
    """

    @staticmethod
    def load(uploaded_file) -> LoadedDocument:
        """Load a single uploaded file.

        Args:
            uploaded_file: a Streamlit UploadedFile-like object with
                `.name` and `.getbuffer()` / `.read()`.

        Raises:
            ValueError: for empty, unsupported, oversized, or corrupted files.
        """
        filename = uploaded_file.name
        file_bytes = uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read()
        validate_upload(filename, bytes(file_bytes))

        suffix = os.path.splitext(filename)[1].lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(bytes(file_bytes))
                tmp_path = tmp.name

            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif suffix in (".docx", ".doc"):
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            if not docs or all(not d.page_content.strip() for d in docs):
                raise ValueError(
                    f"'{filename}' could not be read — it may be corrupted, "
                    f"password-protected, or a scanned image with no extractable text."
                )

            # Attach consistent metadata
            for d in docs:
                d.metadata["source"] = filename
                if "page" not in d.metadata:
                    d.metadata["page"] = 0

            num_pages = len(docs) if suffix == ".pdf" else 1
            return LoadedDocument(filename=filename, documents=docs, num_pages=num_pages)

        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to process '{filename}': {exc}") from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @classmethod
    def load_multiple(cls, uploaded_files) -> "tuple[List[LoadedDocument], List[str]]":
        """Load several uploaded files, skipping any that fail with a
        collected list of errors rather than crashing the whole batch.
        """
        loaded: List[LoadedDocument] = []
        errors: List[str] = []
        for f in uploaded_files:
            try:
                loaded.append(cls.load(f))
            except ValueError as exc:
                errors.append(str(exc))
        if errors and not loaded:
            # All files failed
            raise ValueError(" | ".join(errors))
        return loaded, errors
