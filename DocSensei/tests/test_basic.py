"""
tests/test_basic.py

Lightweight sanity tests that don't require API keys or network access —
they check the pure-Python logic (chunking, token counting, validation).

Run with:
    python -m pytest tests/
or simply:
    python -m unittest tests.test_basic
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

from utils.splitter import (
    split_into_sentences,
    SentenceChunker,
    get_splitter,
    attach_chunk_metadata,
    STRATEGY_RECURSIVE,
    STRATEGY_SENTENCE,
)
from utils.helpers import approx_token_count, validate_upload


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_basic_sentences(self):
        text = "This is one sentence. This is another! Is this a third?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_into_sentences(""), [])

    def test_sentence_chunker_respects_target_size(self):
        text = " ".join([f"Sentence number {i}." for i in range(50)])
        chunker = SentenceChunker(target_chunk_size=200, overlap_sentences=1)
        chunks = chunker._chunk_text(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks[:-1]:  # last chunk may be shorter
            self.assertGreaterEqual(len(c), 100)


class TestSplitterFactory(unittest.TestCase):
    def test_recursive_strategy_returns_documents(self):
        docs = [Document(page_content="Hello world. " * 100, metadata={"source": "test.pdf", "page": 0})]
        splitter = get_splitter(STRATEGY_RECURSIVE, chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        chunks = attach_chunk_metadata(chunks, STRATEGY_RECURSIVE)
        self.assertGreater(len(chunks), 1)
        self.assertIn("citation", chunks[0].metadata)

    def test_sentence_strategy_returns_documents(self):
        docs = [Document(page_content="Hello world. " * 100, metadata={"source": "test.pdf", "page": 0})]
        splitter = get_splitter(STRATEGY_SENTENCE, chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        chunks = attach_chunk_metadata(chunks, STRATEGY_SENTENCE)
        self.assertGreater(len(chunks), 0)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            get_splitter("not_a_real_strategy", 100, 20)


class TestHelpers(unittest.TestCase):
    def test_token_count_approx(self):
        self.assertEqual(approx_token_count(""), 0)
        self.assertGreater(approx_token_count("one two three four"), 0)

    def test_validate_upload_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_upload("empty.pdf", b"")

    def test_validate_upload_rejects_unsupported_type(self):
        with self.assertRaises(ValueError):
            validate_upload("notes.txt", b"some content")

    def test_validate_upload_accepts_pdf(self):
        try:
            validate_upload("notes.pdf", b"some content")
        except ValueError:
            self.fail("validate_upload raised unexpectedly for a valid PDF")


if __name__ == "__main__":
    unittest.main()
