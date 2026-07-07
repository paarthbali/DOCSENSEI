"""
utils/helpers.py

Small shared utility functions used across the DocSensei project:
- approximate token counting (no heavy tokenizer dependency)
- timestamp formatting
- upload validation
- a safe-call wrapper for consistent error handling
"""

from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Any, Callable, Tuple


SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc")


def approx_token_count(text: str) -> int:
    """Approximate token count without requiring a heavy tokenizer.

    Rule of thumb: 1 token is roughly 0.75 words for English text.
    This is accurate enough for display purposes (statistics panel),
    not for billing-accurate calculations.
    """
    if not text:
        return 0
    word_count = len(text.split())
    return int(word_count / 0.75)


def format_timestamp(dt: datetime | None = None) -> str:
    """Return a human-readable timestamp string."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_upload(filename: str, file_bytes: bytes) -> None:
    """Validate an uploaded file before processing.

    Raises:
        ValueError: if the file is empty, unsupported, or too large.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f"'{filename}' appears to be empty.")

    lower_name = filename.lower()
    if not lower_name.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(
            f"'{filename}' is not a supported file type. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    max_size_mb = 50
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit."
        )


def safe_call(default: Any = None) -> Callable:
    """Decorator that catches exceptions and returns (result, error_message).

    Usage:
        @safe_call()
        def risky():
            ...
        result, err = risky()
        if err:
            st.error(err)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Any, str | None]:
            try:
                return func(*args, **kwargs), None
            except Exception as exc:  # noqa: BLE001 - intentional broad catch for UI safety
                return default, str(exc)

        return wrapper

    return decorator


class Timer:
    """Simple context manager to measure elapsed time in seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start
