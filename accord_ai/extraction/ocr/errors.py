"""OCR error types."""
from __future__ import annotations


class OCRReadError(ValueError):
    """Image unreadable or no text extracted.

    API layer returns 422. Caller should ask the user to re-upload
    a clearer image.
    """


class OCRConfigError(RuntimeError):
    """Missing system dependency (tesseract binary) or Python package.

    API layer returns 503. Operator needs to install the dependency.
    """
