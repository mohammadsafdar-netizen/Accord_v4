"""Unit tests for accord_ai.extraction.ocr.engine (Phase 1.5).

All tests mock Tesseract so the suite passes without the binary installed.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from accord_ai.extraction.ocr.engine import _preprocess, ocr_image
from accord_ai.extraction.ocr.errors import OCRConfigError, OCRReadError


def _make_png(width: int = 300, height: int = 200) -> bytes:
    """Create a minimal synthetic PNG image."""
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. happy path — tesseract returns expected text
# ---------------------------------------------------------------------------

def test_ocr_image_happy_path(monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.engine._tesseract_available", lambda: True
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda img, lang="eng": "JOHN DOE\nDL TX12345678\nDOB 01/15/1985",
    )
    text = ocr_image(_make_png())
    assert "JOHN DOE" in text
    assert "TX12345678" in text


# ---------------------------------------------------------------------------
# 2. preprocessing upscales small images
# ---------------------------------------------------------------------------

def test_ocr_image_preprocessing_upscales(monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.engine._tesseract_available", lambda: True
    )
    captured = {}

    def fake_tess(img, lang="eng"):
        captured["size"] = img.size
        return "some text"

    monkeypatch.setattr("pytesseract.image_to_string", fake_tess)

    # 500x300 — both edges < 1500; should be upscaled
    ocr_image(_make_png(width=500, height=300))

    assert captured["size"][0] >= 1500 or captured["size"][1] >= 1500


# ---------------------------------------------------------------------------
# 3. non-image bytes → OCRReadError
# ---------------------------------------------------------------------------

def test_ocr_image_raises_on_non_image(monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.engine._tesseract_available", lambda: True
    )
    with pytest.raises(OCRReadError, match="cannot open image"):
        ocr_image(b"this is not an image")


# ---------------------------------------------------------------------------
# 4. empty Tesseract output → OCRReadError
# ---------------------------------------------------------------------------

def test_ocr_image_raises_on_empty_output(monkeypatch):
    monkeypatch.setattr(
        "accord_ai.extraction.ocr.engine._tesseract_available", lambda: True
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string", lambda img, lang="eng": "   \n  "
    )
    with pytest.raises(OCRReadError, match="unable to extract text"):
        ocr_image(_make_png())
