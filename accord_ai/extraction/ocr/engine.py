"""Tesseract OCR engine wrapper.

Local-only — no cloud calls. All image data stays on the host.

Preprocessing pipeline (applied before OCR):
  1. Convert to RGB — handles HEIC/CMYK/palette modes gracefully.
  2. Upscale if longest edge < 1500px — phone photos are often 800–1200px
     and Tesseract accuracy drops sharply below that threshold.
  3. Convert to grayscale — reduces colour noise that confuses Tesseract.

Errors:
  OCRConfigError — tesseract binary not installed; caller returns 503.
  OCRReadError   — image unreadable, or Tesseract returned empty text; 422.
"""
from __future__ import annotations

import io

from accord_ai.extraction.ocr.errors import OCRConfigError, OCRReadError


def _tesseract_available() -> bool:
    import shutil
    return shutil.which("tesseract") is not None


def _preprocess(img):
    """Return a preprocessed PIL image suitable for Tesseract."""
    img = img.convert("RGB")
    if max(img.size) < 1500:
        ratio = 1500 / max(img.size)
        from PIL import Image as _PILImage
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            _PILImage.LANCZOS,
        )
    img = img.convert("L")  # grayscale
    return img


def ocr_image(image_bytes: bytes) -> str:
    """Return raw OCR text from image bytes.

    Args:
        image_bytes: Raw bytes of a JPG/PNG/HEIC/WebP image.

    Returns:
        Raw text string extracted by Tesseract (may be noisy).

    Raises:
        OCRConfigError: tesseract binary not found.
        OCRReadError:   bytes are not a valid image, or no text was extracted.
    """
    if not _tesseract_available():
        raise OCRConfigError(
            "tesseract binary not found — run: sudo apt install tesseract-ocr tesseract-ocr-eng"
        )

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise OCRConfigError(f"Pillow not installed: {exc}") from exc

    try:
        import pytesseract
    except ImportError as exc:
        raise OCRConfigError(f"pytesseract not installed: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise OCRReadError(f"cannot open image: {exc}") from exc

    img = _preprocess(img)

    try:
        text = pytesseract.image_to_string(img, lang="eng")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRConfigError(
            "tesseract binary not found — run: sudo apt install tesseract-ocr tesseract-ocr-eng"
        ) from exc
    except Exception as exc:
        raise OCRReadError(f"tesseract error: {exc}") from exc

    if not text.strip():
        raise OCRReadError("unable to extract text — image may be blank, corrupt, or too low-contrast")

    return text
