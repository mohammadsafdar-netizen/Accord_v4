"""OCR facade — image bytes → structured fields (Phase 1.5).

Two-stage pipeline:
  1. Tesseract extracts raw text (engine.py) — local, CPU-only.
  2. Qwen3.5-9B parses raw text → typed Pydantic fields (parser.py).

All processing is local. No PII leaves the host.

Usage::

    from accord_ai.extraction.ocr import ocr_document
    fields = await ocr_document(image_bytes, "drivers_license", engine)
    # returns DriverLicenseFields | InsuranceCardFields | RegistrationFields
"""
from __future__ import annotations

from typing import Literal, Union

from accord_ai.extraction.ocr.engine import ocr_image
from accord_ai.extraction.ocr.errors import OCRConfigError, OCRReadError
from accord_ai.extraction.ocr.parser import (
    DriverLicenseFields,
    InsuranceCardFields,
    RegistrationFields,
    parse_driver_license,
    parse_insurance_card,
    parse_registration,
)
from accord_ai.llm.engine import Engine

DocKind = Literal["drivers_license", "insurance_card", "vehicle_registration"]

OCRResult = Union[DriverLicenseFields, InsuranceCardFields, RegistrationFields]

__all__ = [
    "ocr_document",
    "DocKind",
    "OCRResult",
    "OCRReadError",
    "OCRConfigError",
    "DriverLicenseFields",
    "InsuranceCardFields",
    "RegistrationFields",
]


async def ocr_document(
    image_bytes: bytes,
    kind: DocKind,
    engine: Engine,
) -> OCRResult:
    """Extract structured fields from an image.

    Args:
        image_bytes: Raw image bytes (JPG/PNG/HEIC/WebP).
        kind:        Document type — determines which parser is used.
        engine:      LLM engine for the structured-parse stage.

    Returns:
        Typed Pydantic model (DriverLicenseFields, InsuranceCardFields, or
        RegistrationFields). Fields absent in the image are None.

    Raises:
        OCRReadError:   Bytes are not a valid image, or no text was extracted.
        OCRConfigError: tesseract binary is not installed.
        ValueError:     Unknown kind value.
    """
    text = ocr_image(image_bytes)  # raises OCRReadError / OCRConfigError

    if kind == "drivers_license":
        return await parse_driver_license(text, engine)
    if kind == "insurance_card":
        return await parse_insurance_card(text, engine)
    if kind == "vehicle_registration":
        return await parse_registration(text, engine)

    raise ValueError(f"unknown kind: {kind!r}")
