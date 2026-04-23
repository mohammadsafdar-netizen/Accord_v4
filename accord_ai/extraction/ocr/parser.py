"""LLM-based structured parser for OCR raw text.

Two-stage OCR pipeline (Phase 1.5):
  Stage 1 — Tesseract extracts raw text (engine.py)
  Stage 2 — Qwen3.5-9B parses raw text into typed fields (this module)

Each parser uses the existing Engine with guided_json so the LLM is
constrained to emit valid schema JSON. max_tokens=512 — these outputs are
small (<20 fields, all short strings or null).

Privacy: the raw OCR text contains PII. This module never logs it; callers
that need to log should run redact_pii_text() first.
"""
from __future__ import annotations

import json
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field

from accord_ai.llm.engine import Engine


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class DriverLicenseFields(BaseModel):
    first_name:          Optional[str]  = None
    last_name:           Optional[str]  = None
    date_of_birth:       Optional[date] = None
    license_number:      Optional[str]  = None
    license_state:       Optional[str]  = None
    license_class:       Optional[str]  = None
    license_expiration:  Optional[date] = None
    address:             Optional[str]  = None


class InsuranceCardFields(BaseModel):
    carrier:         Optional[str]  = None
    policy_number:   Optional[str]  = None
    effective_date:  Optional[date] = None
    expiration_date: Optional[date] = None
    named_insured:   Optional[str]  = None


class RegistrationFields(BaseModel):
    vin:                      Optional[str]  = None
    year:                     Optional[int]  = None
    make:                     Optional[str]  = None
    model:                    Optional[str]  = None
    registration_state:       Optional[str]  = None
    registration_expiration:  Optional[date] = None
    owner_name:               Optional[str]  = None


# ---------------------------------------------------------------------------
# Parser prompts
# ---------------------------------------------------------------------------

_DL_SYSTEM = (
    "You are an insurance data extractor. "
    "Extract driver license fields from the OCR text below. "
    "Fields not present in the text → null. "
    "Dates must be in ISO-8601 format (YYYY-MM-DD). "
    "Output ONLY the JSON object — no preamble, no markdown fences."
)

_IC_SYSTEM = (
    "You are an insurance data extractor. "
    "Extract insurance card fields from the OCR text below. "
    "Fields not present in the text → null. "
    "Dates must be in ISO-8601 format (YYYY-MM-DD). "
    "Output ONLY the JSON object — no preamble, no markdown fences."
)

_REG_SYSTEM = (
    "You are an insurance data extractor. "
    "Extract vehicle registration fields from the OCR text below. "
    "VIN must be exactly 17 alphanumeric characters if present. "
    "Fields not present in the text → null. "
    "Dates must be in ISO-8601 format (YYYY-MM-DD). "
    "Output ONLY the JSON object — no preamble, no markdown fences."
)


# ---------------------------------------------------------------------------
# Shared call helper
# ---------------------------------------------------------------------------

async def _parse(
    text: str,
    system: str,
    schema_cls: type,
    engine: Engine,
) -> object:
    """Call the engine with guided_json and parse the response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"OCR text:\n{text}"},
    ]
    response = await engine.generate(
        messages,
        max_tokens=512,
        json_schema=schema_cls.model_json_schema(),
    )
    # guided_json guarantees valid JSON matching the schema
    return schema_cls.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

async def parse_driver_license(text: str, engine: Engine) -> DriverLicenseFields:
    return await _parse(text, _DL_SYSTEM, DriverLicenseFields, engine)


async def parse_insurance_card(text: str, engine: Engine) -> InsuranceCardFields:
    return await _parse(text, _IC_SYSTEM, InsuranceCardFields, engine)


async def parse_registration(text: str, engine: Engine) -> RegistrationFields:
    return await _parse(text, _REG_SYSTEM, RegistrationFields, engine)
