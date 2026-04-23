"""Verify the compact schema-text generator produces v3-style output.

This is the foundation of the v3 prompt-composition port: schema goes INTO
the system message as compact text (not as raw JSON schema). Generator
tests are unit-level (no LLM, no network).
"""
from pydantic import BaseModel
from typing import List, Literal, Optional

import pytest

from accord_ai.llm.prompts.schema_text import (
    build_schema_text,
    estimate_schema_tokens,
)
from accord_ai.schema import CustomerSubmission


# --- Simple Pydantic models for unit-testing the generator ---------------

class _Address(BaseModel):
    line_one: str
    city: str
    zip_code: str


class _Contact(BaseModel):
    name: str
    phone: Optional[str] = None


class _SimpleSubmission(BaseModel):
    business_name: str
    active: bool
    count: int
    amount: float
    entity_type: Literal["llc", "corp", "partnership"]
    address: _Address
    contacts: List[_Contact]
    description: Optional[str] = None


# --- Simple-model tests (fast, predictable) ------------------------------


def test_generator_emits_simple_types():
    text = build_schema_text(_SimpleSubmission)
    assert '"business_name": str' in text
    assert '"active": bool' in text
    assert '"count": int' in text
    assert '"amount": float' in text


def test_generator_emits_literal_as_union():
    text = build_schema_text(_SimpleSubmission)
    assert '"entity_type":' in text
    # Literal["llc", "corp", "partnership"] → "llc" | "corp" | "partnership"
    assert '"llc"' in text
    assert '"corp"' in text
    assert '"partnership"' in text
    assert " | " in text  # union separator


def test_generator_inlines_nested_object():
    text = build_schema_text(_SimpleSubmission)
    # Nested Address model should be expanded inline (not just "Address")
    assert '"address":' in text
    assert '"line_one": str' in text
    assert '"city": str' in text
    assert '"zip_code": str' in text


def test_generator_renders_list_of_models():
    text = build_schema_text(_SimpleSubmission)
    assert '"contacts": [' in text
    # Contact's fields should appear inside the [...] array
    assert '"name": str' in text


def test_generator_handles_optional_as_nullable():
    text = build_schema_text(_SimpleSubmission)
    # Optional[str] with default None should render as "str" (not "str | null")
    # because the anyOf strips nulls when it's a simple Optional
    assert '"description": str' in text


def test_generator_output_is_smaller_than_raw_json_schema():
    compact = build_schema_text(_SimpleSubmission)
    raw = _SimpleSubmission.model_json_schema()
    import json
    raw_text = json.dumps(raw)
    assert len(compact) < len(raw_text), (
        f"compact {len(compact)} should be smaller than raw {len(raw_text)}"
    )


def test_generator_respects_exclude():
    text = build_schema_text(_SimpleSubmission, exclude=("description",))
    assert '"description"' not in text
    assert '"business_name"' in text  # other fields remain


def test_estimate_schema_tokens_returns_positive_int():
    assert estimate_schema_tokens("x" * 100) > 0
    assert estimate_schema_tokens("") >= 0


# --- Real-model smoke tests (CustomerSubmission) -------------------------


def test_customer_submission_generates_successfully():
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    assert text.startswith("{")
    assert text.endswith("}")
    assert len(text) > 500  # non-trivial
    assert len(text) < 10000  # way smaller than raw JSON schema (~26KB)


def test_customer_submission_contains_critical_fields():
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    # Top-level fields that must be extractable
    required = [
        "business_name", "ein", "entity_type", "mailing_address",
        "contacts", "policy_dates", "lob_details",
    ]
    for field in required:
        assert f'"{field}":' in text, f"missing field: {field}"


def test_customer_submission_discriminated_union_expands():
    # lob_details is a discriminated union — the text should hint at
    # at least one LOB's structure (e.g. vehicles array for CA)
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    assert '"lob_details":' in text
    # At least one LOB-specific field should appear somewhere
    # (could be vehicles, drivers, classifications, payroll_by_class, etc.)
    lob_markers = ["vehicles", "classifications", "payroll_by_class"]
    assert any(m in text for m in lob_markers)


def test_customer_submission_entity_type_enum_rendered():
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    assert '"entity_type":' in text
    # At least "llc" or "corporation" should appear as a quoted option
    assert '"llc"' in text or '"corporation"' in text


def test_customer_submission_conflicts_excluded():
    # conflicts is a system-managed field; LLM should never extract it
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    # Substring check — "conflicts" as a top-level property key
    assert '"conflicts":' not in text


def test_customer_submission_token_budget():
    """Compact schema should fit comfortably in a system prompt."""
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",))
    estimated = estimate_schema_tokens(text)
    # Budget: 4000 tokens for schema leaves room for harness (2000) + user
    # turn (2000+) inside Qwen3.5-9B's 16k ceiling
    assert estimated < 4000, f"schema text too large: {estimated} tokens"


def test_deep_nesting_collapses_at_max_depth():
    text = build_schema_text(CustomerSubmission, exclude=("conflicts",), max_depth=2)
    # With shallower max_depth, some deeper structures collapse
    assert "{...}" in text or len(text) < 2500
