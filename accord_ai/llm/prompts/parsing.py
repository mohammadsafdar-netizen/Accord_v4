"""Shared helpers for parsing LLM output.

LLMs often emit structured content wrapped in markdown code fences despite
explicit instructions not to. This module provides minimal normalizers
reusable across prompt families (refiner, future extractor, any LLM-judge).
"""
from __future__ import annotations

import re

# Regex explanation:
#   ^\s*          leading whitespace
#   ```           opening fence
#   (?:json)?     optional lowercase/uppercase "json" label
#   \s*\n?        optional whitespace + optional newline
#                 (the newline is optional so single-line ```json{"x":1}```
#                  also unwraps)
#   (.*?)         lazy content capture
#   \n?\s*        optional newline + optional whitespace
#   ```           closing fence
#   \s*$          trailing whitespace, end of string
# DOTALL lets . match newlines; IGNORECASE lets the json label be mixed case.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Finds the first {...} block in free-form prose — used as a fallback when
# fence-stripping leaves non-JSON text (e.g. FREE extraction mode where the
# model may prefix the JSON with a sentence). Matches the outermost braces.
_FIRST_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove surrounding ```...``` fences (with optional 'json' label).

    Leaves text alone if no fences are present, if fences aren't balanced,
    or if the input is empty. Idempotent — applying twice is the same as
    once, since the returned value contains no outer fences.
    """
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


# ---------------------------------------------------------------------------
# Shared: LLM text -> CustomerSubmission
# ---------------------------------------------------------------------------

import json as _json
from typing import Any, Callable, Dict, Optional, Type

from pydantic import ValidationError as _ValidationError

from accord_ai.schema import CustomerSubmission as _CustomerSubmission


# Type alias for an optional dict-level postprocess hook. Caller passes a
# function ``(delta_dict) -> delta_dict`` that runs after JSON-parse and
# before pydantic-validate. Used by Extractor.extract to plug in the
# 5-step postprocess pipeline (unfold/strip/phantom/coerce/cap) without
# every consumer of parse_submission_output needing to opt in.
_DeltaProcessor = Callable[[Dict[str, Any]], Dict[str, Any]]


def parse_submission_output(
    text: str,
    *,
    error_cls: Type[ValueError] = ValueError,
    postprocess: Optional[_DeltaProcessor] = None,
) -> _CustomerSubmission:
    """Strip fences, JSON-parse, optionally postprocess, validate.

    Raises `error_cls` (defaults to ValueError) with `from`-chained original
    exception on non-JSON output or schema-validation failure. Callers pass
    their module-specific subtype (ExtractionOutputError, RefinerOutputError).

    The optional `postprocess` hook runs on the raw dict between JSON-parse
    and pydantic-validate. If it raises, the failure surfaces as a
    schema-validation error (postprocess bugs shouldn't propagate as
    JSON-decode errors).
    """
    raw = strip_code_fences(text)
    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError as first_err:
        # Fallback: find the first {...} block in the text. Handles FREE
        # extraction mode where the model may emit prose before/after JSON.
        m = _FIRST_JSON_BLOCK_RE.search(raw)
        if m:
            try:
                data = _json.loads(m.group(0))
            except _json.JSONDecodeError:
                raise error_cls(f"non-JSON output: {first_err}") from first_err
        else:
            raise error_cls(f"non-JSON output: {first_err}") from first_err
    if postprocess is not None:
        try:
            data = postprocess(data)
        except Exception as e:
            raise error_cls(f"postprocess failed: {e}") from e
    try:
        return _CustomerSubmission.model_validate(data)
    except _ValidationError as e:
        raise error_cls("schema validation failed") from e
