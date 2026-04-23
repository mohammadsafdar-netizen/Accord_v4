"""Multi-model refiner cascade (Phase 1.8).

Priority order: Gemini → Claude → Local Qwen.

Each RefinerClient.refine() returns Optional[CustomerSubmission]:
  - None  → fall through to next client
  - value → return immediately

PII redaction: external clients (Gemini, Claude) are responsible for
redacting their own HTTP payloads using redact_dict() before any outbound
call.  LocalRefiner sends nothing external — no redaction needed.

_reapply_patch_to_original merges a patch dict back into the original
(non-redacted) submission so that PII in untouched fields is preserved.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Protocol, Tuple, Union, runtime_checkable

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("refiner_cascade")

# ---------------------------------------------------------------------------
# PII redaction helpers
# ---------------------------------------------------------------------------

_PII_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{2}-\d{7}\b"),                                  "<ein>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                           "<ssn>"),
    (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),                          "<vin>"),
    (re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "<phone>"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                            "<dob>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),                        "<dob>"),
]


def _redact_str(s: str) -> str:
    for pattern, tag in _PII_RULES:
        s = pattern.sub(tag, s)
    return s


def redact_dict(obj: Any) -> Any:
    """Recursively redact PII from strings inside dicts and lists."""
    if isinstance(obj, dict):
        return {k: redact_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_dict(item) for item in obj]
    if isinstance(obj, str):
        return _redact_str(obj)
    return obj


# ---------------------------------------------------------------------------
# Patch merge
# ---------------------------------------------------------------------------

def _deep_merge(target: dict, source: dict) -> None:
    for key, val in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(val, dict):
            _deep_merge(target[key], val)
        else:
            target[key] = val


def _reapply_patch_to_original(
    original: CustomerSubmission, patch: dict
) -> CustomerSubmission:
    """Merge patch into original's full dict, return a new CustomerSubmission.

    Keys present in patch override the corresponding field in original.
    Keys absent from patch keep the original (non-redacted) value.
    """
    base = original.model_dump()
    _deep_merge(base, patch)
    return CustomerSubmission.model_validate(base)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RefinerClient(Protocol):
    provider: str

    async def refine(
        self,
        *,
        original_user_message: str,
        current_submission: CustomerSubmission,
        verdict: JudgeVerdict,
    ) -> Optional[CustomerSubmission]:
        ...


# ---------------------------------------------------------------------------
# CascadingRefiner
# ---------------------------------------------------------------------------

class CascadingRefiner:
    """Tries each RefinerClient in order; returns the first non-None result.

    Falls through on:
      - client returning None
      - any exception raised by the client

    If all clients fail, returns the original submission unchanged.
    """

    def __init__(self, clients: List[RefinerClient]) -> None:
        self._clients = list(clients)

    # Expose module-level utilities as instance methods for testability.
    def redact_dict(self, obj: Any) -> Any:
        return redact_dict(obj)

    def _reapply_patch_to_original(
        self, original: CustomerSubmission, patch: dict
    ) -> CustomerSubmission:
        return _reapply_patch_to_original(original, patch)

    async def refine(
        self,
        *,
        original_user_message: str,
        current_submission: CustomerSubmission,
        verdict: JudgeVerdict,
    ) -> CustomerSubmission:
        for client in self._clients:
            try:
                result = await client.refine(
                    original_user_message=original_user_message,
                    current_submission=current_submission,
                    verdict=verdict,
                )
            except Exception as exc:
                _logger.warning(
                    "cascade: provider=%s raised %s — falling through",
                    client.provider,
                    type(exc).__name__,
                )
                continue

            if result is None:
                _logger.debug(
                    "cascade: provider=%s returned None — falling through",
                    client.provider,
                )
                continue

            _logger.info("cascade: provider=%s succeeded", client.provider)
            return result

        _logger.warning("cascade: all %d providers failed — returning original", len(self._clients))
        return current_submission
