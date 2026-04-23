"""Phase 1.8 — CascadingRefiner tests.

Covers:
  1. cascade tries first client before second
  2. fallthrough on None
  3. fallthrough on exception
  4. all fail → original returned
  5. PII (EIN) not in Gemini HTTP payload
  6. patch reapplied to original preserves EIN
  7. build_refiner → None when ACCORD_DISABLE_REFINEMENT set
  8. build_refiner → CascadingRefiner when flag absent
  9. Gemini timeout → returns None (cascade falls through)
 10. redact_dict nested list
 11. LocalRefiner → None on RefinerOutputError
 12. _reapply_patch_to_original direct unit test
"""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock

import pytest

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.harness.refiner_cascade import (
    CascadingRefiner,
    _reapply_patch_to_original,
    redact_dict,
)
from accord_ai.schema import CustomerSubmission


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_verdict(passed: bool = False) -> JudgeVerdict:
    if passed:
        return JudgeVerdict(passed=True)
    return JudgeVerdict(
        passed=False,
        reasons=("business_name is required",),
        failed_paths=("business_name",),
    )


def _make_submission(**kwargs) -> CustomerSubmission:
    return CustomerSubmission(**kwargs)


class _FakeClient:
    """Minimal RefinerClient-protocol compatible test double."""

    def __init__(self, provider: str, result) -> None:
        self.provider = provider
        self._result = result
        self.call_count = 0

    async def refine(self, **kwargs) -> Optional[CustomerSubmission]:
        self.call_count += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


# ---------------------------------------------------------------------------
# 1. cascade tries first client before second
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cascade_tries_first_client():
    sub = _make_submission(business_name="Acme")
    fixed = _make_submission(business_name="Acme LLC")
    first = _FakeClient("first", fixed)
    second = _FakeClient("second", fixed)

    cascade = CascadingRefiner([first, second])
    result = await cascade.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert result is fixed
    assert first.call_count == 1
    assert second.call_count == 0


# ---------------------------------------------------------------------------
# 2. fallthrough on None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cascade_fallthrough_on_none():
    sub = _make_submission(business_name="Acme")
    fixed = _make_submission(business_name="Acme LLC")
    first = _FakeClient("first", None)
    second = _FakeClient("second", fixed)

    cascade = CascadingRefiner([first, second])
    result = await cascade.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert result is fixed
    assert first.call_count == 1
    assert second.call_count == 1


# ---------------------------------------------------------------------------
# 3. fallthrough on exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cascade_fallthrough_on_exception():
    sub = _make_submission(business_name="Acme")
    fixed = _make_submission(business_name="Acme LLC")
    first = _FakeClient("first", RuntimeError("boom"))
    second = _FakeClient("second", fixed)

    cascade = CascadingRefiner([first, second])
    result = await cascade.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert result is fixed
    assert first.call_count == 1
    assert second.call_count == 1


# ---------------------------------------------------------------------------
# 4. all fail → original returned unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cascade_all_fail_returns_original():
    sub = _make_submission(business_name="Acme")
    first = _FakeClient("first", None)
    second = _FakeClient("second", None)
    third = _FakeClient("third", RuntimeError("still broken"))

    cascade = CascadingRefiner([first, second, third])
    result = await cascade.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert result is sub  # exact same object — nothing changed


# ---------------------------------------------------------------------------
# 5. PII (EIN) not in Gemini HTTP payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pii_ein_not_in_gemini_http_payload(monkeypatch):
    captured: dict = {}

    class _FakeHTTPClient:
        def __init__(self, **kwargs): pass

        async def __aenter__(self): return self

        async def __aexit__(self, *args): pass

        async def post(self, url, *, json=None, **kwargs):
            captured["payload"] = json
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": '{"business_name": "Patched"}'}]}}
                ]
            }
            return resp

    import accord_ai.harness.refiner_clients.gemini as _gemini_mod
    monkeypatch.setattr(_gemini_mod, "httpx", MagicMock(AsyncClient=_FakeHTTPClient))

    from accord_ai.harness.refiner_clients.gemini import GeminiRefiner

    sub = _make_submission(business_name="Acme", ein="12-3456789")
    refiner = GeminiRefiner(api_key="fake-key")
    await refiner.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert captured, "HTTP call was not made"
    raw_payload = json.dumps(captured["payload"])
    assert "12-3456789" not in raw_payload, "Raw EIN must not appear in outbound payload"
    assert "<ein>" in raw_payload, "Redacted EIN placeholder expected in payload"


# ---------------------------------------------------------------------------
# 6. patch reapplied to original — original EIN preserved (not redacted token)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patch_reapplied_preserves_original_ein(monkeypatch):
    class _FakeHTTPClient:
        def __init__(self, **kwargs): pass

        async def __aenter__(self): return self

        async def __aexit__(self, *args): pass

        async def post(self, url, *, json=None, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            # Patch only business_name — not ein
            resp.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": '{"business_name": "Fixed Corp"}'}]}}
                ]
            }
            return resp

    import accord_ai.harness.refiner_clients.gemini as _gemini_mod
    monkeypatch.setattr(_gemini_mod, "httpx", MagicMock(AsyncClient=_FakeHTTPClient))

    from accord_ai.harness.refiner_clients.gemini import GeminiRefiner

    sub = _make_submission(business_name="Acme", ein="12-3456789")
    refiner = GeminiRefiner(api_key="fake-key")
    result = await refiner.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )

    assert result is not None
    assert result.business_name == "Fixed Corp"
    # Original EIN (non-redacted) must be preserved
    assert result.ein == "12-3456789"


# ---------------------------------------------------------------------------
# 7. build_refiner returns None when ACCORD_DISABLE_REFINEMENT is set
# ---------------------------------------------------------------------------

def test_build_refiner_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("ACCORD_DISABLE_REFINEMENT", "true")

    from accord_ai.harness.refiner import build_refiner
    from accord_ai.config import Settings

    result = build_refiner(Settings())
    assert result is None


# ---------------------------------------------------------------------------
# 8. build_refiner returns CascadingRefiner when flag absent
# ---------------------------------------------------------------------------

def test_build_refiner_returns_cascade_when_enabled(monkeypatch):
    monkeypatch.setenv("ACCORD_DISABLE_REFINEMENT", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from accord_ai.harness.refiner import build_refiner
    from accord_ai.config import Settings

    result = build_refiner(Settings())
    assert isinstance(result, CascadingRefiner)


# ---------------------------------------------------------------------------
# 9. GeminiRefiner timeout → returns None (cascade falls through)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_timeout_returns_none(monkeypatch):
    import httpx as _httpx

    class _TimeoutClient:
        def __init__(self, **kwargs): pass

        async def __aenter__(self): return self

        async def __aexit__(self, *args): pass

        async def post(self, *args, **kwargs):
            raise _httpx.TimeoutException("timed out")

    import accord_ai.harness.refiner_clients.gemini as _gemini_mod
    monkeypatch.setattr(_gemini_mod, "httpx", MagicMock(
        AsyncClient=_TimeoutClient,
        TimeoutException=_httpx.TimeoutException,
    ))

    from accord_ai.harness.refiner_clients.gemini import GeminiRefiner

    sub = _make_submission(business_name="Acme")
    refiner = GeminiRefiner(api_key="fake-key")

    # TimeoutException propagates as a generic Exception; GeminiRefiner catches
    # it because httpx.AsyncClient.post raises inside the try block.
    # The method should fall through by returning None.
    result = await refiner.refine(
        original_user_message="hi",
        current_submission=sub,
        verdict=_make_verdict(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# 10. redact_dict handles nested lists
# ---------------------------------------------------------------------------

def test_redact_dict_nested_list():
    obj = {
        "contacts": [
            {"name": "Alice", "ein": "12-3456789", "email": "alice@example.com"},
            {"name": "Bob", "ein": "98-7654321"},
        ],
        "business_name": "Acme",
    }
    redacted = redact_dict(obj)

    assert redacted["business_name"] == "Acme"
    assert redacted["contacts"][0]["name"] == "Alice"
    assert redacted["contacts"][0]["ein"] == "<ein>"
    assert redacted["contacts"][0]["email"] == "<email>"
    assert redacted["contacts"][1]["ein"] == "<ein>"
    # Raw EINs not present anywhere in the output
    assert "12-3456789" not in json.dumps(redacted)
    assert "98-7654321" not in json.dumps(redacted)


# ---------------------------------------------------------------------------
# 11. LocalRefiner returns None on RefinerOutputError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_refiner_returns_none_on_refiner_output_error():
    from accord_ai.harness.refiner import RefinerOutputError
    from accord_ai.harness.refiner_clients.local import LocalRefiner

    class _BadRefiner:
        async def refine(self, **kwargs):
            raise RefinerOutputError("bad output")

    local = LocalRefiner(_BadRefiner())  # type: ignore[arg-type]
    result = await local.refine(
        original_user_message="hi",
        current_submission=_make_submission(business_name="Acme"),
        verdict=_make_verdict(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# 12. _reapply_patch_to_original merges fields correctly
# ---------------------------------------------------------------------------

def test_reapply_patch_to_original_merges():
    original = _make_submission(business_name="Old Name", ein="12-3456789")
    patch = {"business_name": "New Name"}

    result = _reapply_patch_to_original(original, patch)

    assert result.business_name == "New Name"
    assert result.ein == "12-3456789"   # preserved from original
