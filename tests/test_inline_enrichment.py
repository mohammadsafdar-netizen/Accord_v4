"""Tests for Phase 1.6.B inline enrichment runner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accord_ai.schema import CommercialAutoDetails, CustomerSubmission, Vehicle
from accord_ai.validation.inline import InlineEnrichmentRunner, _apply_patch, _deep_merge
from accord_ai.validation.types import PrefillPatch, ValidationFinding, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InlinePassValidator:
    name: str = "fill_biz_name"
    applicable_fields = ("business_name",)
    inline_eligible = True

    async def run(self, sub):
        return ValidationResult(validator=self.name, ran_at=datetime.now(tz=timezone.utc),
                                duration_ms=0, success=True)

    async def prefill(self, sub, just_extracted):
        if "business_name" in just_extracted:
            return PrefillPatch(
                patch={"dba": "FillBot"},
                source=self.name,
                confidence=0.9,
            )
        return None


class _FinalizeOnlyValidator:
    name: str = "finalize_only"
    applicable_fields = ("business_name",)
    inline_eligible = False

    async def run(self, sub):
        return ValidationResult(validator=self.name, ran_at=datetime.now(tz=timezone.utc),
                                duration_ms=0, success=True)

    async def prefill(self, sub, just_extracted):
        raise AssertionError("prefill should never be called on finalize_only validator")


class _SlowInlineValidator:
    name: str = "slow_inline"
    applicable_fields = ("business_name",)
    inline_eligible = True

    async def run(self, sub):
        return ValidationResult(validator=self.name, ran_at=datetime.now(tz=timezone.utc),
                                duration_ms=0, success=True)

    async def prefill(self, sub, just_extracted):
        await asyncio.sleep(999)
        return PrefillPatch(patch={"dba": "never"}, source=self.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_fires_only_inline_eligible():
    """Finalize-only validators must not be called via prefill during a turn."""
    runner = InlineEnrichmentRunner(
        validators=[_InlinePassValidator(), _FinalizeOnlyValidator()],
        timeout_s=1.0,
    )
    sub = CustomerSubmission(business_name="Acme")
    enriched, conflicts = await runner.enrich(sub, {"business_name": "Acme"})
    # Inline validator ran; finalize-only was silently skipped (no assertion error)
    assert enriched.dba == "FillBot"
    assert conflicts == []


@pytest.mark.asyncio
async def test_runner_timeout_doesnt_break_turn():
    """A slow validator must time out and leave the submission unchanged."""
    runner = InlineEnrichmentRunner(
        validators=[_SlowInlineValidator()],
        timeout_s=0.05,
    )
    sub = CustomerSubmission(business_name="Acme")
    enriched, conflicts = await runner.enrich(sub, {"business_name": "Acme"})
    assert enriched is sub
    assert conflicts == []


@pytest.mark.asyncio
async def test_runner_disabled_returns_submission_unchanged():
    """ACCORD_INLINE_ENRICHMENT=false → submission unchanged, no prefill calls."""
    runner = InlineEnrichmentRunner(
        validators=[_InlinePassValidator()],
        timeout_s=1.0,
        enabled=False,
    )
    sub = CustomerSubmission(business_name="Acme")
    enriched, conflicts = await runner.enrich(sub, {"business_name": "Acme"})
    assert enriched is sub
    assert enriched.dba is None


@pytest.mark.asyncio
async def test_multiple_validators_run_parallel():
    """Two inline validators both apply their patches."""
    class _V1:
        name = "v1"
        applicable_fields = ("business_name",)
        inline_eligible = True
        async def run(self, s): ...
        async def prefill(self, s, j):
            return PrefillPatch(patch={"dba": "from_v1"}, source="v1")

    class _V2:
        name = "v2"
        applicable_fields = ("business_name",)
        inline_eligible = True
        async def run(self, s): ...
        async def prefill(self, s, j):
            return PrefillPatch(patch={"website": "http://example.com"}, source="v2")

    runner = InlineEnrichmentRunner(validators=[_V1(), _V2()], timeout_s=1.0)
    sub = CustomerSubmission(business_name="Acme")
    enriched, _ = await runner.enrich(sub, {"business_name": "Acme"})
    assert enriched.dba == "from_v1"
    assert enriched.website == "http://example.com"


def test_deep_merge_handles_index_keyed_lists():
    """_deep_merge must update specific list items by integer key."""
    target = {"lob_details": {"vehicles": [{"vin": "ABC", "year": None}, {"vin": "DEF"}]}}
    source = {"lob_details": {"vehicles": {0: {"year": 2024}}}}
    _deep_merge(target, source)
    assert target["lob_details"]["vehicles"][0]["year"] == 2024
    assert target["lob_details"]["vehicles"][1]["vin"] == "DEF"
