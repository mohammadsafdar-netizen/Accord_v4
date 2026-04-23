"""Tests for Phase 1.6.A — Validator Framework + OFAC + /enrich wire-up."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission, GeneralLiabilityDetails, WorkersCompDetails
from accord_ai.validation.engine import ValidationEngine, build_engine
from accord_ai.validation.ofac import (
    OFACValidator,
    _check_name,
    _normalize,
    _tokens_of,
    load_index_from_file,
)
from accord_ai.validation.types import (
    ValidationFinding,
    ValidationResult,
    Validator,
)

_FIXTURE_CSV = (
    Path(__file__).parent / "fixtures" / "validation" / "sdn_synthetic.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )


def _make_app(tmp_path):
    settings = _settings(tmp_path)
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake), intake


def _submission(business_name: str = "Acme LLC") -> CustomerSubmission:
    return CustomerSubmission(business_name=business_name)


# ---------------------------------------------------------------------------
# 1–4: ValidationEngine
# ---------------------------------------------------------------------------


class _AlwaysPassValidator:
    name: str = "always_pass"
    applicable_fields: List[str] = []

    async def run(self, submission: Any) -> ValidationResult:
        from datetime import datetime, timezone
        return ValidationResult(
            validator=self.name,
            ran_at=datetime.now(tz=timezone.utc),
            duration_ms=1.0,
            success=True,
            findings=[],
        )


class _AlwaysErrorValidator:
    name: str = "always_error"
    applicable_fields: List[str] = []

    async def run(self, submission: Any) -> ValidationResult:
        raise RuntimeError("validator exploded")


class _SlowValidator:
    name: str = "slow"
    applicable_fields: List[str] = []

    async def run(self, submission: Any) -> ValidationResult:
        await asyncio.sleep(999)
        from datetime import datetime, timezone
        return ValidationResult(
            validator=self.name,
            ran_at=datetime.now(tz=timezone.utc),
            duration_ms=0.0,
            success=True,
        )


@pytest.mark.asyncio
async def test_engine_empty_validators_returns_no_results():
    engine = ValidationEngine(validators=[])
    results = await engine.run_all(_submission())
    assert results == []


@pytest.mark.asyncio
async def test_engine_runs_validator_and_returns_result():
    engine = ValidationEngine(validators=[_AlwaysPassValidator()])
    results = await engine.run_all(_submission())
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].validator == "always_pass"


@pytest.mark.asyncio
async def test_engine_validator_exception_returns_failure_result():
    engine = ValidationEngine(validators=[_AlwaysErrorValidator()])
    results = await engine.run_all(_submission())
    assert len(results) == 1
    assert results[0].success is False
    assert "validator exploded" in (results[0].error or "")


@pytest.mark.asyncio
async def test_engine_timeout_returns_failure_result():
    engine = ValidationEngine(validators=[_SlowValidator()], timeout_s=0.05)
    results = await engine.run_all(_submission())
    assert len(results) == 1
    assert results[0].success is False
    assert "timed out" in (results[0].error or "")


# ---------------------------------------------------------------------------
# 5–10: OFACValidator
# ---------------------------------------------------------------------------


def _load_fixture():
    load_index_from_file(_FIXTURE_CSV)


def test_ofac_normalize_strips_corp_suffix():
    assert _normalize("Iran Petroleum Corp") == "iran petroleum"


@pytest.mark.asyncio
async def test_ofac_exact_match_returns_error_finding():
    _load_fixture()
    sub = CustomerSubmission(business_name="Iran Petroleum Corp")
    result = await OFACValidator().run(sub)
    assert result.success is True
    errors = [f for f in result.findings if f.severity == "error"]
    assert len(errors) >= 1
    assert errors[0].field_path == "business_name"


@pytest.mark.asyncio
async def test_ofac_no_match_returns_no_findings():
    _load_fixture()
    sub = CustomerSubmission(business_name="Rainbow Flower Shop")
    result = await OFACValidator().run(sub)
    assert result.success is True
    assert result.findings == []


@pytest.mark.asyncio
async def test_ofac_screens_contacts_full_name():
    _load_fixture()
    from accord_ai.schema import Contact
    sub = CustomerSubmission(
        business_name="Totally Legit LLC",
        contacts=[Contact(full_name="Hassan Rahimi")],
    )
    result = await OFACValidator().run(sub)
    assert result.success is True
    matching = [f for f in result.findings if "contacts" in f.field_path]
    assert len(matching) >= 1


@pytest.mark.asyncio
async def test_ofac_warning_for_partial_match():
    _load_fixture()
    # "North Korea Trading" — partial overlap with "North Korea Trading Company"
    sub = CustomerSubmission(business_name="North Korea Trading")
    result = await OFACValidator().run(sub)
    assert result.success is True
    # High overlap (3/4 tokens) → score > 0.6 → at least a warning
    assert len(result.findings) >= 1


@pytest.mark.asyncio
async def test_ofac_empty_business_name_no_findings():
    _load_fixture()
    sub = CustomerSubmission()
    result = await OFACValidator().run(sub)
    assert result.success is True
    assert result.findings == []


# ---------------------------------------------------------------------------
# 11–13: /enrich API endpoint
# ---------------------------------------------------------------------------


def _seed_session(intake) -> str:
    sid = intake.store.create_session()
    sub = CustomerSubmission(
        business_name="Clean Trucking Solutions LLC",
        lob_details=GeneralLiabilityDetails(),
    )
    intake.store.update_submission(sid, sub)
    return sid


def test_enrich_returns_ok_with_zero_validators(tmp_path):
    """Default (ENABLE_EXTERNAL_VALIDATION=false) → validators_run=0."""
    app, intake = _make_app(tmp_path)
    sid = _seed_session(intake)

    with TestClient(app) as client:
        r = client.post("/enrich", json={"session_id": sid})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["validators_run"] == 0
    assert body["results"] == []
    assert body["cached"] is False


def test_enrich_unknown_session_returns_404(tmp_path):
    app, intake = _make_app(tmp_path)

    with TestClient(app) as client:
        r = client.post("/enrich", json={"session_id": "does-not-exist"})

    assert r.status_code == 404


def test_enrich_caches_result_on_second_call(tmp_path, monkeypatch):
    """Second call with identical submission returns cached=True."""
    app, intake = _make_app(tmp_path)
    sid = _seed_session(intake)

    with TestClient(app) as client:
        r1 = client.post("/enrich", json={"session_id": sid})
        r2 = client.post("/enrich", json={"session_id": sid})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is True
