"""Tests for is_session_sft_eligible (Phase 2.7)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from accord_ai.feedback.eligibility import EligibilityReason, is_session_sft_eligible
from accord_ai.validation.types import ValidationFinding, ValidationResult


def _ok_result(name: str = "nhtsa") -> ValidationResult:
    return ValidationResult(
        validator=name,
        ran_at=datetime.now(timezone.utc),
        duration_ms=5.0,
        success=True,
        findings=[],
    )


def _error_result(name: str = "nhtsa") -> ValidationResult:
    return ValidationResult(
        validator=name,
        ran_at=datetime.now(timezone.utc),
        duration_ms=5.0,
        success=True,
        findings=[
            ValidationFinding(
                validator=name,
                field_path="vin",
                severity="error",
                message="VIN not found",
            )
        ],
    )


def _failed_result(name: str = "nhtsa") -> ValidationResult:
    return ValidationResult(
        validator=name,
        ran_at=datetime.now(timezone.utc),
        duration_ms=5000.0,
        success=False,
        error="timed out after 5.0s",
    )


def test_clean_session_is_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[_ok_result()],
        correction_count=0,
    )
    assert result.eligible is True
    assert result.reason == "clean"


def test_non_finalized_not_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="active",
        validation_results=[_ok_result()],
        correction_count=0,
    )
    assert result.eligible is False
    assert "active" in result.reason


def test_with_corrections_not_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[_ok_result()],
        correction_count=2,
    )
    assert result.eligible is False
    assert "2" in result.reason


def test_validation_results_none_not_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=None,
        correction_count=0,
    )
    assert result.eligible is False
    assert "no validation results" in result.reason


def test_validator_error_severity_not_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[_ok_result("zip"), _error_result("nhtsa")],
        correction_count=0,
    )
    assert result.eligible is False
    assert "nhtsa" in result.reason


def test_validator_failed_success_false_not_eligible():
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[_failed_result("nhtsa")],
        correction_count=0,
    )
    assert result.eligible is False
    assert "nhtsa" in result.reason


def test_empty_validation_results_eligible():
    """Empty list = validators ran but none configured — still eligible."""
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[],
        correction_count=0,
    )
    assert result.eligible is True


def test_warning_severity_does_not_block_eligibility():
    """Warning findings are OK; only 'error' severity blocks capture."""
    warning_result = ValidationResult(
        validator="zip",
        ran_at=datetime.now(timezone.utc),
        duration_ms=5.0,
        success=True,
        findings=[
            ValidationFinding(
                validator="zip",
                field_path="zip_code",
                severity="warning",
                message="ZIP+4 not matched",
            )
        ],
    )
    result = is_session_sft_eligible(
        session_id="s1",
        tenant="acme",
        status="finalized",
        validation_results=[warning_result],
        correction_count=0,
    )
    assert result.eligible is True
