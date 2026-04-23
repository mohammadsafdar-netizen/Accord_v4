"""Tests for build_review_payload transformer (Phase 1.6.E) — 8 tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from accord_ai.schema import CustomerSubmission, FieldConflict
from accord_ai.validation.review import (
    ReviewPayload,
    build_review_payload,
)
from accord_ai.validation.types import ValidationFinding, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(validator: str, findings=None, success=True) -> ValidationResult:
    return ValidationResult(
        validator=validator,
        ran_at=datetime.now(tz=timezone.utc),
        duration_ms=0.0,
        success=success,
        findings=findings or [],
    )


def _finding(severity, message="test", validator="cross_field", field_path="test") -> ValidationFinding:
    return ValidationFinding(
        validator=validator,
        field_path=field_path,
        severity=severity,
        message=message,
    )


def _empty_sub() -> CustomerSubmission:
    return CustomerSubmission()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_review_builds_conflicts_from_submission_conflicts_list():
    """FieldConflict entries on submission → appear in ReviewPayload.conflicts."""
    sub = CustomerSubmission(
        conflicts=[
            FieldConflict(
                field_path="lob_details.vehicles[0].year",
                user_value=2023,
                enriched_value=2024,
                source="nhtsa_vpic",
            )
        ]
    )
    payload = build_review_payload("sid-1", sub, [])
    assert len(payload.conflicts) == 1
    c = payload.conflicts[0]
    assert c.field_path == "lob_details.vehicles[0].year"
    assert c.user_value == 2023
    assert c.enriched_value == 2024
    assert c.source == "nhtsa_vpic"
    assert c.id  # uuid generated


def test_review_compliance_bucket_groups_ofac_fmcsa():
    """OFAC and FMCSA results go into compliance bucket, not warnings."""
    ofac_result = _result("ofac_sdn", findings=[
        _finding("info", "No SDN matches", validator="ofac_sdn", field_path="business_name")
    ])
    fmcsa_result = _result("fmcsa_safer", findings=[
        _finding("warning", "Safety rating: CONDITIONAL", validator="fmcsa_safer",
                 field_path="lob_details.fmcsa_dot_number")
    ])
    payload = build_review_payload("sid-2", _empty_sub(), [ofac_result, fmcsa_result])

    comp_validators = {c.validator for c in payload.compliance}
    assert "ofac_sdn" in comp_validators
    assert "fmcsa_safer" in comp_validators
    # compliance items must NOT appear in warnings
    warn_validators = {f.validator for f in payload.warnings}
    assert "ofac_sdn" not in warn_validators
    assert "fmcsa_safer" not in warn_validators


def test_review_ready_to_finalize_true_when_no_errors_or_conflicts():
    """No errors, no conflicts → ready_to_finalize=True."""
    result = _result("cross_field", findings=[
        _finding("warning", "minor issue"),
        _finding("info", "some info"),
    ])
    payload = build_review_payload("sid-3", _empty_sub(), [result])
    assert payload.ready_to_finalize is True


def test_review_ready_to_finalize_false_when_errors_present():
    """Error finding → ready_to_finalize=False."""
    result = _result("cross_field", findings=[
        _finding("error", "DOT carrier must declare vehicles"),
    ])
    payload = build_review_payload("sid-4", _empty_sub(), [result])
    assert payload.ready_to_finalize is False
    assert payload.summary.errors >= 1


def test_review_warnings_and_info_correctly_separated():
    """Warnings land in payload.warnings; info lands in payload.info."""
    result = _result("nhtsa_vpic", findings=[
        _finding("warning", "Year mismatch"),
        _finding("info", "VIN decoded successfully"),
    ])
    payload = build_review_payload("sid-5", _empty_sub(), [result])
    assert any(f.severity == "warning" for f in payload.warnings)
    assert any(f.severity == "info" for f in payload.info)


def test_review_summary_counts_match_reality():
    """Summary counts accurately reflect bucketed findings."""
    result = _result("cross_field", findings=[
        _finding("error", "E1"),
        _finding("warning", "W1"),
        _finding("warning", "W2"),
        _finding("info", "I1"),
    ])
    payload = build_review_payload("sid-6", _empty_sub(), [result])
    assert payload.summary.errors == 1
    assert payload.summary.warnings == 3  # errors surface in warnings list too (1 error + 2 warnings)
    assert payload.summary.info == 1


def test_review_empty_submission_returns_valid_shape():
    """Empty submission + no results → valid ReviewPayload with all-empty lists."""
    payload = build_review_payload("sid-7", _empty_sub(), [])
    assert isinstance(payload, ReviewPayload)
    assert payload.ready_to_finalize is True
    assert payload.conflicts == []
    assert payload.prefills == []
    assert payload.compliance == []
    assert payload.warnings == []
    assert payload.info == []


def test_review_prefills_list_empty_by_default():
    """prefills is always [] — Phase 3 seat."""
    result = _result("census_naics", findings=[_finding("info", "NAICS filled")])
    payload = build_review_payload("sid-8", _empty_sub(), [result])
    assert payload.prefills == []


def test_resolved_conflict_does_not_block_ready_to_finalize():
    """A resolved FieldConflict is excluded from conflicts_pending — ready_to_finalize=True."""
    sub = CustomerSubmission(
        conflicts=[
            FieldConflict(
                field_path="lob_details.vehicles[0].year",
                user_value=2023,
                enriched_value=2024,
                source="nhtsa_vpic",
                resolved=True,
            )
        ]
    )
    payload = build_review_payload("sid-9", sub, [])
    assert payload.ready_to_finalize is True
    assert payload.summary.conflicts_pending == 0
    assert payload.conflicts == []


def test_mixed_resolved_unresolved_conflicts():
    """Only unresolved conflicts appear in payload.conflicts and count toward conflicts_pending."""
    sub = CustomerSubmission(
        conflicts=[
            FieldConflict(
                field_path="vehicles[0].year",
                user_value=2020,
                enriched_value=2021,
                source="nhtsa_vpic",
                resolved=True,
            ),
            FieldConflict(
                field_path="vehicles[1].vin",
                user_value="BAD1",
                enriched_value="GOOD1",
                source="nhtsa_vpic",
                resolved=False,
            ),
        ]
    )
    payload = build_review_payload("sid-10", sub, [])
    assert payload.ready_to_finalize is False
    assert payload.summary.conflicts_pending == 1
    assert len(payload.conflicts) == 1
    assert payload.conflicts[0].field_path == "vehicles[1].vin"
