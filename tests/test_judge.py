"""SchemaJudge tests — v3-aligned, declarative (P10.0.g.8).

The per-LOB critical-field lists are the source of truth. Tests
parametrize across each LOB's full list so adding a critical field in
``critical_fields.py`` automatically generates a test case — no
hand-written per-rule test needed.
"""
from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from accord_ai.harness.critical_fields import (
    _CA_CRITICAL,
    _COMMON_CRITICAL,
    _GL_CRITICAL,
    _WC_CRITICAL,
    get_critical_fields,
)
from accord_ai.harness.judge import (
    JudgeVerdict,
    SchemaJudge,
    _is_empty,
    _resolve,
)
from accord_ai.schema import (
    Address,
    Contact,
    CustomerSubmission,
    PolicyDates,
)
from tests._fixtures import valid_ca, valid_gl, valid_wc


def _judge(sub: CustomerSubmission) -> JudgeVerdict:
    return SchemaJudge().evaluate(sub)


# ---------------------------------------------------------------------------
# JudgeVerdict shape
# ---------------------------------------------------------------------------

def test_verdict_is_frozen():
    v = JudgeVerdict(passed=True)
    with pytest.raises(FrozenInstanceError):
        v.passed = False


def test_verdict_defaults_are_empty_tuples():
    v = JudgeVerdict(passed=True)
    assert v.reasons == ()
    assert v.failed_paths == ()


# ---------------------------------------------------------------------------
# _is_empty + _resolve unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,expected", [
    (None, True),
    ("",   True),
    ("   ", True),
    ([],   True),
    ({},   True),
    ("x",   False),
    ([1],   False),
    ({"k": 1}, False),
    (0,     False),   # zero is a real value, not empty
    (False, False),   # false is a real value, not empty
])
def test_is_empty(v, expected):
    assert _is_empty(v) is expected


def test_resolve_nested_attribute():
    sub = CustomerSubmission(
        mailing_address=Address(city="Austin"),
    )
    assert _resolve(sub, "mailing_address.city") == "Austin"


def test_resolve_list_index():
    sub = CustomerSubmission(contacts=[Contact(full_name="Jane")])
    assert _resolve(sub, "contacts[0].full_name") == "Jane"


def test_resolve_missing_intermediate_returns_none():
    assert _resolve(CustomerSubmission(), "mailing_address.city") is None


def test_resolve_out_of_range_index_returns_none():
    sub = CustomerSubmission(contacts=[Contact(full_name="Jane")])
    assert _resolve(sub, "contacts[5].full_name") is None


def test_resolve_malformed_segment_returns_none():
    sub = CustomerSubmission(business_name="Acme")
    assert _resolve(sub, "business_name[bogus]") is None


# ---------------------------------------------------------------------------
# Baseline — fully-valid submissions pass every LOB
# ---------------------------------------------------------------------------

def test_valid_ca_passes():
    v = _judge(valid_ca())
    assert v.passed, f"unexpected CA failures: {v.reasons}"


def test_valid_gl_passes():
    v = _judge(valid_gl())
    assert v.passed, f"unexpected GL failures: {v.reasons}"


def test_valid_wc_passes():
    v = _judge(valid_wc())
    assert v.passed, f"unexpected WC failures: {v.reasons}"


# ---------------------------------------------------------------------------
# Empty submission fails every LOB-agnostic path + lob_details gate
# ---------------------------------------------------------------------------

def test_empty_submission_fails_all_common_paths():
    v = _judge(CustomerSubmission())
    assert not v.passed
    for path, _ in _COMMON_CRITICAL:
        assert path in v.failed_paths, (
            f"expected COMMON path {path!r} in failed_paths "
            f"(got {v.failed_paths})"
        )
    assert "lob_details" in v.failed_paths


# ---------------------------------------------------------------------------
# Per-LOB critical-field parametrized tests
# ---------------------------------------------------------------------------
# For every critical field, zeroing it must cause the judge to emit
# that exact path in failed_paths. Adding a new critical to
# critical_fields.py automatically generates its test case.

@pytest.mark.parametrize(
    "path,reason", _CA_CRITICAL, ids=[p for p, _ in _CA_CRITICAL],
)
def test_ca_critical_path_required(path, reason):
    sub = _zero_out_path(valid_ca(), path)
    v = _judge(sub)
    assert not v.passed, f"judge should fail when {path} is missing"
    assert path in v.failed_paths, (
        f"path {path!r} missing from failed_paths={v.failed_paths}"
    )


@pytest.mark.parametrize(
    "path,reason", _GL_CRITICAL, ids=[p for p, _ in _GL_CRITICAL],
)
def test_gl_critical_path_required(path, reason):
    sub = _zero_out_path(valid_gl(), path)
    v = _judge(sub)
    assert not v.passed, f"judge should fail when {path} is missing"
    assert path in v.failed_paths, (
        f"path {path!r} missing from failed_paths={v.failed_paths}"
    )


@pytest.mark.parametrize(
    "path,reason", _WC_CRITICAL, ids=[p for p, _ in _WC_CRITICAL],
)
def test_wc_critical_path_required(path, reason):
    sub = _zero_out_path(valid_wc(), path)
    v = _judge(sub)
    assert not v.passed, f"judge should fail when {path} is missing"
    assert path in v.failed_paths, (
        f"path {path!r} missing from failed_paths={v.failed_paths}"
    )


# ---------------------------------------------------------------------------
# Cross-field invariant: policy date ordering
# ---------------------------------------------------------------------------

def test_date_ordering_effective_after_expiration_fails():
    sub = valid_ca().model_copy(update={
        "policy_dates": PolicyDates(
            effective_date=date(2027, 5, 1),
            expiration_date=date(2026, 5, 1),
        ),
    })
    v = _judge(sub)
    assert not v.passed
    assert "policy_dates.effective_date" in v.failed_paths
    assert "policy_dates.expiration_date" in v.failed_paths
    assert any("after" in r for r in v.reasons)


def test_date_ordering_equal_is_ok():
    sub = valid_ca().model_copy(update={
        "policy_dates": PolicyDates(
            effective_date=date(2026, 5, 1),
            expiration_date=date(2026, 5, 1),
        ),
    })
    assert _judge(sub).passed


def test_date_ordering_only_fires_when_both_set():
    """Effective-only (no expiration) passes the ordering rule — the
    effective-date-required rule has already gated presence."""
    sub = valid_ca().model_copy(update={
        "policy_dates": PolicyDates(
            effective_date=date(2026, 5, 1),
            expiration_date=None,
        ),
    })
    v = _judge(sub)
    assert v.passed, f"unexpected failures: {v.reasons}"


# ---------------------------------------------------------------------------
# get_critical_fields dispatch
# ---------------------------------------------------------------------------

def test_get_critical_fields_dispatches_commercial_auto():
    fields = get_critical_fields("commercial_auto")
    paths = {p for p, _ in fields}
    # v3 CA plugin: 10 common + 13 CA-specific = 23.
    assert len(fields) == 23
    assert "lob_details.fleet_use_type" in paths
    assert "business_name" in paths


def test_get_critical_fields_dispatches_general_liability():
    fields = get_critical_fields("general_liability")
    paths = {p for p, _ in fields}
    assert "operations_description" in paths
    assert "annual_revenue" in paths


def test_get_critical_fields_dispatches_workers_comp():
    fields = get_critical_fields("workers_comp")
    paths = {p for p, _ in fields}
    assert "lob_details.payroll_by_class" in paths
    assert (
        "lob_details.coverage.employers_liability_per_accident"
        in paths
    )


def test_unknown_lob_falls_back_to_common():
    fields = get_critical_fields("bogus_lob")
    paths = {p for p, _ in fields}
    assert "business_name" in paths
    assert "mailing_address.city" in paths
    # No LOB-specific path leaks through.
    assert "lob_details.payroll_by_class" not in paths
    assert "lob_details.fleet_use_type" not in paths


def test_get_critical_fields_returns_list_copy():
    """Caller must not be able to mutate the shared critical-field list."""
    a = get_critical_fields("commercial_auto")
    b = get_critical_fields("commercial_auto")
    a.clear()
    assert len(b) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEX_RE = re.compile(r"^(.+)\[(\d+)\]$")


def _zero_out_path(
    sub: CustomerSubmission, path: str,
) -> CustomerSubmission:
    """Return a copy of ``sub`` with the value at ``path`` set to empty.

    Scalar leaves become None. List-typed leaves become []. Used by the
    parametrized per-path tests so the baseline submission stays
    fully-valid while each case perturbs exactly one field.
    """
    dumped = sub.model_dump()
    _walk(dumped, path.split("."))
    return CustomerSubmission.model_validate(dumped)


def _walk(container, segments):
    """Descend `segments` into `container` (a dict) and zero the leaf.

    List indexing is handled via the [N] suffix on a segment; pydantic
    models have already been dumped to dicts so every intermediate is
    a dict or a list of dicts.
    """
    seg = segments[0]
    is_leaf = len(segments) == 1
    m = _INDEX_RE.match(seg)

    if m:
        root, idx = m.group(1), int(m.group(2))
        target_list = container.get(root)
        if not isinstance(target_list, list) or len(target_list) <= idx:
            return  # already absent; nothing to zero
        if is_leaf:
            # Leaf list-element path: empty the whole list so the
            # out-of-range index becomes the trigger.
            container[root] = []
            return
        _walk(target_list[idx], segments[1:])
        return

    if is_leaf:
        current = container.get(seg)
        # Preserve container shape: list → [], everything else → None.
        container[seg] = [] if isinstance(current, list) else None
        return

    child = container.get(seg)
    if isinstance(child, dict):
        _walk(child, segments[1:])
    # Intermediate missing / non-dict — already absent, nothing to zero.
