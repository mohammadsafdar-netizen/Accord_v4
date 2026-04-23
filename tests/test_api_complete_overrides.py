"""Tests for FE-override merging in /complete (P10.0.f.3).

The v3 FE editor lets brokers correct extraction errors before finalizing;
those corrections arrive as `CompleteRequest.fields_data`. This module
covers:
  1. fill_submission's new `field_overrides` kwarg — unit-level merge
     semantics (precedence, empty-string clear, unknown widgets, key
     normalization, malformed keys).
  2. The /complete handler wiring — overrides reach the PDF bytes, the
     audit event fires only when fields_data is non-empty, and the
     unknown_fields surface bubbles into the response.

Filled PDFs are inspected by re-parsing the bytes with PyMuPDF and
walking widgets — the same ground-truth path used by
test_forms_widget_ground_truth.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fitz")
import fitz  # noqa: E402  (import order gated by the importorskip above)

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.forms import fill_submission
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    PolicyDates,
    Vehicle,
)
from accord_ai.testing import FakeEngine


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "accord.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as c:
        yield c, intake


def _seed_ca_session(intake, tenant: str = "acme") -> str:
    sid = intake.store.create_session(tenant=tenant)
    submission = CustomerSubmission(
        business_name="Acme Trucking",
        mailing_address=Address(
            line_one="123 Main", city="Austin", state="TX", zip_code="78701",
        ),
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Alice", last_name="Jones")],
            vehicles=[
                Vehicle(year=2024, make="Freightliner", model="Cascadia"),
            ],
        ),
    )
    intake.store.update_submission(sid, submission, tenant=tenant)
    return sid


def _load_pdf_widget(pdf_bytes: bytes, widget_name: str) -> Optional[str]:
    """Read a widget's /V value out of the filled PDF. None if absent."""
    doc = fitz.open(stream=bytearray(pdf_bytes), filetype="pdf")
    try:
        for page in doc:
            for w in page.widgets():
                if w.field_name == widget_name:
                    return w.field_value
    finally:
        doc.close()
    return None


def _pdf_bytes_for_form(intake, sid: str, form_number: str) -> bytes:
    """Read the saved filled PDF bytes for a form."""
    pdf = intake.filled_pdf_store.load(sid, "acme", form_number)
    assert pdf is not None, f"no filled pdf for form {form_number}"
    return pdf


# ---------------------------------------------------------------------------
# /complete behavior — overrides on vs off
# ---------------------------------------------------------------------------

def test_complete_without_overrides_uses_mapper_only(client):
    """Empty / omitted fields_data leaves mapper output untouched."""
    c, intake = client
    sid = _seed_ca_session(intake)

    # Omitted fields_data
    resp1 = c.post(
        "/complete",
        json={"submission_id": sid, "tenant_slug": "acme"},
    )
    assert resp1.status_code == 200
    pdf1 = _pdf_bytes_for_form(intake, sid, "125")

    # Explicit empty dict
    resp2 = c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {},
        },
    )
    assert resp2.status_code == 200
    pdf2 = _pdf_bytes_for_form(intake, sid, "125")

    # Byte-stable: omitted vs {} is the same fill — no override path taken.
    assert pdf1 == pdf2
    assert _load_pdf_widget(pdf1, "NamedInsured_FullName_A") == "Acme Trucking"


def test_complete_override_replaces_mapper_value(client):
    c, intake = client
    sid = _seed_ca_session(intake)

    resp = c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {
                    "NamedInsured_FullName_A": "Corrected Co",
                },
            },
        },
    )
    assert resp.status_code == 200
    pdf = _pdf_bytes_for_form(intake, sid, "125")
    assert _load_pdf_widget(pdf, "NamedInsured_FullName_A") == "Corrected Co"


def test_complete_override_with_form_prefix_normalized(client):
    """'form_125' and '125' must be indistinguishable to the filler."""
    c, intake = client

    # Run A — "form_125" prefix
    sid_a = _seed_ca_session(intake)
    resp_a = c.post(
        "/complete",
        json={
            "submission_id": sid_a,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {"NamedInsured_FullName_A": "Override Co"},
            },
        },
    )
    assert resp_a.status_code == 200
    pdf_a = _pdf_bytes_for_form(intake, sid_a, "125")

    # Run B — bare "125"
    sid_b = _seed_ca_session(intake)
    resp_b = c.post(
        "/complete",
        json={
            "submission_id": sid_b,
            "tenant_slug": "acme",
            "fields_data": {
                "125": {"NamedInsured_FullName_A": "Override Co"},
            },
        },
    )
    assert resp_b.status_code == 200
    pdf_b = _pdf_bytes_for_form(intake, sid_b, "125")

    # Byte-stable filler → identical hashes → identical bytes. If the
    # normalization regressed, one run would carry the mapper's
    # "Acme Trucking" and the other wouldn't.
    form_a = next(f for f in resp_a.json()["forms"] if f["form_number"] == "125")
    form_b = next(f for f in resp_b.json()["forms"] if f["form_number"] == "125")
    assert form_a["content_hash"] == form_b["content_hash"]
    assert pdf_a == pdf_b
    assert _load_pdf_widget(pdf_a, "NamedInsured_FullName_A") == "Override Co"


def test_complete_override_empty_string_filtered_v3_compat(client):
    """v3 wire-compat (P10.0.g.5): empty-string override values are
    FILTERED by translate_payload's null-sentinel treatment, NOT used
    to clear the widget. v3 FE sends "NullObject" or omits the key;
    empty string is treated the same way. The mapper's value remains.

    This intentionally diverges from v4's original semantic (empty
    clears); "exactly same as v3" is the current cutover goal.
    """
    c, intake = client
    sid = _seed_ca_session(intake)

    resp = c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {"NamedInsured_FullName_A": ""},
            },
        },
    )
    assert resp.status_code == 200
    pdf = _pdf_bytes_for_form(intake, sid, "125")
    # Empty override was filtered; the mapper's value stands (from
    # _seed_ca_session, which sets business_name="Acme Trucking").
    assert _load_pdf_widget(pdf, "NamedInsured_FullName_A") == "Acme Trucking"
    form_info = next(
        f for f in resp.json()["forms"] if f["form_number"] == "125"
    )
    # Empty-string values are filtered BEFORE reaching fill_submission,
    # so they never become unknown_fields candidates either.
    assert "NamedInsured_FullName_A" not in form_info["fill_result"][
        "unknown_fields"
    ]


def test_complete_override_unknown_widget_tallied(client):
    """Typoed override widget name → surfaced in FillResult.unknown_fields."""
    c, intake = client
    sid = _seed_ca_session(intake)

    resp = c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {
                "125": {"Not_A_Real_Field_A": "x"},
            },
        },
    )
    assert resp.status_code == 200
    form_125 = next(
        f for f in resp.json()["forms"] if f["form_number"] == "125"
    )
    assert "Not_A_Real_Field_A" in form_125["fill_result"]["unknown_fields"]


def test_complete_override_key_normalization_rejects_bad_keys(
    client, caplog,
):
    """Malformed form keys are skipped with a WARNING — not a crash.

    accord_ai's logger has propagate=False, so pytest's default caplog
    handler (attached to root) never sees child-logger records. Attach
    caplog.handler directly to 'accord_ai.forms.pipeline' — same pattern
    as test_forms_mapper_canonical.
    """
    c, intake = client
    sid = _seed_ca_session(intake)

    pipeline_logger = logging.getLogger("accord_ai.forms.pipeline")
    pipeline_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="accord_ai.forms.pipeline")
    try:
        resp = c.post(
            "/complete",
            json={
                "submission_id": sid,
                "tenant_slug": "acme",
                "fields_data": {
                    "foo": {"NamedInsured_FullName_A": "Ignored"},
                    "form_xx": {"NamedInsured_FullName_A": "AlsoIgnored"},
                    # Valid one alongside — prove the merge continues.
                    "form_125": {"NamedInsured_FullName_A": "Applied"},
                },
            },
        )
    finally:
        pipeline_logger.removeHandler(caplog.handler)

    assert resp.status_code == 200
    messages = " ".join(r.message for r in caplog.records)
    assert "malformed override key" in messages
    assert "'foo'" in messages
    assert "'form_xx'" in messages

    pdf = _pdf_bytes_for_form(intake, sid, "125")
    # Valid key survived the merge.
    assert _load_pdf_widget(pdf, "NamedInsured_FullName_A") == "Applied"


def test_complete_audit_emitted_only_when_overrides_provided(client):
    c, intake = client

    # Run 1 — no fields_data → no audit event.
    sid1 = _seed_ca_session(intake)
    c.post(
        "/complete",
        json={"submission_id": sid1, "tenant_slug": "acme"},
    )
    events = intake.store.list_audit_events(
        event_type="complete.overrides_applied",
    )
    assert events == []

    # Run 2 — non-empty fields_data → one audit event with shape payload.
    sid2 = _seed_ca_session(intake)
    c.post(
        "/complete",
        json={
            "submission_id": sid2,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {
                    "NamedInsured_FullName_A": "Override",
                    "NamedInsured_TaxIdentifier_A": "",   # empty-clear
                },
                "127": {"Driver_GivenName_A": "OverrideDriver"},
            },
        },
    )
    events = intake.store.list_audit_events(
        event_type="complete.overrides_applied",
    )
    assert len(events) == 1
    e = events[0]
    assert e.tenant == "acme"
    assert e.session_id == sid2
    assert set(e.payload["form_numbers"]) == {"form_125", "127"}
    # v3 wire-compat (P10.0.g.5): empty-string values are filtered during
    # translate_payload, so the form_125 count is 1 (not 2) post-filter.
    assert e.payload["override_counts"] == {"form_125": 1, "127": 1}
    # empty_overrides counter is now always 0 — v3 filters them before
    # they ever reach fill_submission.
    assert e.payload["empty_overrides"] == 0


def test_complete_override_precedence_over_mapper(client):
    """Mapper would output "Acme Trucking"; override must replace it."""
    c, intake = client
    sid = _seed_ca_session(intake)

    # Baseline: mapper path.
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})
    pdf_mapper = _pdf_bytes_for_form(intake, sid, "125")
    assert (
        _load_pdf_widget(pdf_mapper, "NamedInsured_FullName_A")
        == "Acme Trucking"
    )

    # Override path: same submission, override wins.
    c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {"NamedInsured_FullName_A": "BrokerEdited Co"},
            },
        },
    )
    pdf_override = _pdf_bytes_for_form(intake, sid, "125")
    assert (
        _load_pdf_widget(pdf_override, "NamedInsured_FullName_A")
        == "BrokerEdited Co"
    )
    # Bytes differ — override made it onto the PDF.
    assert pdf_mapper != pdf_override


def test_complete_override_partial_form_coverage(client):
    """Override covers 1 widget; other mapper-derived values survive."""
    c, intake = client
    sid = _seed_ca_session(intake)

    c.post(
        "/complete",
        json={
            "submission_id": sid,
            "tenant_slug": "acme",
            "fields_data": {
                "form_125": {
                    "NamedInsured_FullName_A": "Only This Changed",
                },
            },
        },
    )
    pdf = _pdf_bytes_for_form(intake, sid, "125")
    # Overridden widget.
    assert (
        _load_pdf_widget(pdf, "NamedInsured_FullName_A")
        == "Only This Changed"
    )
    # Untouched by the override — still carries the mapper's value.
    assert (
        _load_pdf_widget(pdf, "NamedInsured_MailingAddress_CityName_A")
        == "Austin"
    )
    assert (
        _load_pdf_widget(pdf, "NamedInsured_MailingAddress_StateOrProvinceCode_A")
        == "TX"
    )


# ---------------------------------------------------------------------------
# Pipeline-level unit coverage
# ---------------------------------------------------------------------------

def test_fill_submission_direct_field_overrides_param():
    """Call fill_submission directly — no API — and prove the merge works."""
    submission = CustomerSubmission(
        business_name="Original Co",
        mailing_address=Address(
            line_one="1 Street", city="Dallas", state="TX", zip_code="75201",
        ),
        policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
        lob_details=CommercialAutoDetails(
            drivers=[Driver(first_name="Bob", last_name="Lee")],
            vehicles=[Vehicle(year=2023, make="Ford", model="F-150")],
        ),
    )

    # No overrides.
    out_plain = fill_submission(submission)
    assert (
        _load_pdf_widget(out_plain["125"].pdf_bytes, "NamedInsured_FullName_A")
        == "Original Co"
    )

    # With overrides — accepts both "form_125" and "125" shape.
    overrides: Dict[str, Dict[str, object]] = {
        "form_125": {"NamedInsured_FullName_A": "Direct Override"},
    }
    out_over = fill_submission(submission, field_overrides=overrides)
    assert (
        _load_pdf_widget(out_over["125"].pdf_bytes, "NamedInsured_FullName_A")
        == "Direct Override"
    )

    # Empty-string override clears the widget.
    out_cleared = fill_submission(
        submission,
        field_overrides={"125": {"NamedInsured_FullName_A": ""}},
    )
    assert (
        _load_pdf_widget(
            out_cleared["125"].pdf_bytes, "NamedInsured_FullName_A",
        )
        == ""
    )


# ---------------------------------------------------------------------------
# P10.0.f.4 / M3 — backfill missing override-path coverage
# ---------------------------------------------------------------------------

def test_complete_override_for_form_not_in_lob_is_dropped_with_warning(
    client, caplog,
):
    """Override key for a form the LOB doesn't emit must be silently
    non-fatal — but surface a WARNING so ops can notice stale FE state.

    Commercial auto emits 125/127/129/137/163; sending an override for
    form 126 (GL) against a CA session must not 500 the request, must
    not fill 126, and must produce a log line naming the unapplied form.
    """
    import logging as _logging
    c, intake = client
    sid = _seed_ca_session(intake)

    pipeline_logger = _logging.getLogger("accord_ai.forms.pipeline")
    pipeline_logger.addHandler(caplog.handler)
    original_level = pipeline_logger.level
    pipeline_logger.setLevel(_logging.DEBUG)
    try:
        resp = c.post("/complete", json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "fields_data": {
                "125": {"NamedInsured_FullName_A": "Overridden"},
                "126": {"NamedInsured_FullName_A": "Ghost"},   # not in LOB
            },
        })
    finally:
        pipeline_logger.removeHandler(caplog.handler)
        pipeline_logger.setLevel(original_level)

    assert resp.status_code == 200
    forms_emitted = {f["form_number"] for f in resp.json()["forms"]}
    assert "126" not in forms_emitted
    # Valid override on 125 still landed.
    assert "125" in forms_emitted

    warns = [
        r for r in caplog.records
        if "unapplied" in r.getMessage().lower()
        or ("not in this submission" in r.getMessage().lower())
    ]
    assert warns, "expected unapplied-override warning"
    assert "126" in warns[0].getMessage()


def test_complete_override_with_no_lob_submission_still_200(client):
    """Submission with no LOB emits zero forms. An override payload against
    such a session must not crash; it's a no-op (nothing to override)."""
    c, intake = client
    sid = intake.store.create_session(tenant="acme")
    # No LOB details → map_submission returns {} → no forms to fill
    from accord_ai.schema import CustomerSubmission
    intake.store.update_submission(
        sid, CustomerSubmission(business_name="No LOB Yet"), tenant="acme",
    )
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {
            "125": {"NamedInsured_FullName_A": "Goes Nowhere"},
        },
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["forms"] == []
    assert body["total_written"] == 0


def test_complete_override_oversized_value_returns_422(client):
    """Widget value > 100_000 chars fails at pydantic validation (422).
    Cap raised from 10K to 100K for v3 wire-compat (P10.0.g.7); real FE
    payloads never approach this boundary, but pathological multi-MB
    values are still rejected."""
    c, intake = client
    sid = _seed_ca_session(intake)
    giant = "x" * 100_001
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data": {
            "125": {"NamedInsured_FullName_A": giant},
        },
    })
    assert resp.status_code == 422
    body = resp.json()
    assert "max value length" in str(body).lower()


def test_complete_override_too_many_forms_returns_422(client):
    """Cap raised from 16 to 32 forms (v3 wire-compat, P10.0.g.7).
    A real FE sends up to 10 forms; 33+ is clearly malicious."""
    c, intake = client
    sid = _seed_ca_session(intake)
    fields_data = {
        f"form_{i:03d}": {"W": "v"} for i in range(33)   # > cap of 32
    }
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   fields_data,
    })
    assert resp.status_code == 422
    assert "max forms" in str(resp.json()).lower()


def test_complete_override_too_many_widgets_per_form_returns_422(client):
    """Cap raised from 500 to 2000 widgets per form (v3 wire-compat,
    P10.0.g.7). ACORD 160 has ~1135 widgets; 2001+ exceeds every
    real ACORD form."""
    c, intake = client
    sid = _seed_ca_session(intake)
    widgets = {f"Widget_{i}": "v" for i in range(2001)}   # > cap of 2000
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {"125": widgets},
    })
    assert resp.status_code == 422
    assert "max widgets" in str(resp.json()).lower()


def test_complete_override_nested_value_accepted_v3_compat(client):
    """v3 wire-compat (P10.0.g.7): fields_data inner values are Any —
    nested dicts/lists round-trip without 422. This enables v3's
    structured shapes for form 163 (`_header`, `drivers` sub-dicts).

    Note: nested values passed to a form OTHER than 163 will just end
    up in the unknown_fields tally during fill (expected — fill_form
    doesn't know how to route nested shapes for other forms).
    """
    c, intake = client
    sid = _seed_ca_session(intake)
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data": {
            "163": {
                "_header": {"policy_number": "POL-123"},
                "drivers": [{"first_name": "Alice"}, {"first_name": "Bob"}],
            },
        },
    })
    assert resp.status_code == 200, resp.json()


def test_complete_override_scalar_types_accepted(client):
    """str / int / float / bool / None must all validate — caller edits
    can come as typed JSON from the FE."""
    c, intake = client
    sid = _seed_ca_session(intake)
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data": {
            "125": {
                "NamedInsured_FullName_A":     "Acme",          # str
                "Producer_CustomerIdentifier_A": 12345,          # int
            },
        },
    })
    assert resp.status_code == 200
