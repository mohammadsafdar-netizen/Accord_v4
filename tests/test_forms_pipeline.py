"""Tests for fill_submission (P10.A.4 pipeline half)."""
from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("fitz")

from accord_ai.forms import FilledForm, fill_submission
from accord_ai.schema import (
    Address,
    CommercialAutoDetails,
    CustomerSubmission,
    Driver,
    GeneralLiabilityCoverage,
    GeneralLiabilityDetails,
    PolicyDates,
    Vehicle,
)


def _ca_submission() -> CustomerSubmission:
    return CustomerSubmission(
        business_name="Acme Trucking",
        ein="12-3456789",
        mailing_address=Address(
            line_one="123 Main", city="Austin", state="TX", zip_code="78701",
        ),
        policy_dates=PolicyDates(
            effective_date=date(2026, 5, 1),
            expiration_date=date(2027, 5, 1),
        ),
        lob_details=CommercialAutoDetails(
            drivers=[
                Driver(
                    first_name="Alice", last_name="Jones", license_state="TX",
                ),
            ],
            vehicles=[
                Vehicle(
                    year=2024, make="Freightliner", model="Cascadia",
                    vin="1FUJGLDR0LLAB1234",
                ),
            ],
        ),
    )


# --- Happy path --------------------------------------------------------------

def test_fill_submission_ca_produces_all_lob_forms():
    out = fill_submission(_ca_submission())
    assert set(out.keys()) == {"125", "127", "129", "137", "163"}
    for ff in out.values():
        assert isinstance(ff, FilledForm)
        assert ff.pdf_bytes.startswith(b"%PDF-")
        assert len(ff.content_hash) == 64          # sha256 hex


def test_fill_submission_gl_produces_125_and_126():
    s = CustomerSubmission(
        business_name="GlobeX",
        lob_details=GeneralLiabilityDetails(
            coverage=GeneralLiabilityCoverage(each_occurrence_limit=1_000_000),
        ),
    )
    out = fill_submission(s)
    assert set(out.keys()) == {"125", "126"}


def test_fill_submission_no_lob_returns_empty():
    out = fill_submission(CustomerSubmission(business_name="Acme"))
    assert out == {}


# --- Content hash ------------------------------------------------------------

def test_fill_submission_content_hash_is_deterministic():
    """Byte-stable filler → byte-stable hash. Load-bearing for dedup."""
    a = fill_submission(_ca_submission())
    b = fill_submission(_ca_submission())
    for form in a:
        assert a[form].content_hash == b[form].content_hash
        assert a[form].pdf_bytes == b[form].pdf_bytes


def test_fill_submission_content_hash_changes_with_data():
    s1 = _ca_submission()
    s2 = _ca_submission().model_copy(update={"business_name": "Different Inc"})
    a = fill_submission(s1)
    b = fill_submission(s2)
    # 125 holds business_name — its hash must differ
    assert a["125"].content_hash != b["125"].content_hash
    # 163 is an unmapped form at this stage — its output is invariant to the
    # business_name change, so its hash should match.
    assert a["163"].content_hash == b["163"].content_hash


# --- FilledForm surface ------------------------------------------------------

def test_filled_form_is_frozen():
    out = fill_submission(_ca_submission())
    ff = next(iter(out.values()))
    with pytest.raises((AttributeError, TypeError)):
        ff.form_number = "X"  # type: ignore[misc]


def test_filled_form_to_dict_excludes_bytes():
    out = fill_submission(_ca_submission())
    d = out["125"].to_dict()
    assert "pdf_bytes" not in d
    assert d["form_number"] == "125"
    assert d["byte_length"] > 0
    assert len(d["content_hash"]) == 64
    assert d["fill_result"]["form_number"] == "125"
