"""Tests for the PDF filler (P10.A.2)."""
from __future__ import annotations

import pytest

pytest.importorskip("fitz")   # PyMuPDF — skip suite if not installed

from accord_ai.forms import (
    FillResult,
    UnknownFormError,
    fill_form,
    load_form_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pdf_field_values(pdf_bytes: bytes) -> dict:
    """Re-open a filled PDF and return {field_name: field_value} for every widget."""
    import fitz
    doc = fitz.open(stream=bytearray(pdf_bytes), filetype="pdf")
    out = {}
    try:
        for page in doc:
            for w in page.widgets():
                out[w.field_name] = w.field_value
    finally:
        doc.close()
    return out


def _first_text_field(form_number: str) -> str:
    """Return the first WRITABLE text field name.

    We cannot just iterate spec.fields — the first entry in several ACORD
    templates is a read-only label (e.g. 'Form_EditionIdentifier_A' =
    'ACORD 0125 2016-03r1 Acroform'), which the filler correctly skips.
    We open the PDF and look for the first text widget without the
    read-only flag so tests target a fillable target.
    """
    import fitz
    from accord_ai.forms.registry import template_path
    spec = load_form_spec(form_number)
    doc = fitz.open(str(template_path(form_number)))
    try:
        for page in doc:
            for w in page.widgets():
                if w.field_type != fitz.PDF_WIDGET_TYPE_TEXT:
                    continue
                if w.field_flags & 1:   # read-only
                    continue
                if w.field_name in spec.fields:
                    return w.field_name
    finally:
        doc.close()
    raise RuntimeError(f"no writable text field found for form {form_number}")


def _first_checkbox_field(form_number: str) -> str:
    import fitz
    from accord_ai.forms.registry import template_path
    spec = load_form_spec(form_number)
    doc = fitz.open(str(template_path(form_number)))
    try:
        for page in doc:
            for w in page.widgets():
                if w.field_type != fitz.PDF_WIDGET_TYPE_CHECKBOX:
                    continue
                if w.field_flags & 1:
                    continue
                if w.field_name in spec.fields:
                    return w.field_name
    finally:
        doc.close()
    raise RuntimeError(f"no writable checkbox field found for form {form_number}")


# ---------------------------------------------------------------------------
# Happy path — fill + read back
# ---------------------------------------------------------------------------

def test_fill_text_field_roundtrip():
    text_field = _first_text_field("125")
    pdf_bytes, res = fill_form("125", {text_field: "Acme Corporation"})
    assert isinstance(res, FillResult)
    assert res.filled_count == 1
    assert res.error_count == 0
    assert res.unknown_fields == ()

    values = _read_pdf_field_values(pdf_bytes)
    assert values[text_field] == "Acme Corporation"


def test_fill_checkbox_truthy_values():
    cb = _first_checkbox_field("125")
    for v in (True, "yes", "x", "1", "on"):
        pdf_bytes, res = fill_form("125", {cb: v})
        assert res.filled_count == 1, f"value={v!r}"
        # PyMuPDF returns "Yes" / "On" / etc. depending on the template's
        # /ON appearance state; just assert it's no longer the "Off" state.
        got = _read_pdf_field_values(pdf_bytes)[cb]
        assert got not in ("Off", "", False, None), f"value={v!r} → {got!r}"


def test_checkbox_falsy_stays_unchecked():
    cb = _first_checkbox_field("125")
    pdf_bytes, res = fill_form("125", {cb: False})
    # Empty fill → cleared, not treated as an unfilled-but-checked state.
    values = _read_pdf_field_values(pdf_bytes)
    assert values[cb] in ("Off", False, "", None)


def test_empty_string_clears_field():
    text_field = _first_text_field("125")
    pdf_bytes, res = fill_form("125", {text_field: ""})
    # Empty strings are skipped from fill and the field is cleared.
    assert res.filled_count == 0
    values = _read_pdf_field_values(pdf_bytes)
    assert values[text_field] in ("", None)


def test_none_value_clears_field():
    text_field = _first_text_field("125")
    pdf_bytes, res = fill_form("125", {text_field: None})
    assert res.filled_count == 0
    values = _read_pdf_field_values(pdf_bytes)
    assert values[text_field] in ("", None)


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------

def test_unknown_field_tallied_not_errored():
    pdf_bytes, res = fill_form("125", {"THIS_FIELD_DOES_NOT_EXIST": "x"})
    assert res.filled_count == 0
    assert res.error_count == 0
    assert "THIS_FIELD_DOES_NOT_EXIST" in res.unknown_fields


def test_mix_of_known_and_unknown_fields():
    text_field = _first_text_field("125")
    pdf_bytes, res = fill_form("125", {
        text_field: "Hello",
        "bogus_field_1": "x",
        "bogus_field_2": "y",
    })
    assert res.filled_count == 1
    assert set(res.unknown_fields) == {"bogus_field_1", "bogus_field_2"}


def test_result_is_frozen():
    _, res = fill_form("125", {})
    with pytest.raises((AttributeError, TypeError)):
        res.filled_count = 999  # type: ignore[misc]


def test_result_to_dict_shape():
    _, res = fill_form("125", {"BAD": "v"})
    d = res.to_dict()
    assert d["form_number"] == "125"
    assert d["filled"] == 0
    assert d["unknown_fields"] == ["BAD"]
    assert isinstance(d["error_messages"], list)


# ---------------------------------------------------------------------------
# Byte-stability (critical for content-hash dedup in P10.C)
# ---------------------------------------------------------------------------

def test_deterministic_output_bytes():
    text_field = _first_text_field("125")
    b1, _ = fill_form("125", {text_field: "Acme"})
    b2, _ = fill_form("125", {text_field: "Acme"})
    assert b1 == b2


def test_different_values_produce_different_bytes():
    text_field = _first_text_field("125")
    b1, _ = fill_form("125", {text_field: "Acme"})
    b2, _ = fill_form("125", {text_field: "Globex"})
    assert b1 != b2


# ---------------------------------------------------------------------------
# /AP null trick — form 163 is the one v3 hand-debugged for this.
# ---------------------------------------------------------------------------

def test_form_163_text_fill_visible_after_save():
    """Regression: before /AP null, form 163 stored /V correctly but kept
    showing template sample values on reopen because of shared/indirect AP
    streams. This test will fail if the /AP null trick regresses."""
    text_field = _first_text_field("163")
    pdf_bytes, res = fill_form("163", {text_field: "PREMIUM_MARKER_TEXT"})
    assert res.filled_count == 1
    values = _read_pdf_field_values(pdf_bytes)
    assert values[text_field] == "PREMIUM_MARKER_TEXT"


# ---------------------------------------------------------------------------
# Unknown form + edge cases
# ---------------------------------------------------------------------------

def test_unknown_form_raises():
    with pytest.raises(UnknownFormError):
        fill_form("999", {})


def test_empty_input_returns_valid_blank_pdf():
    pdf_bytes, res = fill_form("125", {})
    assert res.filled_count == 0
    assert pdf_bytes.startswith(b"%PDF-")
    # Still a parseable PDF with widgets readable.
    values = _read_pdf_field_values(pdf_bytes)
    assert len(values) > 0


def test_large_real_form_fills_without_crash():
    """Smoke-test filling a substantial subset of form 125's writable text fields.

    Uses the PDF itself (not spec order) so we skip read-only labels like
    the edition identifier. filled_count may exceed the input size because
    some field_names map to multiple widgets (multi-page duplicates) —
    we only assert every requested name round-trips correctly.
    """
    import fitz
    from accord_ai.forms.registry import template_path

    spec = load_form_spec("125")
    inputs: dict = {}
    taken = 0
    doc = fitz.open(str(template_path("125")))
    try:
        for page in doc:
            if taken >= 25:
                break
            for w in page.widgets():
                if taken >= 25:
                    break
                if w.field_type != fitz.PDF_WIDGET_TYPE_TEXT:
                    continue
                if w.field_flags & 1:
                    continue
                fname = w.field_name
                if fname in spec.fields and fname not in inputs:
                    inputs[fname] = f"val_{taken}"
                    taken += 1
    finally:
        doc.close()

    pdf_bytes, res = fill_form("125", inputs)
    # Duplicate-widget fields bump filled_count above input size — accept ≥ 25.
    assert res.filled_count >= 25
    assert res.error_count == 0
    values = _read_pdf_field_values(pdf_bytes)
    for name in inputs:
        assert values[name] == inputs[name]
