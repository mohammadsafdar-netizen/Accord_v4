"""Ground-truth mapping audit (P10.S.6b).

Reads the widget-level ground truth extracted by scripts/extract_pdf_widgets.py
from the actual blank PDF bytes and validates every _FORM_ALIASES entry against
it. This catches three classes of bug that the existing registry-spec-based
invariant can't:

  1. Widget name typos that happen to also exist in v3's legacy JSON (both
     stale) — PDF is the only source of truth.
  2. Mappings to read-only widgets — PyMuPDF silently skips those at fill
     time, so the fill succeeds but the PDF is blank.
  3. Type conflicts — a checkbox widget mapped as a text field, or vice
     versa. Widget.update() accepts both but the output is garbled.

The ground-truth JSONs live at accord_ai/form_fields_enriched/ and must be
regenerated whenever a blank PDF changes. See scripts/extract_pdf_widgets.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from accord_ai.forms.mapper import _FORM_ALIASES


GROUND_TRUTH_DIR = Path(__file__).parent.parent / "accord_ai" / "form_fields_enriched"


def _load_ground_truth() -> Dict[str, Dict[str, dict]]:
    """Return {form_number: {widget_name: widget_dict}}. One widget_dict per
    unique name (duplicates on multi-page forms collapse — their type and
    readonly flags should match anyway)."""
    out: Dict[str, Dict[str, dict]] = {}
    for p in sorted(GROUND_TRUTH_DIR.glob("form_*_widgets.json")):
        data = json.loads(p.read_text())
        form = data["form"]
        widgets: Dict[str, dict] = {}
        for w in data["widgets"]:
            widgets.setdefault(w["name"], w)
        out[form] = widgets
    return out


@pytest.fixture(scope="module")
def ground_truth() -> Dict[str, Dict[str, dict]]:
    gt = _load_ground_truth()
    if not gt:
        pytest.skip(
            "ground-truth JSONs not extracted — "
            "run: uv run python scripts/extract_pdf_widgets.py"
        )
    return gt


# ---------------------------------------------------------------------------
# Ground truth sanity
# ---------------------------------------------------------------------------

def test_ground_truth_covers_all_ten_forms(ground_truth):
    assert set(ground_truth.keys()) == {
        "125", "126", "127", "129", "130", "131", "137", "159", "160", "163",
    }


def test_every_form_has_widgets(ground_truth):
    for form, widgets in ground_truth.items():
        assert len(widgets) > 0, f"form {form} has zero widgets"


# ---------------------------------------------------------------------------
# Mapping invariants against PDF truth
# ---------------------------------------------------------------------------

def test_every_mapped_widget_exists_in_pdf(ground_truth):
    """Every ACORD field name in _FORM_ALIASES must be a real widget on the
    actual PDF. Stronger than the existing registry-spec check because this
    reads PDF bytes, not v3's legacy metadata."""
    errors = []
    for form, aliases in _FORM_ALIASES.items():
        widgets = ground_truth[form]
        for widget_name, schema_key in aliases.items():
            if widget_name not in widgets:
                errors.append(f"[{form}] {widget_name!r} → {schema_key!r}")
    assert not errors, (
        "Mapped widgets not found in PDF:\n  " + "\n  ".join(errors)
    )


def test_no_mapped_widget_is_readonly(ground_truth):
    """Read-only widgets are silently skipped by PyMuPDF at fill time.
    A mapping to one is a mapping to nothing."""
    errors = []
    for form, aliases in _FORM_ALIASES.items():
        widgets = ground_truth[form]
        for widget_name, schema_key in aliases.items():
            w = widgets.get(widget_name)
            if w and w["is_readonly"]:
                errors.append(
                    f"[{form}] {widget_name!r} → {schema_key!r} is read-only"
                )
    assert not errors, "Read-only widgets mapped:\n  " + "\n  ".join(errors)


def test_checkbox_concepts_map_to_checkbox_widgets(ground_truth):
    """LOB-discriminator indicators, @gl.occurrence, and explicit checkbox
    names must map to widgets with type=checkbox. Mapping a checkbox concept
    to a text widget means the bool True gets rendered as the string 'True'."""
    errors = []
    for form, aliases in _FORM_ALIASES.items():
        widgets = ground_truth[form]
        for widget_name, schema_key in aliases.items():
            w = widgets.get(widget_name)
            if not w:
                continue
            # Computed @-keys that are known-boolean:
            is_bool_concept = (
                schema_key.startswith("@lob.")
                or schema_key == "@gl.occurrence"
                or "Indicator" in widget_name
                or "CommercialGeneralLiability" in widget_name
            )
            if is_bool_concept and w["type"] not in ("checkbox", "radio"):
                errors.append(
                    f"[{form}] {widget_name!r} is {w['type']} "
                    f"but mapped as bool concept ({schema_key!r})"
                )
    assert not errors, (
        "Bool-concept widgets are not checkbox/radio:\n  "
        + "\n  ".join(errors)
    )


def test_text_mappings_do_not_target_checkbox_widgets(ground_truth):
    """Schema paths that resolve to string values (not @-prefixed computed
    resolvers, not indicator widgets) must map to text/combobox widgets."""
    errors = []
    for form, aliases in _FORM_ALIASES.items():
        widgets = ground_truth[form]
        for widget_name, schema_key in aliases.items():
            w = widgets.get(widget_name)
            if not w:
                continue
            if schema_key.startswith("@"):
                continue   # computed resolvers handle their own types
            if "Indicator" in widget_name:
                continue
            if w["type"] == "checkbox":
                errors.append(
                    f"[{form}] {widget_name!r} is checkbox "
                    f"but mapped to text schema key {schema_key!r}"
                )
    assert not errors, (
        "Checkbox widgets mapped as text:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Registry-spec agreement (v3 legacy JSON vs PDF truth)
# ---------------------------------------------------------------------------

def test_legacy_registry_spec_agrees_with_pdf_truth_on_names(ground_truth):
    """The registry's load_form_spec reads v3's form_fields/*.json. Those
    files should agree with PDF truth on unique widget names — if they
    don't, v3's metadata has drifted from the PDFs and our mapper can
    pass the legacy invariant while failing against reality. This test
    catches that drift before it causes production fill-failures."""
    from accord_ai.forms.registry import load_form_spec

    mismatches = []
    for form in ground_truth:
        spec = load_form_spec(form)
        legacy_names = set(spec.fields.keys())
        truth_names = set(ground_truth[form].keys())
        only_in_legacy = legacy_names - truth_names
        only_in_truth = truth_names - legacy_names
        if only_in_legacy or only_in_truth:
            mismatches.append(
                f"[{form}] legacy-only={len(only_in_legacy)} "
                f"truth-only={len(only_in_truth)} "
                f"(first legacy-only: {sorted(only_in_legacy)[:3]}, "
                f"first truth-only: {sorted(only_in_truth)[:3]})"
            )
    assert not mismatches, (
        "Legacy registry drift from PDF:\n  " + "\n  ".join(mismatches)
    )
