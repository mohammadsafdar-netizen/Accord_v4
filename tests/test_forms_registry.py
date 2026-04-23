"""Tests for the form registry (P10.A.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from accord_ai.forms import (
    FormSpec,
    UnknownFormError,
    forms_for_lob,
    list_all_forms,
    load_form_spec,
    template_path,
)


# --- LOB → forms -------------------------------------------------------------

def test_commercial_auto_forms():
    assert forms_for_lob("commercial_auto") == ["125", "127", "129", "137", "163"]


def test_general_liability_forms():
    assert forms_for_lob("general_liability") == ["125", "126"]


def test_workers_comp_forms():
    assert forms_for_lob("workers_comp") == ["125", "130"]


def test_forms_for_lob_returns_fresh_list():
    a = forms_for_lob("commercial_auto")
    a.append("999")
    b = forms_for_lob("commercial_auto")
    assert "999" not in b


def test_forms_for_lob_unknown():
    with pytest.raises(UnknownFormError):
        forms_for_lob("personal_auto")  # type: ignore[arg-type]


# --- Disk inventory ----------------------------------------------------------

def test_list_all_forms_matches_disk():
    forms = list_all_forms()
    # v3 assets copied in: 10 blanks.
    assert set(forms) == {
        "125", "126", "127", "129", "130",
        "131", "137", "159", "160", "163",
    }
    # Sorted numerically (by length then lexicographic).
    assert forms == sorted(forms, key=lambda s: (len(s), s))


def test_every_lob_form_exists_on_disk():
    disk = set(list_all_forms())
    for lob in ("commercial_auto", "general_liability", "workers_comp"):
        for f in forms_for_lob(lob):  # type: ignore[arg-type]
            assert f in disk, f"LOB {lob} references missing form {f}"


# --- Template path -----------------------------------------------------------

def test_template_path_returns_existing_pdf():
    p = template_path("125")
    assert isinstance(p, Path) and p.is_file()
    assert p.name == "acord_125_blank.pdf"


def test_template_path_unknown():
    with pytest.raises(UnknownFormError):
        template_path("999")


# --- Field spec loading ------------------------------------------------------

def test_load_form_spec_125_structure():
    spec = load_form_spec("125")
    assert isinstance(spec, FormSpec)
    assert spec.form_number == "125"
    assert spec.field_count == len(spec.fields) > 0
    # v3 data: form 125 has 548 fields (384 text + 164 checkbox).
    assert spec.field_count == 548
    assert spec.text_field_count == 384
    assert spec.checkbox_field_count == 164
    assert spec.template_path.is_file()


def test_load_form_spec_all_forms_parse():
    for f in list_all_forms():
        spec = load_form_spec(f)
        assert spec.form_number == f
        assert spec.field_count > 0


def test_load_form_spec_fields_are_typed():
    spec = load_form_spec("125")
    sample = next(iter(spec.fields.values()))
    assert sample.type in ("text", "checkbox")
    assert isinstance(sample.tooltip, str)


def test_load_form_spec_is_cached():
    a = load_form_spec("125")
    b = load_form_spec("125")
    assert a is b   # lru_cache returns the same instance


def test_load_form_spec_unknown():
    with pytest.raises(UnknownFormError):
        load_form_spec("999")


# --- Corruption detection ----------------------------------------------------

def test_corrupted_spec_count_mismatch(tmp_path, monkeypatch):
    """Hand-craft a bad spec and verify load raises ValueError."""
    from accord_ai.forms import registry as reg

    bad_fields_dir    = tmp_path / "form_fields"
    bad_templates_dir = tmp_path / "form_templates"
    bad_fields_dir.mkdir()
    bad_templates_dir.mkdir()

    (bad_fields_dir / "form_888_fields.json").write_text(json.dumps({
        "form_number": "888",
        "field_count": 5,           # lies
        "text_field_count": 1,
        "checkbox_field_count": 0,
        "fields": {"x": {"type": "text", "tooltip": ""}},  # actually 1
    }))
    (bad_templates_dir / "acord_888_blank.pdf").write_bytes(b"%PDF-1.4\n%mock\n")

    monkeypatch.setattr(reg, "_FIELDS_DIR",    bad_fields_dir)
    monkeypatch.setattr(reg, "_TEMPLATES_DIR", bad_templates_dir)
    reg.load_form_spec.cache_clear()

    with pytest.raises(ValueError, match="field_count=5"):
        reg.load_form_spec("888")


def test_corrupted_spec_bad_field_type(tmp_path, monkeypatch):
    from accord_ai.forms import registry as reg

    bad_fields_dir    = tmp_path / "form_fields"
    bad_templates_dir = tmp_path / "form_templates"
    bad_fields_dir.mkdir()
    bad_templates_dir.mkdir()

    (bad_fields_dir / "form_889_fields.json").write_text(json.dumps({
        "form_number": "889",
        "field_count": 1,
        "text_field_count": 0,
        "checkbox_field_count": 0,
        "fields": {"x": {"type": "signature", "tooltip": ""}},  # unsupported
    }))
    (bad_templates_dir / "acord_889_blank.pdf").write_bytes(b"%PDF-1.4\n%mock\n")

    monkeypatch.setattr(reg, "_FIELDS_DIR",    bad_fields_dir)
    monkeypatch.setattr(reg, "_TEMPLATES_DIR", bad_templates_dir)
    reg.load_form_spec.cache_clear()

    with pytest.raises(ValueError, match="unsupported type"):
        reg.load_form_spec("889")
