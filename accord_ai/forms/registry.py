"""Form registry — reads form_templates/ + form_fields/ as package data.

Shape on disk (inherited verbatim from v3 so we can reuse the same assets):

  accord_ai/form_templates/acord_<N>_blank.pdf   — ACORD blank PDF
  accord_ai/form_fields/form_<N>_fields.json     — {form_number, field_count,
                                                    text_field_count,
                                                    checkbox_field_count,
                                                    fields: {name: {type, tooltip}}}

LOB → form list is canonical here. v3 encoded it on each LobPlugin; v4
centralizes it so the registry is the only source of truth. When a new LOB
or form lands, update _LOB_FORMS below and drop the blank PDF + field spec
into the two directories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Tuple

# Local Lob alias — schema.py encodes the three LOB literals inside each
# discriminated-union model but doesn't export a named alias. Keeping the
# list here avoids cross-module churn; if schema.py grows one in the
# future, swap the import in.
Lob = Literal["commercial_auto", "general_liability", "workers_comp"]

# ----------------------------------------------------------------------------
# Paths — resolved once at import.
# ----------------------------------------------------------------------------

_MODULE_DIR     = Path(__file__).resolve().parent.parent      # accord_ai/
_TEMPLATES_DIR  = _MODULE_DIR / "form_templates"
_FIELDS_DIR     = _MODULE_DIR / "form_fields"


# ----------------------------------------------------------------------------
# LOB → ordered list of form numbers.
#
# Ordering matters: the first form is the "front page" (ACORD 125 for most
# commercial lines) and downstream code uses index 0 as the default when
# only one form can be rendered. Keep 125 first wherever it appears.
# ----------------------------------------------------------------------------

_LOB_FORMS: Mapping[Lob, Tuple[str, ...]] = {
    "commercial_auto":   ("125", "127", "129", "137", "163"),
    "general_liability": ("125", "126"),
    "workers_comp":      ("125", "130"),
}


# Three supported PDF AcroForm widget types. Dropdowns are stored in the v3
# JSON assets as their own type, but the summary `text_field_count` counts
# text+dropdown together (see cross-check in load_form_spec below).
FieldType = Literal["text", "checkbox", "dropdown"]


@dataclass(frozen=True)
class FormField:
    name: str
    type: FieldType
    tooltip: str


@dataclass(frozen=True)
class FormSpec:
    form_number: str
    field_count: int
    text_field_count: int
    checkbox_field_count: int
    fields: Mapping[str, FormField]      # keyed by field name
    template_path: Path                  # absolute path to the blank PDF


class UnknownFormError(KeyError):
    """Raised for an unknown form number or LOB."""


# ----------------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------------

def forms_for_lob(lob: Lob) -> List[str]:
    """Return ordered list of form numbers required for a LOB.

    Always returns a fresh list (callers may mutate for their own purposes).
    """
    try:
        return list(_LOB_FORMS[lob])
    except KeyError as e:
        raise UnknownFormError(f"no forms registered for LOB: {lob!r}") from e


def list_all_forms() -> List[str]:
    """All form numbers present on disk, sorted numerically."""
    out = []
    for p in _FIELDS_DIR.glob("form_*_fields.json"):
        # filename pattern: form_<N>_fields.json
        parts = p.stem.split("_")
        if len(parts) >= 2:
            out.append(parts[1])
    return sorted(out, key=lambda s: (len(s), s))


def template_path(form_number: str) -> Path:
    """Absolute path to the blank PDF. Raises UnknownFormError if missing."""
    p = _TEMPLATES_DIR / f"acord_{form_number}_blank.pdf"
    if not p.is_file():
        raise UnknownFormError(
            f"blank template not found for form {form_number}: {p}"
        )
    return p


@lru_cache(maxsize=None)
def load_form_spec(form_number: str) -> FormSpec:
    """Load + parse a form field spec. Cached — specs are immutable."""
    json_path = _FIELDS_DIR / f"form_{form_number}_fields.json"
    if not json_path.is_file():
        raise UnknownFormError(
            f"no field spec for form {form_number}: {json_path}"
        )

    data = json.loads(json_path.read_text())

    # Validate PDF exists alongside the spec — fail loud at registry load time,
    # not deep inside the PDF filler.
    pdf_path = template_path(form_number)

    fields: Dict[str, FormField] = {}
    for name, meta in data.get("fields", {}).items():
        ftype = meta.get("type")
        if ftype not in ("text", "checkbox", "dropdown"):
            raise ValueError(
                f"form {form_number}: field {name!r} has unsupported "
                f"type {ftype!r}"
            )
        fields[name] = FormField(
            name=name, type=ftype, tooltip=meta.get("tooltip", ""),
        )

    spec = FormSpec(
        form_number=str(data["form_number"]),
        field_count=int(data["field_count"]),
        text_field_count=int(data["text_field_count"]),
        checkbox_field_count=int(data["checkbox_field_count"]),
        fields=fields,
        template_path=pdf_path,
    )

    # Cross-check counts so a corrupted JSON surfaces immediately.
    if len(spec.fields) != spec.field_count:
        raise ValueError(
            f"form {form_number}: field_count={spec.field_count} but "
            f"fields dict has {len(spec.fields)} entries"
        )
    text_n     = sum(1 for f in spec.fields.values() if f.type == "text")
    checkbox_n = sum(1 for f in spec.fields.values() if f.type == "checkbox")
    dropdown_n = sum(1 for f in spec.fields.values() if f.type == "dropdown")
    # v3 convention: text_field_count == text + dropdown (dropdowns are
    # serialized alongside text inputs in the ACORD widget taxonomy).
    if (
        text_n + dropdown_n != spec.text_field_count
        or checkbox_n != spec.checkbox_field_count
    ):
        raise ValueError(
            f"form {form_number}: declared text={spec.text_field_count}/"
            f"checkbox={spec.checkbox_field_count} but actual "
            f"text+dropdown={text_n + dropdown_n}/checkbox={checkbox_n}"
        )
    return spec
