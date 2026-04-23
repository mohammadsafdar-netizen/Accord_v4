"""Fill ACORD templates with field values.

Returns (pdf_bytes, FillResult). No disk I/O — callers own writes.

Widget handling (PyMuPDF widget types):
  TEXT (7)       : widget.field_value = str(value)
  CHECKBOX (2)   : widget.field_value = truthy(value)
  COMBOBOX (3)   : treat like text
  LISTBOX (4)    : treat like text
  RADIOBUTTON (5): option label, or "Off" for falsy
  SIGNATURE (6)  : skipped — no programmatic fill

Byte-stability: doc.tobytes uses no_new_id=True + deflate=True + garbage=4,
so identical inputs produce identical outputs. Downstream content-hash
dedup (P10.C upload cache) depends on this.

The /AP null trick:
  ACORD 163 (and some hand-authored ACORD templates) encode widget
  appearance streams /AP as indirect xrefs or share them across widgets.
  widget.update() doesn't always regenerate those streams — the value
  gets stored in /V but the visible rendering still shows the template
  sample. Nulling /AP first forces update() to rebuild a fresh dict
  entry from /DA + /V + /Rect. Applied both when clearing and when
  filling. This is the single most important v3 PDF correctness fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from accord_ai.forms.registry import (
    UnknownFormError,
    load_form_spec,
    template_path,
)
from accord_ai.logging_config import get_logger

_logger = get_logger("forms.filler")


_TRUTHY = {"1", "true", "yes", "y", "on", "checked", "x"}


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


@dataclass(frozen=True)
class FillResult:
    form_number: str
    filled_count: int
    skipped_count: int
    error_count: int
    errors: Tuple[str, ...]
    unknown_fields: Tuple[str, ...]     # field names not present in the form

    def to_dict(self) -> dict:
        return {
            "form_number":    self.form_number,
            "filled":         self.filled_count,
            "skipped":        self.skipped_count,
            "errors":         self.error_count,
            "error_messages": list(self.errors[:5]),
            "unknown_fields": list(self.unknown_fields[:10]),
        }


@dataclass
class _FillCounters:
    """Internal accumulator — frozen FillResult built at the end."""
    filled: int = 0
    skipped: int = 0
    errors_n: int = 0
    errors: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


@lru_cache(maxsize=32)
def _load_template_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _clear_widget(widget, fitz_mod, doc) -> None:
    """Clear a widget's value. Swallows per-widget errors at debug level."""
    ftype = widget.field_type
    try:
        if ftype == fitz_mod.PDF_WIDGET_TYPE_CHECKBOX:
            widget.field_value = False
            widget.update()
        elif ftype == fitz_mod.PDF_WIDGET_TYPE_RADIOBUTTON:
            widget.field_value = "Off"
            widget.update()
        elif ftype == fitz_mod.PDF_WIDGET_TYPE_SIGNATURE:
            return
        else:
            # text / combo / listbox / unknown: xref-level rewrite — setting
            # widget.field_value = "" doesn't survive save/reopen on many
            # ACORD templates. Null /AP so update() regenerates appearance.
            xref = widget.xref
            if xref:
                for key, value in (("V", "()"), ("AP", "null")):
                    try:
                        doc.xref_set_key(xref, key, value)
                    except Exception:
                        pass
            try:
                widget.field_value = ""
                widget.update()
            except Exception:
                pass
    except Exception as exc:
        _logger.debug(
            "clear widget failed: name=%s type=%s err=%s",
            widget.field_name, widget.field_type, exc,
        )


def _assign_widget(widget, value, fitz_mod) -> bool:
    """Assign value per widget type. Returns False if skipped (signature)."""
    ftype = widget.field_type
    if ftype == fitz_mod.PDF_WIDGET_TYPE_CHECKBOX:
        widget.field_value = _is_truthy(value)
        return True
    if ftype == fitz_mod.PDF_WIDGET_TYPE_RADIOBUTTON:
        if _is_truthy(value) or (isinstance(value, str) and value.strip()):
            widget.field_value = str(value)
        else:
            widget.field_value = "Off"
        return True
    if ftype == fitz_mod.PDF_WIDGET_TYPE_SIGNATURE:
        return False
    # text / combo / listbox / unknown
    widget.field_value = str(value)
    return True


def fill_form(
    form_number: str,
    field_values: Mapping[str, object],
    *,
    template_override: Optional[Path] = None,
) -> Tuple[bytes, FillResult]:
    """Fill an ACORD blank with `field_values`. Returns (pdf_bytes, FillResult).

    Raises:
        UnknownFormError: form_number is not registered (no template/spec).
        RuntimeError: PyMuPDF is not installed.

    Never raises on per-field failure — those are tallied into FillResult.
    Empty-string / None values are treated as "clear this field", not as fills.
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF filling") from e

    # Validate form_number against the registry — fail loud for unknown forms.
    spec = load_form_spec(form_number)
    tpl_path = (
        template_override if template_override is not None
        else template_path(form_number)
    )
    if not tpl_path.is_file():
        raise UnknownFormError(f"template not found: {tpl_path}")

    counters = _FillCounters()

    # Pre-compute field-name presence check against the spec so we can
    # report unknown fields without walking the PDF first.
    spec_fields = spec.fields
    requested_non_empty: Dict[str, object] = {}
    for name, value in field_values.items():
        if name not in spec_fields:
            counters.unknown.append(name)
            continue
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        requested_non_empty[name] = value

    template_bytes = _load_template_bytes(tpl_path)
    doc = fitz.open(stream=bytearray(template_bytes), filetype="pdf")
    try:
        # Step 1 — clear every writable widget we aren't about to fill.
        fill_keys = set(requested_non_empty.keys())
        for page in doc:
            for widget in page.widgets():
                if widget.field_flags & 1:                     # read-only
                    continue
                if widget.field_name in fill_keys:
                    continue
                _clear_widget(widget, fitz, doc)

        # Step 2 — fill.
        filled_names: set = set()
        for page in doc:
            for widget in page.widgets():
                fname = widget.field_name
                if fname not in requested_non_empty:
                    continue
                if widget.field_flags & 1:
                    counters.skipped += 1
                    continue
                rect = widget.rect
                if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                    counters.skipped += 1
                    continue

                # /AP null first — see module docstring.
                xref = widget.xref
                if xref:
                    try:
                        doc.xref_set_key(xref, "AP", "null")
                    except Exception:
                        pass

                try:
                    if not _assign_widget(
                        widget, requested_non_empty[fname], fitz,
                    ):
                        counters.skipped += 1
                        continue
                    widget.update()
                    filled_names.add(fname)
                    counters.filled += 1
                except Exception as exc:
                    counters.errors.append(f"{fname}: {exc}")
                    counters.errors_n += 1

        # Fields requested but never matched to a widget (e.g. widget
        # stripped by an unusual template variant) — count as skipped.
        unmatched = (
            len(requested_non_empty) - len(filled_names) - counters.errors_n
        )
        counters.skipped += max(0, unmatched - counters.skipped)

        pdf_bytes = doc.tobytes(deflate=True, garbage=4, no_new_id=True)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    result = FillResult(
        form_number=form_number,
        filled_count=counters.filled,
        skipped_count=counters.skipped,
        error_count=counters.errors_n,
        errors=tuple(counters.errors),
        unknown_fields=tuple(counters.unknown),
    )
    _logger.info(
        "form %s: filled=%d skipped=%d errors=%d unknown=%d",
        form_number, result.filled_count, result.skipped_count,
        result.error_count, len(result.unknown_fields),
    )
    return pdf_bytes, result
