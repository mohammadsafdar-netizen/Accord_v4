"""Submission → filled PDFs pipeline.

Thin orchestrator over mapper + filler. The only new concept it adds is
the content hash: SHA-256 over the raw PDF bytes, used by FilledPdfStore
for dedup and by callers for detecting "did anything change since last
/complete". The hash is stable because the filler output is byte-stable
(no_new_id=True, deterministic deflate) — verified by the dedicated
test_deterministic_output_bytes test in P10.A.2.

No disk I/O here. fill_submission returns bytes + metadata; callers
decide whether to store, upload, or stream.

FE override merge (P10.0.f.3):
  `fill_submission` accepts an optional `field_overrides` kwarg keyed by
  form_number, where each value is `{widget_name: override_value}`. The
  override layer runs AFTER the mapper produces its per-form dict:
    * Override value wins when present — even when the mapper computed
      a value for the same widget.
    * Override value of "" (empty string) is treated as "clear this
      field" — the override key is dropped from the merged dict so the
      filler clears the widget (same semantic as mapper returning None).
    * Override keys absent from the caller's payload fall through to
      the mapper's value.
  Callers that don't need overrides can omit the kwarg; behavior is
  identical to the pre-P10.0.f.3 signature.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from accord_ai.forms.filler import FillResult, fill_form
from accord_ai.forms.mapper import map_submission
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("forms.pipeline")

# v3 FE sends form keys as "form_125"; v4 internal convention is "125".
# Accept either and normalize to the bare 3-4 digit number. Anchored both
# ends — "form_125xyz" is rejected, not silently truncated.
_FORM_KEY_RE = re.compile(r"^(?:form_)?(\d{3,4})$")


@dataclass(frozen=True)
class FilledForm:
    form_number: str
    pdf_bytes: bytes
    content_hash: str           # sha256 hex — 64 chars
    fill_result: FillResult

    def to_dict(self) -> dict:
        """Metadata-only projection — omits pdf_bytes. Used for API responses."""
        return {
            "form_number":  self.form_number,
            "content_hash": self.content_hash,
            "byte_length":  len(self.pdf_bytes),
            "fill_result":  self.fill_result.to_dict(),
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_overrides(
    field_overrides: Optional[Mapping[str, Mapping[str, object]]],
) -> Dict[str, Dict[str, object]]:
    """Normalize FE override payload keys (e.g. "form_125" → "125").

    * Keys matching ``^(form_)?\\d{3,4}$`` are accepted; the bare digit
      form is the internal convention.
    * Malformed keys (wrong shape, non-numeric suffix, extra junk) are
      skipped with a WARNING log — the merge continues for the valid
      entries. A broken FE payload must not tank /complete.
    * Non-dict values are also skipped with a warning; the override
      layer is `form_number -> {widget: value}` only.
    * The caller's mapping is never mutated — a defensive copy is
      returned.
    """
    if not field_overrides:
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for raw_key, widgets in field_overrides.items():
        if not isinstance(raw_key, str):
            _logger.warning(
                "fill_submission: non-string override key %r — skipping",
                raw_key,
            )
            continue
        match = _FORM_KEY_RE.match(raw_key)
        if match is None:
            _logger.warning(
                "fill_submission: malformed override key %r "
                "(expected 'NNN' or 'form_NNN') — skipping",
                raw_key,
            )
            continue
        if not isinstance(widgets, Mapping):
            _logger.warning(
                "fill_submission: override for form=%s is not a mapping "
                "(got %s) — skipping",
                raw_key, type(widgets).__name__,
            )
            continue
        form_number = match.group(1)
        # Defensive copy — never hold a reference to the caller's dict.
        out[form_number] = {str(k): v for k, v in widgets.items()}
    return out


def _merge_overrides(
    mapped: Mapping[str, object],
    overrides: Mapping[str, object],
) -> Dict[str, object]:
    """Merge one form's override dict into its mapper dict.

    Precedence rules:
      * Caller-supplied override value WINS over the mapper's value.
      * Override value of ``""`` (empty string) is treated as "clear
        this field" — the merged dict omits the key entirely, matching
        the filler's existing "absent-or-empty → clear" semantic.
      * Widget names absent from the override payload keep the mapper's
        value.
      * Unknown widget names (override keys not present in the mapper's
        output) are forwarded verbatim so the filler tallies them in
        `FillResult.unknown_fields` — no special handling here.
    """
    merged: Dict[str, object] = dict(mapped)
    for widget, value in overrides.items():
        if isinstance(value, str) and value == "":
            merged.pop(widget, None)
            continue
        merged[widget] = value
    return merged


def fill_submission(
    submission: CustomerSubmission,
    *,
    field_overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, FilledForm]:
    """Fill every form required by the submission's LOB.

    Args:
        submission: The CustomerSubmission to fill forms for.
        field_overrides: Optional FE override payload keyed by
            form_number (accepts both ``"125"`` and ``"form_125"``
            shapes; normalized internally). Values are
            ``{widget_name: override_value}`` dicts — caller-supplied
            values win over mapper output, and an empty-string value
            clears the widget. Unknown form keys are skipped with a
            WARNING log; unknown widget names are forwarded to the
            filler which tallies them as ``unknown_fields``. See module
            docstring for the full semantic.

    Returns:
        {form_number: FilledForm}, preserving the LOB's form ordering.

    Empty dict when submission.lob_details is None — a submission without
    an LOB has nothing to fill. (That's the caller's problem to report —
    this function is neutral.)

    Forms with an empty mapping (scaffolded but not yet populated — 137/
    159/160/163 as of P10.A.3b) still produce a FilledForm: a blank PDF
    with all widgets cleared. That's intentional — downstream code can
    include them in the upload manifest without special-casing.
    """
    mapped = map_submission(submission)
    normalized_overrides = _normalize_overrides(field_overrides)
    out: Dict[str, FilledForm] = {}
    override_forms_applied = 0
    for form_number, field_values in mapped.items():
        form_overrides = normalized_overrides.get(form_number)
        if form_overrides:
            merged = _merge_overrides(field_values, form_overrides)
            override_forms_applied += 1
        else:
            merged = dict(field_values)
        pdf_bytes, result = fill_form(form_number, merged)
        out[form_number] = FilledForm(
            form_number=form_number,
            pdf_bytes=pdf_bytes,
            content_hash=_sha256(pdf_bytes),
            fill_result=result,
        )

    # P10.0.f.4 / M1 — flag override form keys the LOB didn't emit. These
    # are silently dropped otherwise (e.g. FE editor sent corrections for
    # form 126 on a commercial_auto session whose LOB doesn't emit 126).
    # WARN level so ops can notice stale FE state without failing /complete.
    unapplied = sorted(set(normalized_overrides.keys()) - set(mapped.keys()))
    if unapplied:
        _logger.warning(
            "fill_submission: overrides supplied for forms not in this "
            "submission's LOB (unapplied=%s)",
            unapplied,
        )

    _logger.info(
        "fill_submission: forms=%d total_bytes=%d override_forms=%d "
        "unapplied_overrides=%d",
        len(out),
        sum(len(f.pdf_bytes) for f in out.values()),
        override_forms_applied,
        len(unapplied),
    )
    return out
