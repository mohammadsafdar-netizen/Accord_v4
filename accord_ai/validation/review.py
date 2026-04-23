"""Review payload transformer — converts ValidationResult list into FE-ready shape.

build_review_payload() takes the raw validator outputs + the submission's stored
FieldConflict list and produces a ReviewPayload grouped into:
  - conflicts: user-vs-enriched disagreements (from inline prefill, surfaced now)
  - compliance: OFAC/FMCSA/SAM/SEC/Tax1099 grouped as pass/fail items
  - warnings: everything else at warning or error severity
  - info: informational findings
  - prefills: deferred prefill seat (empty until Phase 3)
"""

from __future__ import annotations

import uuid
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from accord_ai.validation.types import Severity, ValidationFinding, ValidationResult


# ---------------------------------------------------------------------------
# Review payload models
# ---------------------------------------------------------------------------


class ReviewSummary(BaseModel):
    errors: int = 0
    warnings: int = 0
    info: int = 0
    conflicts_pending: int = 0
    prefills_pending: int = 0


class ConflictItem(BaseModel):
    id: str
    field_path: str
    user_value: Any
    enriched_value: Any
    source: str
    severity: Severity = "warning"
    message: str


class PrefillItem(BaseModel):
    id: str
    field_path: str
    suggested_value: Any
    source: str
    confidence: float = 1.0


class ComplianceItem(BaseModel):
    validator: str
    status: Literal["clean", "verified", "warning", "error"]
    label: str
    findings: List[ValidationFinding] = Field(default_factory=list)


class ReviewPayload(BaseModel):
    session_id: str
    ready_to_finalize: bool
    summary: ReviewSummary
    conflicts: List[ConflictItem] = Field(default_factory=list)
    prefills: List[PrefillItem] = Field(default_factory=list)  # Phase 3 seat
    compliance: List[ComplianceItem] = Field(default_factory=list)
    warnings: List[ValidationFinding] = Field(default_factory=list)
    info: List[ValidationFinding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification config
# ---------------------------------------------------------------------------

_COMPLIANCE_VALIDATORS = frozenset({
    "ofac_sdn",
    "fmcsa_safer",
    "sam_gov",
    "sec_edgar",
    "tax1099",
})

_COMPLIANCE_LABELS = {
    "ofac_sdn":    "OFAC SDN Screening",
    "fmcsa_safer": "FMCSA SAFER Carrier Check",
    "sam_gov":     "SAM.gov Registration",
    "sec_edgar":   "SEC EDGAR Filing",
    "tax1099":     "IRS TIN Matching",
}


def _status_from_findings(findings: List[ValidationFinding]) -> str:
    if not findings:
        return "clean"
    severities = {f.severity for f in findings}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    return "verified"


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def build_review_payload(
    session_id: str,
    submission: Any,
    results: List[ValidationResult],
) -> ReviewPayload:
    """Transform raw validator results + submission conflicts → ReviewPayload."""
    # --- Conflicts from inline prefill (FieldConflict entries on submission) ---
    # Only unresolved conflicts count — resolved ones are kept for audit but
    # do not block ready_to_finalize.
    raw_conflicts = [
        c for c in (getattr(submission, "conflicts", None) or [])
        if not getattr(c, "resolved", False)
    ]
    conflicts: List[ConflictItem] = []
    for c in raw_conflicts:
        field_path = getattr(c, "field_path", "") or ""
        user_val = getattr(c, "user_value", None)
        enriched_val = getattr(c, "enriched_value", None)
        source = getattr(c, "source", "unknown") or "unknown"
        conflicts.append(ConflictItem(
            id=str(uuid.uuid4()),
            field_path=field_path,
            user_value=user_val,
            enriched_value=enriched_val,
            source=source,
            severity="warning",
            message=(
                f"{source} reports {field_path!r} should be "
                f"{enriched_val!r} (user said {user_val!r})"
            ),
        ))

    # --- Bucket findings by validator type ---
    compliance: List[ComplianceItem] = []
    warnings: List[ValidationFinding] = []
    info_items: List[ValidationFinding] = []
    errors_count = 0

    for result in results:
        if result.validator in _COMPLIANCE_VALIDATORS:
            compliance.append(ComplianceItem(
                validator=result.validator,
                status=_status_from_findings(result.findings),
                label=_COMPLIANCE_LABELS.get(result.validator, result.validator),
                findings=result.findings,
            ))
            # Compliance errors still count toward error total
            for f in result.findings:
                if f.severity == "error":
                    errors_count += 1
        else:
            for finding in result.findings:
                if finding.severity == "error":
                    errors_count += 1
                    warnings.append(finding)   # errors surface in the warnings list
                elif finding.severity == "warning":
                    warnings.append(finding)
                else:
                    info_items.append(finding)

    summary = ReviewSummary(
        errors=errors_count,
        warnings=len(warnings),
        info=len(info_items),
        conflicts_pending=len(conflicts),
        prefills_pending=0,
    )

    return ReviewPayload(
        session_id=session_id,
        ready_to_finalize=(errors_count == 0 and len(conflicts) == 0),
        summary=summary,
        conflicts=conflicts,
        prefills=[],
        compliance=compliance,
        warnings=warnings,
        info=info_items,
    )
