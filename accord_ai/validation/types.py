"""Core types for the validation framework."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


Severity = Literal["info", "warning", "error"]


class ValidationFinding(BaseModel):
    validator: str
    field_path: str
    severity: Severity
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    validator: str
    ran_at: datetime
    duration_ms: float
    success: bool
    findings: List[ValidationFinding] = Field(default_factory=list)
    error: Optional[str] = None


class PrefillPatch(BaseModel):
    """Returned by Validator.prefill() — fields to add + conflicts to record."""
    patch: Dict[str, Any]           # partial submission shape, merged into submission
    conflicts: List[Any] = Field(default_factory=list)  # list of FieldConflict
    source: str
    confidence: float = 1.0


@runtime_checkable
class Validator(Protocol):
    name: str
    applicable_fields: tuple[str, ...]
    inline_eligible: bool    # True = runs during conversation turn (inline prefill)

    async def run(self, submission: Any) -> ValidationResult: ...

    async def prefill(
        self,
        submission: Any,
        just_extracted: dict,
    ) -> Optional[PrefillPatch]:
        """Called inline when applicable_fields may have just changed.

        Returns a PrefillPatch to apply, or None for no-op.
        Default implementation returns None — override for inline validators.
        """
        return None
