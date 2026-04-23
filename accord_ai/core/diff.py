"""Submission-diff application — scalars (6.a) + nested-model recursion (6.b).

Later phases refine this module without changing `apply_diff`'s signature:
  - 6.d: LOB transition rejection via LobTransitionError

List merge (6.c): replace-if-longer-or-equal. When the diff's list is at
least as long as current's, replace; otherwise keep current. Protects
against an LLM turn that accidentally drops items from an earlier turn.

Aliasing: `model_copy(update=...)` is a shallow copy. Fields touched by the
diff are rebuilt (scalars replaced, nested models reconstructed via
recursive merge, lists either replaced or kept-as-is). Fields NOT in
`diff.model_fields_set` share their reference with `current`. Because our
models set `validate_assignment=True`, accidental mutation of a shared
nested field raises — safe in practice. Don't mutate the returned tree;
always go through `apply_diff`.
"""
from __future__ import annotations

from pydantic import BaseModel

from accord_ai.core.vehicle_merge import merge_drivers, merge_vehicles
from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    GeneralLiabilityDetails,
    WorkersCompDetails,
)

# Members of the LobDetails discriminated union. A diff that swaps between
# any two of these is a mid-session LOB switch — policy says reject.
_LOB_DETAILS_TYPES = (CommercialAutoDetails, GeneralLiabilityDetails, WorkersCompDetails)


class LobTransitionError(ValueError):
    """Raised when `apply_diff` would change the LOB discriminator.

    Orchestrator policy: once a session has a LOB set, subsequent diffs
    cannot change it. Callers catch this to decide: start a new session,
    escalate to a human, or coerce the diff.
    """


def apply_diff(
    current: CustomerSubmission,
    diff: CustomerSubmission,
) -> CustomerSubmission:
    """Return current overlaid with diff. See module docstring for rules."""
    return _merge_model(current, diff)


def _merge_model(current: BaseModel, diff: BaseModel) -> BaseModel:
    """Generic recursive merge — same rules as apply_diff, on any BaseModel."""
    assert type(current) is type(diff), (
        f"_merge_model requires same-type args, got "
        f"{type(current).__name__} vs {type(diff).__name__}"
    )
    updates = {}
    for field_name in diff.model_fields_set:
        new_value = getattr(diff, field_name)
        if new_value is None:
            continue   # loose removal protection

        old_value = getattr(current, field_name)

        # Both are models of the same concrete type → recurse
        if (
            isinstance(new_value, BaseModel)
            and isinstance(old_value, BaseModel)
            and type(new_value) is type(old_value)
        ):
            updates[field_name] = _merge_model(old_value, new_value)
            continue

        # List merge — vehicles/drivers get 3-tier VIN-primary merge;
        # all other lists use the 6.c policy (replace-if-longer-or-equal).
        if isinstance(new_value, list) and isinstance(old_value, list):
            if isinstance(current, CommercialAutoDetails):
                if field_name == "vehicles":
                    updates[field_name] = merge_vehicles(old_value, new_value)
                    continue
                if field_name == "drivers":
                    updates[field_name] = merge_drivers(old_value, new_value)
                    continue
            # 6.c fallback: replace only if diff's list is at least as long.
            if len(new_value) >= len(old_value):
                updates[field_name] = new_value
            # else: keep current's list (don't add to updates)
            continue

        # LOB transition guard (6.d): the discriminated union picks one of three
        # *Details subclasses. If current and diff are two *different* LOB
        # subclasses, someone is switching LOB mid-session — reject explicitly.
        if (
            isinstance(new_value, _LOB_DETAILS_TYPES)
            and isinstance(old_value, _LOB_DETAILS_TYPES)
        ):
            raise LobTransitionError(
                f"cannot transition LOB from {type(old_value).__name__} "
                f"to {type(new_value).__name__} — lob_details is immutable once set"
            )

        # Type mismatch (non-LOB) or scalar. Naive replace.
        updates[field_name] = new_value

    return current.model_copy(update=updates)
