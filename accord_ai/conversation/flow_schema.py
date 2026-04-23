"""Pydantic models for the conversation flows YAML document (Phase 3.1).

This IS the contract between flows.yaml and the Phase 3.2 state machine.
The Condition discriminated union supports recursive nesting (all_of / any_of).
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Condition nodes (discriminated on `kind`)
# ---------------------------------------------------------------------------


class FieldSetCondition(BaseModel):
    """True when the submission path has a non-empty / non-None value."""

    kind: Literal["field_set"] = "field_set"
    path: str  # dot-path, e.g. "ein", "lob_details.vehicles[0].vin"


class FieldEqualsCondition(BaseModel):
    """True when the submission path equals the given value."""

    kind: Literal["field_equals"] = "field_equals"
    path: str
    value: Union[str, int, bool]


class AllCondition(BaseModel):
    """True when every nested condition is true."""

    kind: Literal["all_of"] = "all_of"
    conditions: List[Condition]  # forward ref — resolved by model_rebuild()


class AnyCondition(BaseModel):
    """True when at least one nested condition is true."""

    kind: Literal["any_of"] = "any_of"
    conditions: List[Condition]  # forward ref — resolved by model_rebuild()


Condition = Annotated[
    Union[FieldSetCondition, FieldEqualsCondition, AllCondition, AnyCondition],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Flow building blocks
# ---------------------------------------------------------------------------


class Question(BaseModel):
    id: str                             # unique within its flow
    text: str                           # prompt shown to the user
    expected_fields: List[str] = []     # submission paths this question fills
    skip_when: Optional[Condition] = None  # skip if already satisfied


class FlowTransition(BaseModel):
    when: Optional[Condition] = None    # None = unconditional (last-resort default)
    flow: str                           # flow id to transition to


class Flow(BaseModel):
    id: str
    description: str
    entry_conditions: List[Condition] = []  # all must be true to enter
    required_fields: List[str] = []         # must be filled before the flow exits
    questions: List[Question] = []
    next: List[FlowTransition] = []         # evaluated in order; first match wins


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------


class FlowsDocument(BaseModel):
    version: str = "1"
    initial_flow: str       # id of the first flow (usually "greet")
    flows: List[Flow]

    def by_id(self, flow_id: str) -> Flow:
        for f in self.flows:
            if f.id == flow_id:
                return f
        raise KeyError(f"flow not found: {flow_id!r}")


# Rebuild after Condition is fully defined to resolve the forward references
# in AllCondition.conditions and AnyCondition.conditions.
AllCondition.model_rebuild()
AnyCondition.model_rebuild()
FlowsDocument.model_rebuild()
