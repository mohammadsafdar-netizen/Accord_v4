"""Tests for flow_schema.py Pydantic models (Phase 3.1) — 6 tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from accord_ai.conversation.flow_schema import (
    AllCondition,
    AnyCondition,
    Condition,
    FieldEqualsCondition,
    FieldSetCondition,
    Flow,
    FlowsDocument,
    FlowTransition,
    Question,
)

_FLOWS_YAML = Path(__file__).parent.parent / "accord_ai" / "conversation" / "flows.yaml"
_ConditionAdapter = TypeAdapter(Condition)


def test_parse_minimal_flow_document():
    doc = FlowsDocument.model_validate(
        {
            "version": "1",
            "initial_flow": "start",
            "flows": [
                {
                    "id": "start",
                    "description": "entry",
                    "questions": [],
                    "next": [],
                }
            ],
        }
    )
    assert doc.initial_flow == "start"
    assert len(doc.flows) == 1
    assert doc.flows[0].id == "start"


def test_parse_full_flows_yaml():
    """loads the real flows.yaml and asserts all 11 flows are present."""
    import yaml

    raw = yaml.safe_load(_FLOWS_YAML.read_text())
    doc = FlowsDocument.model_validate(raw)
    assert len(doc.flows) == 11
    assert doc.initial_flow == "greet"


def test_condition_field_set_parses():
    c = _ConditionAdapter.validate_python({"kind": "field_set", "path": "ein"})
    assert isinstance(c, FieldSetCondition)
    assert c.path == "ein"


def test_condition_field_equals_parses():
    c = _ConditionAdapter.validate_python(
        {"kind": "field_equals", "path": "lob_details.lob", "value": "commercial_auto"}
    )
    assert isinstance(c, FieldEqualsCondition)
    assert c.value == "commercial_auto"


def test_condition_any_of_nested_parses():
    c = _ConditionAdapter.validate_python(
        {
            "kind": "any_of",
            "conditions": [
                {"kind": "field_set", "path": "naics_code"},
                {"kind": "field_set", "path": "naics_description"},
            ],
        }
    )
    assert isinstance(c, AnyCondition)
    assert len(c.conditions) == 2
    assert all(isinstance(sub, FieldSetCondition) for sub in c.conditions)


def test_invalid_condition_kind_fails():
    with pytest.raises(ValidationError):
        _ConditionAdapter.validate_python({"kind": "bogus", "path": "x"})
