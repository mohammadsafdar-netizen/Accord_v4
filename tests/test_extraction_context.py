"""Tests for ExtractionContext (Phase 3.3) — 4 unit + 4 FlowEngine helper +
4 extractor rendering + 4 controller integration = 16 tests total.

FlowEngine helper and extractor/controller tests live in their own files
(test_flow_engine.py, test_extractor.py, test_controller.py). This file
covers ExtractionContext unit tests only.
"""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from accord_ai.extraction.context import EMPTY_CONTEXT, ExtractionContext


def test_empty_context_is_empty_true():
    assert ExtractionContext().is_empty is True


def test_context_with_just_flow_not_empty():
    ctx = ExtractionContext(current_flow="greet")
    assert ctx.is_empty is False


def test_context_with_just_expected_fields_not_empty():
    ctx = ExtractionContext(expected_fields=("ein",))
    assert ctx.is_empty is False


def test_context_preserves_tuples_and_is_frozen():
    ctx = ExtractionContext(
        current_flow="business_identity",
        expected_fields=("ein", "entity_type"),
    )
    assert ctx.expected_fields == ("ein", "entity_type")
    with pytest.raises(FrozenInstanceError):
        ctx.current_flow = "other"  # type: ignore[misc]


def test_EMPTY_CONTEXT_module_level_singleton():
    assert EMPTY_CONTEXT is EMPTY_CONTEXT  # trivially true
    assert EMPTY_CONTEXT.is_empty is True
    assert isinstance(EMPTY_CONTEXT, ExtractionContext)
    # Confirm it's the actual module-level sentinel, not a new instance
    from accord_ai.extraction.context import EMPTY_CONTEXT as EC2
    assert EMPTY_CONTEXT is EC2
