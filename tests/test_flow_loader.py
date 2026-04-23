"""Tests for flow_loader.py (Phase 3.1) — 8 tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from accord_ai.conversation.flow_loader import _validate_consistency, load_flows
from accord_ai.conversation.flow_schema import FlowsDocument

_REAL_FLOWS = Path(__file__).parent.parent / "accord_ai" / "conversation" / "flows.yaml"

_EXPECTED_FLOW_IDS = {
    "greet",
    "business_identity",
    "addresses",
    "contacts",
    "policy_dates",
    "ca_vehicles_drivers",
    "gl_operations_revenue",
    "wc_payroll_classes",
    "coverage",
    "hazmat_finalize",
    "finalize",
}


# ---------------------------------------------------------------------------
# Loader basic
# ---------------------------------------------------------------------------


def test_loader_returns_flows_document():
    doc = load_flows(_REAL_FLOWS)
    assert isinstance(doc, FlowsDocument)


def test_loader_cached_second_call_same_instance():
    doc1 = load_flows(_REAL_FLOWS)
    doc2 = load_flows(_REAL_FLOWS)
    assert doc1 is doc2


# ---------------------------------------------------------------------------
# Consistency validation errors
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: dict) -> Path:
    out = path / "flows.yaml"
    out.write_text(yaml.dump(content))
    return out


def _minimal_valid() -> dict:
    return {
        "version": "1",
        "initial_flow": "start",
        "flows": [{"id": "start", "description": "entry", "questions": [], "next": []}],
    }


def test_loader_rejects_unknown_initial_flow(tmp_path):
    data = _minimal_valid()
    data["initial_flow"] = "no_such_flow"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="initial_flow"):
        load_flows(p)


def test_loader_rejects_transition_to_unknown_flow(tmp_path):
    data = _minimal_valid()
    data["flows"][0]["next"] = [{"flow": "ghost"}]
    p = tmp_path / "bad_trans.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="unknown flow"):
        load_flows(p)


def test_loader_rejects_duplicate_question_ids_in_flow(tmp_path):
    data = _minimal_valid()
    data["flows"][0]["questions"] = [
        {"id": "q1", "text": "first"},
        {"id": "q1", "text": "duplicate"},
    ]
    p = tmp_path / "dup_q.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="duplicate question ids"):
        load_flows(p)


def test_loader_rejects_yaml_parse_error(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("key: [unclosed")
    with pytest.raises(yaml.YAMLError):
        load_flows(p)


# ---------------------------------------------------------------------------
# Content sanity
# ---------------------------------------------------------------------------


def test_flows_yaml_has_all_9_canonical_flows():
    """Asserts all 11 expected flow ids are present in flows.yaml."""
    doc = load_flows(_REAL_FLOWS)
    found_ids = {f.id for f in doc.flows}
    assert found_ids == _EXPECTED_FLOW_IDS


def test_required_fields_match_critical_fields_plugins():
    """LOB-specific required_fields must be ⊆ that plugin's critical_fields."""
    import accord_ai.lobs  # noqa: F401 — trigger plugin registration
    from accord_ai.lobs.registry import get_critical_fields

    doc = load_flows(_REAL_FLOWS)
    by_id = {f.id: f for f in doc.flows}

    checks = [
        ("ca_vehicles_drivers", "commercial_auto"),
        ("gl_operations_revenue", "general_liability"),
        ("wc_payroll_classes", "workers_comp"),
    ]

    for flow_id, lob_key in checks:
        flow = by_id[flow_id]
        critical_paths = {path for path, _ in get_critical_fields(lob_key)}
        for req in flow.required_fields:
            assert req in critical_paths, (
                f"flow {flow_id!r} required_field {req!r} not in "
                f"{lob_key} critical_fields: {sorted(critical_paths)}"
            )
