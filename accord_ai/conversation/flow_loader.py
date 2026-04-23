"""Loads and validates the flows YAML document (Phase 3.1).

`load_flows()` is the only public entrypoint. It parses flows.yaml into a
typed FlowsDocument, runs structural consistency checks, and caches the
result so repeated calls within a process are free.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from accord_ai.conversation.flow_schema import FlowsDocument

_DEFAULT_PATH = Path(__file__).parent / "flows.yaml"


@lru_cache(maxsize=4)
def load_flows(path: Path = _DEFAULT_PATH) -> FlowsDocument:
    """Parse *path* into a FlowsDocument, validate structural consistency, cache."""
    raw = yaml.safe_load(path.read_text())
    doc = FlowsDocument.model_validate(raw)
    _validate_consistency(doc)
    return doc


def _validate_consistency(doc: FlowsDocument) -> None:
    """Structural checks the Pydantic schema cannot express.

    Raises ValueError on the first violation found.
    """
    flow_ids = {f.id for f in doc.flows}

    if doc.initial_flow not in flow_ids:
        raise ValueError(
            f"initial_flow {doc.initial_flow!r} not in flows: {sorted(flow_ids)}"
        )

    for flow in doc.flows:
        q_ids = [q.id for q in flow.questions]
        if len(q_ids) != len(set(q_ids)):
            dupes = [qid for qid in q_ids if q_ids.count(qid) > 1]
            raise ValueError(
                f"duplicate question ids in flow {flow.id!r}: {sorted(set(dupes))}"
            )

        for trans in flow.next:
            if trans.flow not in flow_ids:
                raise ValueError(
                    f"flow {flow.id!r} transitions to unknown flow {trans.flow!r}"
                )
