"""ExtractionContext — per-turn flow context passed controller → extractor (Phase 3.3).

Populated by the controller from the prior FlowState (what was just asked).
None / empty values are valid; the extractor renders no context block when
nothing is provided (first turn, no engine wired, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionContext:
    """Per-turn context passed from controller → extractor.

    current_flow:     id of the flow that generated the last question
    expected_fields:  submission paths the last question targeted
    question_text:    exact prompt the user was responding to
    rag_snippets:     RESERVED — Phase 3.4 populates from ChromaDB
    """

    current_flow: str | None = None
    expected_fields: tuple[str, ...] = ()
    question_text: str | None = None
    rag_snippets: tuple[str, ...] = ()  # noqa: RUF012 — frozen dataclass, not mutable

    @property
    def is_empty(self) -> bool:
        return (
            self.current_flow is None
            and not self.expected_fields
            and self.question_text is None
            and not self.rag_snippets
        )


EMPTY_CONTEXT = ExtractionContext()
