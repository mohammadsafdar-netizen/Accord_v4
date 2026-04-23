"""Deterministic extraction rules (pure Python, no LLM).

Each rule is a pure function ``(user_text, delta) -> delta`` that applies
a narrow, proven extraction pattern the LLM gets wrong often enough to be
worth encoding in code. The orchestrator calls each enabled rule after
the LLM extraction and before pydantic validation — so output already
survives JSON parse and the 5-step postprocess cleanup.

Env flags:
  * ``ACCORD_NEGATION_RULE=1`` (default on) — apply the negation rule.

Rules are additive to the LLM output: they fill in fields the LLM missed
or set, not override the LLM's value if it's present. When a rule
materially changes a field, an INFO-level log line documents the change.
"""
from accord_ai.harness.rules.negation import apply_negation_rule

__all__ = ["apply_negation_rule"]
