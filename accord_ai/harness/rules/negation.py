"""Negation rule — deterministic patterns that set boolean coverage fields.

Ported from the v3 harness negation section that scored the +0.75 F1 win
on `negation-no-hired-auto` in the 5-scenario sample. The harness prose
taught the LLM these rules, but prose wastes tokens vs. a regex+post-
processing pair that's 10 lines. We keep the deterministic version in
code so it fires regardless of prompt variant and on every turn — not
just when the LLM happens to follow the guidance.

Supported patterns (regex, case-insensitive):

  "no hired auto"          → lob_details.coverage.hired_auto = False
  "no non-owned"           → lob_details.coverage.non_owned_auto = False
  "no hazmat"              → lob_details.hazmat = False
  "no trailer interchange" → lob_details.trailer_interchange = False
  "no driver training"     → lob_details.driver_training = False

Double-negatives are handled at the regex level by refusing to match
``not`` directly preceding the field phrase (``not no hazmat`` won't
fire). The harness document covers a few more double-negative cases
but they're rare in production and easy to add if future evals surface
them.

Only fires when the field is currently None or absent in the delta —
never overrides a value the LLM provided explicitly (e.g. if the LLM
already set hazmat=True, we don't flip to False just because the user
negated somewhere in the turn).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from accord_ai.logging_config import get_logger

_logger = get_logger("harness.rules.negation")


# Pattern → (dotted path, value) map. The pattern's negative lookbehind
# prevents matching "not no hired auto" (a double negative that should
# yield True), keeping this rule narrowly correct. Patterns are
# lowercased — apply_negation_rule normalizes input.
_NEGATION_PATTERNS = [
    # Hired auto — classical CA coverage field
    (
        re.compile(
            r"(?:no|we do(?:n'|\s+no)t\s+(?:have|use|carry))\s+hired\s+(?:auto|autos?|vehicles?)",
            re.IGNORECASE,
        ),
        "lob_details.coverage.hired_auto",
        False,
    ),
    # Non-owned auto
    (
        re.compile(
            r"(?:no|we do(?:n'|\s+no)t\s+(?:have|use|carry))\s+non[-\s]?owned(?:\s+(?:auto|autos?|vehicles?))?",
            re.IGNORECASE,
        ),
        "lob_details.coverage.non_owned_auto",
        False,
    ),
    # Hazmat
    (
        re.compile(
            r"(?:no|we do(?:n'|\s+no)t\s+(?:have|carry|haul|transport))\s+"
            r"(?:hazmat|hazardous|dangerous(?:\s+goods?)?)",
            re.IGNORECASE,
        ),
        "lob_details.hazmat",
        False,
    ),
    # Trailer interchange
    (
        re.compile(
            r"(?:no|we do(?:n'|\s+no)t\s+(?:have|use))\s+trailer\s+interchange",
            re.IGNORECASE,
        ),
        "lob_details.trailer_interchange",
        False,
    ),
    # Driver training
    (
        re.compile(
            r"(?:no|we do(?:n'|\s+no)t\s+(?:have|require|provide))\s+driver\s+training",
            re.IGNORECASE,
        ),
        "lob_details.driver_training",
        False,
    ),
]


def _walk(container: Any, path: str) -> Any:
    """Walk dotted path; return None on miss."""
    cur = container
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _set(container: Dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path in a nested dict, creating intermediates."""
    parts = path.split(".")
    cur = container
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_negation_rule(
    user_text: str, extracted: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply negation patterns to an extraction delta.

    Returns the (possibly-mutated) delta. Idempotent — applying twice
    yields the same result. Does not override fields the LLM already set.
    """
    if os.environ.get("ACCORD_NEGATION_RULE", "1") != "1":
        return extracted
    if not user_text or not isinstance(extracted, dict):
        return extracted

    for pattern, path, value in _NEGATION_PATTERNS:
        if not pattern.search(user_text):
            continue
        existing = _walk(extracted, path)
        # Don't override an LLM value — only fill when absent.
        if existing is None:
            _set(extracted, path, value)
            _logger.info(
                "negation rule fired: path=%s value=%s (from user text)",
                path, value,
            )
    return extracted
