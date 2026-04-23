"""Strict template renderer.

Refuses to format unless the set of kwargs exactly matches the set of
placeholders in the template. Catches drift between prompt files and
callers — a renamed placeholder or forgotten kwarg fails loudly at
render time, not silently in the LLM's output.
"""
from __future__ import annotations

from string import Formatter


def render(template: str, /, **kwargs: str) -> str:
    """Format `template` with `kwargs`.

    Raises ValueError if:
      - a placeholder in the template has no matching kwarg (missing)
      - a kwarg has no matching placeholder (unexpected)

    `template` is positional-only so a placeholder named `template` is legal.
    """
    placeholders = {
        name for _, name, _, _ in Formatter().parse(template)
        if name is not None and name != ""   # ignore plain {} if any
    }
    provided = set(kwargs)
    missing = placeholders - provided
    unexpected = provided - placeholders
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if unexpected:
            parts.append(f"unexpected={sorted(unexpected)}")
        raise ValueError(f"template/kwargs mismatch — {', '.join(parts)}")
    return template.format(**kwargs)
