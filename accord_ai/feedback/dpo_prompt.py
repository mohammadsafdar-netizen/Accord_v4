"""DPO prompt template for correction-driven training pairs (Phase 2.3).

The template is intentionally narrow: one field, one user message, one
JSON output.  DPO learns the value delta between chosen/rejected; keeping
the prompt field-focused prevents the trainer from chasing unrelated
preamble tokens.

DPO_TEMPLATE_VERSION is stored in every exported JSONL record so that
Phase 4.5 can re-filter or re-weight pairs when the template changes.
"""
from __future__ import annotations

DPO_TEMPLATE_VERSION = "v1"

DPO_PROMPT_TEMPLATE = """\
You are an insurance intake extraction assistant.

Extract the following field from the user's message into JSON matching the schema fragment.

FIELD: {field_path}
SCHEMA: {schema_json}
USER MESSAGE: {user_text}

Output JSON only, no commentary.\
"""


def render_prompt(*, field_path: str, schema_json: str, user_text: str) -> str:
    return DPO_PROMPT_TEMPLATE.format(
        field_path=field_path,
        schema_json=schema_json,
        user_text=user_text,
    )
