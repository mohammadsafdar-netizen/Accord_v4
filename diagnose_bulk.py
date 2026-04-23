"""Diagnose why bulk-* scenarios extract zero fields.

Runs one bulk scenario end-to-end against live vLLM, logging every
engine call's raw output + token counts so we can distinguish:
  * truncation      — completion_tokens hits max_tokens exactly
  * LLM refusal     — valid JSON but an empty {} object
  * xgrammar fail   — malformed output → ExtractionOutputError
"""
from __future__ import annotations

import asyncio
import json
import pathlib

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import load_scenarios
from accord_ai.llm.openai_engine import OpenAIEngine
from accord_ai.llm.prompts import extraction as ex, render
from accord_ai.schema import CustomerSubmission


async def diagnose(scenario_id: str) -> None:
    scs = load_scenarios(pathlib.Path("eval/scenarios"))
    target = next((s for s in scs if s.id == scenario_id), None)
    if target is None:
        print(f"scenario not found: {scenario_id}")
        return
    print(f"=== {target.id} ===")
    print(f"  turns: {len(target.turns)}")
    for i, t in enumerate(target.turns):
        print(f"  turn[{i}] length={len(t)} chars")
    print()

    # Build the extractor's system + user exactly as production would,
    # call the engine directly (no harness / controller), and dump the
    # raw output. We want to see what the LLM produced BEFORE any
    # parse/validate step rejected it.
    settings = Settings(llm_timeout_s=120.0, llm_retries=1)
    engine = OpenAIEngine(settings)
    schema = CustomerSubmission.model_json_schema()

    # Simulate turn 0 (the one with the big message).
    msg = target.turns[0]
    current = CustomerSubmission()
    user_content = render(
        ex.USER_TEMPLATE_V1,
        schema=json.dumps(schema),
        current_state=current.model_dump_json(exclude_none=True),
        user_message=msg,
    )
    messages = [
        {"role": "system", "content": ex.SYSTEM_V2},
        {"role": "user", "content": user_content},
    ]

    # Fire the call with the SAME settings production uses.
    print("calling extractor engine (guided_json, max_tokens=2048)...")
    response = await engine.generate(
        messages, max_tokens=2048, json_schema=schema,
    )
    print(f"  tokens_in:  {response.tokens_in}")
    print(f"  tokens_out: {response.tokens_out}")
    print(f"  max_tokens: 2048  →  hit_limit: {response.tokens_out >= 2040}")
    print(f"  latency_ms: {response.latency_ms:.0f}")
    print(f"  text_len:   {len(response.text)}")
    print()
    print("=== raw LLM output ===")
    print(response.text[:3000])
    if len(response.text) > 3000:
        print(f"... (truncated for display; {len(response.text)} total chars)")
    print("=== end ===")
    print()

    # Try to parse the output as production would.
    try:
        parsed = CustomerSubmission.model_validate_json(response.text)
        set_fields = [
            k for k in parsed.model_fields_set
            if getattr(parsed, k) not in (None, [], {})
        ]
        print(f"PARSE: OK — fields set: {len(set_fields)}")
        print(f"  keys: {set_fields}")
    except json.JSONDecodeError as e:
        print(f"PARSE: JSONDecodeError at pos {e.pos}: {e.msg}")
        # Show the area around the failure
        start = max(0, e.pos - 80)
        end = min(len(response.text), e.pos + 80)
        print(f"  context: ...{response.text[start:end]!r}...")
    except Exception as e:
        print(f"PARSE: {type(e).__name__}: {str(e)[:800]}")


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "bulk-all-business-info"
    asyncio.run(diagnose(sid))
