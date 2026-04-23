"""10.d — full-stack integration. Real LLM, real store, real retriever.

Gated by ACCORD_LLM_INTEGRATION=1. Requires vLLM serving
Qwen/Qwen3.5-9B at LLM_BASE_URL (default http://localhost:8000/v1).

Run:
    ACCORD_LLM_INTEGRATION=1 python -m pytest tests/test_intake_integration.py -v

Assertions are deliberately tolerant — we're validating that the stack
is wired correctly (session persists, messages stored, state accumulates)
not that the LLM extracts with specific accuracy. Extraction accuracy is
an evaluation concern (future phase).
"""
import os

import pytest

from accord_ai.app import build_intake_app
from accord_ai.cli.intake import _check_llm_health, run_scripted
from accord_ai.config import Settings


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ACCORD_LLM_INTEGRATION"),
        reason="set ACCORD_LLM_INTEGRATION=1 to run",
    ),
]


@pytest.mark.asyncio
async def test_health_check_succeeds_against_live_vllm():
    """The CLI's startup health check resolves cleanly when vLLM is running."""
    detail = await _check_llm_health(Settings(), timeout_s=10.0)
    assert detail is None, f"vLLM not reachable: {detail}"


@pytest.mark.asyncio
async def test_real_intake_stack_end_to_end(tmp_path, monkeypatch):
    """Multi-turn intake against the real LLM.

    Wire-up validation, not extraction accuracy:
      - build_intake_app constructs the real stack
      - run_scripted processes multiple turns
      - state persists to SQLite
      - messages (user + assistant) persist
      - no exceptions propagate from the stack
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "intake.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))

    app = build_intake_app(Settings())

    results = await run_scripted(
        app,
        messages=[
            "We're Acme Trucking Corporation, a commercial auto business",
            "Our EIN is 12-3456789 and email is ops@acme-trucking.com",
            "Policy should run from May 1 2026 through May 1 2027",
        ],
        max_turns=5,
        print_explainer=False,
    )

    # --- Turn-level sanity ---
    assert len(results) >= 1
    for r in results:
        assert isinstance(r.assistant_message, str)
        assert len(r.assistant_message) > 0, "assistant returned empty text"

    # --- Session persisted ---
    summaries = app.store.list_sessions()
    assert len(summaries) == 1
    loaded = app.store.get_session(summaries[0].session_id)
    assert loaded is not None
    # Status stays active until someone calls finalize
    assert loaded.status == "active"

    # --- At least one message pair persisted per processed turn ---
    msgs = app.store.get_messages(summaries[0].session_id)
    assert len(msgs) == 2 * len(results)
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"

    # --- Extraction likely produced something from "Acme Trucking" mentions ---
    # Tolerant: Qwen usually captures business_name over 2-3 turns, but LLMs
    # vary. If it didn't extract anything, the stack still worked; the failure
    # would be in Phase 7's prompt engineering, out of scope for wire-up.
    final = loaded.submission
    if final.business_name is not None:
        assert "acme" in final.business_name.lower()
