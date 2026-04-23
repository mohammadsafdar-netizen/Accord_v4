"""10.b / 10.b.1 — CLI. FakeEngine injected via build_intake_app; no network.

Interactive tests supply the initial greeting as the first engine response
(responder call before REPL loop), followed by per-turn extract+respond.
"""
from pathlib import Path

import pytest

from accord_ai.app import build_intake_app
from accord_ai.cli.intake import (
    _check_llm_health,
    _load_script_messages,
    list_sessions,
    run_interactive,
    run_scripted,
)
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.schema import CustomerSubmission
from tests._fixtures import valid_ca_dict


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    # CLI tests supply empty refiner FakeEngines; disabling refine prevents
    # a queue-exhausted crash when judge fails on partial/empty state.
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")
    return Settings()


# ============================================================
# run_scripted — batch mode
# ============================================================

@pytest.mark.asyncio
async def test_run_scripted_processes_messages_in_order(tmp_path, monkeypatch):
    """Two turns run in sequence; state accumulates. Turn 1 extracts ein only
    (judge fails on missing business_name) so the loop continues to turn 2."""
    s = _settings(tmp_path, monkeypatch)
    main = FakeEngine([
        {"ein": "12-3456789"},        "What's your business name?",   # turn 1
        {"business_name": "Acme"},    "Got it.",                       # turn 2
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    results = await run_scripted(app, ["EIN 12-3456789", "Acme Trucking"])

    assert len(results) == 2
    assert results[1].submission.business_name == "Acme"
    assert results[1].submission.ein == "12-3456789"


@pytest.mark.asyncio
async def test_run_scripted_stops_early_on_is_complete(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    # First-turn extractor returns a fully-valid submission so the
    # v3-aligned judge sets is_complete=True and the loop exits after
    # the first turn (the second-turn queue entries never run).
    main = FakeEngine([
        valid_ca_dict(),              "Ready to finalize.",
        {"ein": "ignored"},           "ignored",
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    results = await run_scripted(app, ["We are Acme", "follow-up", "more"])

    assert len(results) == 1
    assert results[0].is_complete is True


@pytest.mark.asyncio
async def test_run_scripted_respects_max_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")   # no refiner consumption
    s = _settings(tmp_path, monkeypatch)
    main = FakeEngine([
        {}, "Tell me more?",
        {}, "Still need more?",
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    results = await run_scripted(app, ["m1", "m2", "m3", "m4"], max_turns=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_scripted_persists_state_to_store(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    main = FakeEngine([
        {"business_name": "Persisted Corp"},  "ok",
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    await run_scripted(app, ["we are Persisted Corp"])
    [summary] = app.store.list_sessions()
    loaded = app.store.get_session(summary.session_id)
    assert loaded.submission.business_name == "Persisted Corp"


@pytest.mark.asyncio
async def test_run_scripted_debug_prints_verdict_diagnostics(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")
    s = _settings(tmp_path, monkeypatch)
    main = FakeEngine([{}, "try again"])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    await run_scripted(app, ["hello"], debug=True)
    out = capsys.readouterr().out
    assert "verdict.passed" in out
    assert "failed_paths" in out


# ============================================================
# run_interactive — REPL
# ============================================================

def _queue_inputs(monkeypatch, items):
    it = iter(items)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


@pytest.mark.asyncio
async def test_initial_greeting_shown_before_first_user_input(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(
        s,
        engine=FakeEngine(["Hi there — what's your business name?"]),
        refiner_engine=FakeEngine(),
    )
    _queue_inputs(monkeypatch, ["/quit"])

    await run_interactive(app)
    out = capsys.readouterr().out
    assert "Hi there" in out


@pytest.mark.asyncio
async def test_quit_command_exits_cleanly(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())
    _queue_inputs(monkeypatch, ["/quit"])

    await run_interactive(app)
    out = capsys.readouterr().out
    assert "[quit]" in out


@pytest.mark.asyncio
async def test_help_command_lists_all_commands(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())
    _queue_inputs(monkeypatch, ["/help", "/quit"])

    await run_interactive(app)
    out = capsys.readouterr().out
    assert "/status" in out and "/finalize" in out and "/help" in out


@pytest.mark.asyncio
async def test_status_command_renders_current_submission_via_explainer(
    tmp_path, monkeypatch, capsys
):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())
    sid = app.store.create_session()
    app.store.update_submission(sid, CustomerSubmission(business_name="Acme"))
    _queue_inputs(monkeypatch, ["/status", "/quit"])

    await run_interactive(app, session_id=sid)
    out = capsys.readouterr().out
    assert "Business: Acme" in out


@pytest.mark.asyncio
async def test_finalize_command_finalizes_and_exits(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())
    sid = app.store.create_session()
    _queue_inputs(monkeypatch, ["/finalize"])

    await run_interactive(app, session_id=sid)
    assert app.store.get_session(sid).status == "finalized"


@pytest.mark.asyncio
async def test_process_turn_runs_and_greets_back(tmp_path, monkeypatch, capsys):
    """Empty extract keeps judge failing, so the REPL doesn't hit the finalize
    prompt. /quit exits cleanly after seeing the assistant response."""
    s = _settings(tmp_path, monkeypatch)
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")
    main = FakeEngine([
        "greeting",                                 # initial
        {},                                         # extract (empty → judge still fails)
        "Got it — what's your EIN?",                 # respond
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())
    _queue_inputs(monkeypatch, ["we are Acme", "/quit"])

    await run_interactive(app)
    out = capsys.readouterr().out
    assert "what's your ein" in out.lower()


@pytest.mark.asyncio
async def test_completion_finalize_prompt_y_finalizes(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    # Extractor returns a fully-valid submission so the judge signals
    # is_complete — the CLI then prompts "finalize? [y/n]" and our
    # queued "y" triggers store.finalize_session.
    main = FakeEngine([
        "greeting",                    # initial
        valid_ca_dict(),               # extract (passes judge)
        "Ready to finalize.",          # respond
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())
    _queue_inputs(monkeypatch, ["we are Acme", "y"])

    await run_interactive(app)
    [summary] = app.store.list_sessions()
    assert summary.status == "finalized"


# --- Session resume ---

@pytest.mark.asyncio
async def test_resume_existing_session(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())
    sid = app.store.create_session()
    app.store.update_submission(sid, CustomerSubmission(business_name="PreExisting"))
    _queue_inputs(monkeypatch, ["/quit"])

    await run_interactive(app, session_id=sid)
    out = capsys.readouterr().out
    assert "resumed" in out
    assert sid in out


@pytest.mark.asyncio
async def test_resume_missing_session_prints_error(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())

    await run_interactive(app, session_id="no-such-id")
    err = capsys.readouterr().err
    assert "not found" in err


@pytest.mark.asyncio
async def test_resume_finalized_session_is_refused(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    sid = app.store.create_session()
    app.store.finalize_session(sid)

    await run_interactive(app, session_id=sid)
    err = capsys.readouterr().err
    assert "can't resume" in err


# --- Graceful engine error recovery ---

@pytest.mark.asyncio
async def test_engine_error_mid_turn_keeps_session_alive(
    tmp_path, monkeypatch, capsys,
):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(["greeting"]),
                           refiner_engine=FakeEngine())

    call_count = [0]
    original = app.controller.process_turn

    async def flaky(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated provider down")
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.controller, "process_turn", flaky)
    _queue_inputs(monkeypatch, ["first turn", "/quit"])

    await run_interactive(app)
    out = capsys.readouterr().out
    assert "unexpected error" in out or "engine error" in out
    assert "still active" in out


# ============================================================
# list_sessions
# ============================================================

def test_list_sessions_shows_all_when_tenant_none(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    sid_a = app.store.create_session(tenant="acme")
    sid_g = app.store.create_session(tenant="globex")

    list_sessions(app)
    out = capsys.readouterr().out
    assert sid_a in out
    assert sid_g in out
    assert "acme" in out
    assert "globex" in out


def test_list_sessions_tenant_filter(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    sid_a = app.store.create_session(tenant="acme")
    sid_g = app.store.create_session(tenant="globex")

    list_sessions(app, tenant="acme")
    out = capsys.readouterr().out
    assert sid_a in out
    assert sid_g not in out


def test_list_sessions_empty_prints_sentinel(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())

    list_sessions(app)
    out = capsys.readouterr().out
    assert "[no sessions]" in out


# ============================================================
# Health check
# ============================================================

@pytest.mark.asyncio
async def test_health_check_detects_unreachable_endpoint(monkeypatch):
    """Point at a port nothing's listening on — must return an error detail."""
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1")
    s = Settings()
    detail = await _check_llm_health(s, timeout_s=0.5)
    assert detail is not None
    assert len(detail) > 0


# ============================================================
# Script loader
# ============================================================

def test_load_script_messages_strips_blanks_and_comments(tmp_path):
    f = tmp_path / "script.txt"
    f.write_text(
        "# a comment\n"
        "\n"
        "first message\n"
        "# another comment\n"
        "  \n"
        "second message  \n"
    )
    assert _load_script_messages(f) == ["first message", "second message"]
