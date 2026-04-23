"""10.a — build_intake_app factory + IntakeApp shape."""
from dataclasses import FrozenInstanceError

import pytest

from accord_ai.app import IntakeApp, build_intake_app
from accord_ai.config import Settings
from accord_ai.conversation.controller import ConversationController
from accord_ai.core.store import SessionStore
from accord_ai.llm.fake_engine import FakeEngine
from tests._fixtures import valid_ca_dict


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    return Settings()


# --- IntakeApp shape ---

def test_intake_app_is_frozen(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    with pytest.raises(FrozenInstanceError):
        app.settings = s


def test_build_wires_expected_components(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    assert isinstance(app, IntakeApp)
    assert isinstance(app.store, SessionStore)
    assert isinstance(app.controller, ConversationController)
    assert app.settings is s


# --- Store wired with settings ---

def test_store_uses_settings_db_path(tmp_path, monkeypatch):
    db = tmp_path / "custom.db"
    monkeypatch.setenv("DB_PATH", str(db))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    app = build_intake_app(Settings(), engine=FakeEngine(), refiner_engine=FakeEngine())
    app.store.create_session()
    assert db.exists()


def test_harness_max_refines_threaded_through_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("HARNESS_MAX_REFINES", "3")
    app = build_intake_app(Settings(), engine=FakeEngine(), refiner_engine=FakeEngine())
    assert app.controller._harness._max_refines == 3


# --- End-to-end happy path ---

@pytest.mark.asyncio
async def test_app_end_to_end_single_turn(tmp_path, monkeypatch):
    """Wire + use the full stack: create session, process turn, judge passes."""
    s = _settings(tmp_path, monkeypatch)
    # Extractor returns a fully-valid submission so the expanded judge
    # passes without triggering refinement. Extractor + Responder share
    # the main engine (sequential calls per turn).
    main = FakeEngine([
        valid_ca_dict(),                      # extract
        "Got it — ready to finalize.",        # respond
    ])
    refiner = FakeEngine()   # never called — judge passes on extracted state
    app = build_intake_app(s, engine=main, refiner_engine=refiner)

    sid = app.store.create_session()
    result = await app.controller.process_turn(
        session_id=sid, user_message="we are Acme",
    )
    assert result.verdict.passed is True
    assert result.is_complete is True
    assert result.submission.business_name == "Acme Trucking"
    assert (
        "Acme" in result.assistant_message
        or "ready" in result.assistant_message
    )
    assert (
        app.store.get_session(sid).submission.business_name == "Acme Trucking"
    )


@pytest.mark.asyncio
async def test_app_end_to_end_multi_turn_accumulation(tmp_path, monkeypatch):
    """Turn 1 loads a fully-valid submission; turn 2 overwrites EIN — state
    accumulates through the apply_diff merge semantic."""
    s = _settings(tmp_path, monkeypatch)
    main = FakeEngine([
        valid_ca_dict(),                     # turn 1 extract
        "Got it — what's your EIN?",         # turn 1 respond
        {"ein": "99-9999999"},               # turn 2 extract (EIN override)
        "Perfect.",                          # turn 2 respond
    ])
    app = build_intake_app(s, engine=main, refiner_engine=FakeEngine())

    sid = app.store.create_session()
    await app.controller.process_turn(session_id=sid, user_message="Acme Trucking")
    await app.controller.process_turn(session_id=sid, user_message="EIN 99-9999999")

    final = app.store.get_session(sid).submission
    assert final.business_name == "Acme Trucking"
    assert final.ein == "99-9999999"


# --- Default construction (no overrides) ---

def test_build_without_overrides_constructs_real_stack(tmp_path, monkeypatch):
    """Default: build_engine + build_refiner_engine run; no network call made."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    app = build_intake_app(Settings())
    assert isinstance(app, IntakeApp)


# --- Responder + judge exposed for CLI/API use (10.b.1) ---

def test_build_exposes_responder_and_judge(tmp_path, monkeypatch):
    """Reach-in access: CLI needs these for initial greeting + /status rendering."""
    from accord_ai.conversation.responder import Responder
    from accord_ai.harness.judge import SchemaJudge

    s = _settings(tmp_path, monkeypatch)
    app = build_intake_app(s, engine=FakeEngine(), refiner_engine=FakeEngine())
    assert isinstance(app.responder, Responder)
    assert isinstance(app.judge, SchemaJudge)
    # Same responder instance is wired into the controller (shared)
    assert app.controller._responder is app.responder
