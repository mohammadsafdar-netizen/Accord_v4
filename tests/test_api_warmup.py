"""Boot-warmup tests (P10.0.h.2).

Warmup primes vLLM's xgrammar guided-JSON compilation and prefix cache
on API startup so the first real user turn doesn't pay a 5-10 s cold
penalty. Tests verify:
  * Default (WARMUP_ON_BOOT unset) → warmup does not run. FakeEngine
    queues must not be consumed by the app just because the server
    started.
  * Explicit WARMUP_ON_BOOT=true → warmup fires one extractor + one
    responder call against a dummy submission. Observed via FakeEngine
    call history.
  * Warmup failures (engine raises) don't take down the server — the
    lifespan catches and logs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _build(tmp_path, monkeypatch, *, main, refiner=None, warmup=False):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "warmup.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    if warmup:
        monkeypatch.setenv("WARMUP_ON_BOOT", "true")
    else:
        monkeypatch.delenv("WARMUP_ON_BOOT", raising=False)
    settings = Settings()
    intake = build_intake_app(
        settings,
        engine=main,
        refiner_engine=refiner or FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake), intake


def test_warmup_disabled_by_default_no_calls(tmp_path, monkeypatch):
    main = FakeEngine([])   # empty queue — would raise if warmup fired
    app, intake = _build(tmp_path, monkeypatch, main=main, warmup=False)
    # Enter + exit the lifespan via TestClient. No warmup call should happen.
    with TestClient(app):
        pass
    assert intake.controller._extractor._engine.history == []
    assert intake.responder._engine.history == []


def test_warmup_enabled_fires_extract_and_respond(tmp_path, monkeypatch):
    # Two responses — one for the extractor (a dummy valid-ish JSON so
    # ExtractionOutputError doesn't bubble), one for the responder
    # (plain text). The shared engine is called twice in that order.
    main = FakeEngine([
        {"business_name": "__warmup__"},   # extractor
        "warmup-ok",                        # responder
    ])
    app, intake = _build(tmp_path, monkeypatch, main=main, warmup=True)
    with TestClient(app):
        pass   # startup + warmup + shutdown

    # Two engine calls: one guided-JSON (extractor), one free-form (responder).
    hist = intake.controller._extractor._engine.history
    assert len(hist) == 2
    # First call: extractor — schema attached.
    assert hist[0].json_schema is not None
    assert "business_name" in hist[0].json_schema.get("properties", {})
    # Second call: responder — no schema.
    assert hist[1].json_schema is None


def test_warmup_failure_does_not_prevent_server_from_serving(
    tmp_path, monkeypatch,
):
    """Warmup errors (engine down at boot) must not kill the app."""
    main = FakeEngine([])   # empty queue → first warmup call raises
    app, _ = _build(tmp_path, monkeypatch, main=main, warmup=True)
    with TestClient(app) as client:
        r = client.get("/health")
    # Health endpoint still reachable despite the warmup error.
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
