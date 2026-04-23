"""Rate-limit tests (P10.0.h.3).

slowapi per-IP caps the public surface to prevent unbounded LLM-cost
exposure. These tests verify:
  * Default (RATE_LIMIT_ENABLED unset) → unlimited; no 429 regardless
    of request rate.
  * Explicit opt-in → the configured per-endpoint cap is enforced and
    the Nth+1 request yields a 429 with Retry-After.
  * X-Forwarded-For shapes the per-IP key, so real clients behind
    ngrok/nginx are limited rather than the proxy's single IP.
  * Health endpoint is uncapped (load balancers hammer it — never 429).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _build(tmp_path, monkeypatch, *, enabled, per_minute=5, main=None):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rl.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    monkeypatch.delenv("WARMUP_ON_BOOT", raising=False)
    # Refiner disabled — these tests probe rate-limit behavior, not the
    # judge→refine cycle. Otherwise the default judge (which fails on an
    # empty submission) would trigger refiner calls and exhaust the
    # FakeEngine queue before we hit the rate-limit cap.
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")
    if enabled:
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_START_SESSION_PER_MINUTE", str(per_minute))
        monkeypatch.setenv("RATE_LIMIT_ANSWER_PER_MINUTE", str(per_minute))
        monkeypatch.setenv("RATE_LIMIT_COMPLETE_PER_MINUTE", str(per_minute))
    else:
        monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    settings = Settings()
    intake = build_intake_app(
        settings,
        engine=main or FakeEngine(["greeting"] * 200),
        refiner_engine=FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake)


def test_disabled_by_default_no_429(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch, enabled=False)
    with TestClient(app) as client:
        # Fire well above any default cap — nothing should 429.
        for _ in range(100):
            r = client.post("/start-session")
            assert r.status_code == 200


def test_start_session_limit_enforced(tmp_path, monkeypatch):
    app = _build(tmp_path, monkeypatch, enabled=True, per_minute=5)
    with TestClient(app) as client:
        for i in range(5):
            r = client.post("/start-session")
            assert r.status_code == 200, f"request {i} should be allowed"
        r = client.post("/start-session")
        assert r.status_code == 429
        body = r.json()
        # slowapi's default body is {"error": "Rate limit exceeded: ..."}
        assert "error" in body or "detail" in body


def test_answer_limit_enforced(tmp_path, monkeypatch):
    """/answer also rate-limited. Uses a session already opened so each
    /answer is independent of session creation."""
    main = FakeEngine([
        "greeting",
        # 5 turn-pairs: extract (empty diff), respond. Loop 5x.
        *([{}, "resp"] * 5),
    ])
    app = _build(tmp_path, monkeypatch, enabled=True, per_minute=3, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        for i in range(3):
            r = client.post(
                "/answer",
                json={"session_id": sid, "message": f"turn-{i}"},
            )
            assert r.status_code == 200, f"/answer {i} should be allowed"
        r = client.post("/answer", json={"session_id": sid, "message": "4"})
        assert r.status_code == 429


def test_health_is_not_rate_limited(tmp_path, monkeypatch):
    """Load balancers + uptime probes hammer /health — never 429."""
    app = _build(tmp_path, monkeypatch, enabled=True, per_minute=2)
    with TestClient(app) as client:
        for _ in range(50):
            r = client.get("/health")
            assert r.status_code == 200


def test_x_forwarded_for_separates_per_ip_buckets(tmp_path, monkeypatch):
    """Two distinct XFF values should get independent counters — the
    proxy can deliver traffic for many clients without one tripping
    another's cap. Both send the same number of requests; neither hits
    the cap since they're keyed to different IPs."""
    app = _build(tmp_path, monkeypatch, enabled=True, per_minute=3)
    with TestClient(app) as client:
        # Client A fires 3 — all allowed.
        for i in range(3):
            r = client.post(
                "/start-session",
                headers={"x-forwarded-for": "10.0.0.1"},
            )
            assert r.status_code == 200
        # Client B fires 3 — independent counter.
        for i in range(3):
            r = client.post(
                "/start-session",
                headers={"x-forwarded-for": "10.0.0.2"},
            )
            assert r.status_code == 200


def test_429_response_carries_retry_after(tmp_path, monkeypatch):
    """On a rate-limit hit, clients need Retry-After so they can back
    off without hammering. slowapi's default exception handler
    populates it; we pin the behavior here so a future refactor (e.g.
    custom handler) doesn't silently drop it."""
    app = _build(tmp_path, monkeypatch, enabled=True, per_minute=2)
    with TestClient(app) as client:
        client.post("/start-session")
        client.post("/start-session")
        r = client.post("/start-session")
        assert r.status_code == 429
        headers_lower = {k.lower() for k in r.headers.keys()}
        assert "retry-after" in headers_lower, (
            f"429 should carry Retry-After; got {sorted(headers_lower)}"
        )
