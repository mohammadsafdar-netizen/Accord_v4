"""11.a — FastAPI skeleton + /health + request-context middleware.

Tests use FastAPI's TestClient; IntakeApp is built with FakeEngines so no
real LLM traffic happens during tests.
"""
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine
from tests._fixtures import valid_ca_dict


def _fastapi_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    settings = Settings()
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


# --- Construction ---

def test_build_fastapi_app_returns_fastapi_instance(tmp_path, monkeypatch):
    app = _fastapi_app(tmp_path, monkeypatch)
    assert isinstance(app, FastAPI)


def test_fastapi_app_holds_intake_on_state(tmp_path, monkeypatch):
    app = _fastapi_app(tmp_path, monkeypatch)
    assert app.state.intake is not None
    assert app.state.settings is not None


# --- /health ---

def test_health_endpoint_returns_ok(tmp_path, monkeypatch):
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "llm_base_url" in body


def test_health_endpoint_exposes_llm_config_summary(tmp_path, monkeypatch):
    """Operators need to see which LLM endpoint + model the API is wired to."""
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:9000/v1")
    monkeypatch.setenv("LLM_MODEL", "some/custom-model")
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health")
    body = r.json()
    assert body["llm_base_url"] == "http://vllm:9000/v1"
    assert body["llm_model"] == "some/custom-model"


# --- Request-context middleware ---

def test_response_includes_request_id_header(tmp_path, monkeypatch):
    """Every response gets an X-Request-Id for client correlation."""
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) == 12   # new_request_id() produces 12 hex


def test_request_id_header_is_honored_when_client_supplies_one(tmp_path, monkeypatch):
    """Clients can set X-Request-Id for distributed tracing; API echoes it back."""
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health", headers={"x-request-id": "my-trace-abc123"})
    assert r.headers["x-request-id"] == "my-trace-abc123"


def test_tenant_header_is_echoed_back(tmp_path, monkeypatch):
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health", headers={"x-tenant-slug": "acme"})
    assert r.headers.get("x-tenant-slug") == "acme"


def test_tenant_header_absent_when_not_provided(tmp_path, monkeypatch):
    """Don't invent a tenant header if the client didn't send one."""
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health")
    assert "x-tenant-slug" not in r.headers


def test_context_is_cleared_between_requests(tmp_path, monkeypatch):
    """Second request must not see the first request's tenant in context."""
    from accord_ai.request_context import get_tenant

    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        client.get("/health", headers={"x-tenant-slug": "acme"})
        # After the request, middleware cleared the context
        client.get("/health")
    # In our test's asyncio task, context should be clear
    assert get_tenant() is None


def test_each_request_gets_a_unique_request_id(tmp_path, monkeypatch):
    """Auto-generated request IDs should be distinct per request."""
    app = _fastapi_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r1 = client.get("/health")
        r2 = client.get("/health")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# ============================================================
# 11.b — turn lifecycle endpoints
# ============================================================

from accord_ai.core.store import ConcurrencyError
from accord_ai.schema import CustomerSubmission


def _fastapi_with_engines(tmp_path, monkeypatch, *, main=None, refiner=None):
    """Build FastAPI + TestClient with injectable FakeEngines.

    Auth is disabled here — these tests exercise endpoint behavior, not
    auth. The auth-specific tests use _fastapi_with_auth() instead.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    # Warmup is a production-only concern (it makes real LLM calls on
    # startup). Tests use FakeEngines with empty queues, so warmup
    # would fail the first queue pop — keep it off unless a test
    # specifically opts in.
    monkeypatch.delenv("WARMUP_ON_BOOT", raising=False)
    settings = Settings()
    intake = build_intake_app(
        settings,
        engine=main or FakeEngine(),
        refiner_engine=refiner or FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake), intake


# --- /start-session ---

def test_start_session_returns_id_and_greeting(tmp_path, monkeypatch):
    main = FakeEngine(["Hi there — what's your business name?"])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        r = client.post("/start-session")
    assert r.status_code == 200
    body = r.json()
    # v3 wire contract: /start-session returns at minimum {submission_id,
    # question}. Extra additive fields are permitted (FE ignores unknowns),
    # matching the additive pattern used for /complete (g.3).
    assert isinstance(body["submission_id"], str)
    assert len(body["submission_id"]) == 32
    assert body["question"] == "Hi there — what's your business name?"
    assert {"submission_id", "question"} <= set(body.keys())


def test_start_session_persists_to_store(tmp_path, monkeypatch):
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        body = client.post("/start-session").json()
    assert intake.store.get_session(body["submission_id"]) is not None


def test_start_session_records_tenant_from_header(tmp_path, monkeypatch):
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        body = client.post("/start-session", headers={"x-tenant-slug": "acme"}).json()
    session = intake.store.get_session(body["submission_id"], tenant="acme")
    assert session is not None
    assert session.tenant == "acme"


# --- /answer ---

def test_answer_processes_turn(tmp_path, monkeypatch):
    """Extract → apply → judge passes → respond.

    Extractor returns a fully-valid submission so the v3-aligned judge
    passes without triggering refinement (is_complete=True).
    """
    main = FakeEngine([
        "greeting",                            # start-session
        valid_ca_dict(),                       # extract → passes judge
        "Got it — ready to finalize.",         # respond
    ])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post("/answer", json={"session_id": sid, "message": "we are Acme"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["submission"]["business_name"] == "Acme Trucking"
    assert body["verdict"]["passed"] is True
    assert body["verdict"]["failed_paths"] == []
    assert "Got it" in body["assistant_message"]
    assert body["is_complete"] is True


def test_answer_missing_session_returns_404(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        r = client.post("/answer", json={"session_id": "no-such-id", "message": "hi"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_answer_wrong_tenant_returns_404(tmp_path, monkeypatch):
    """Tenant-leak-safe: wrong-tenant looks like missing, no existence leak."""
    main = FakeEngine(["greeting"])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session", headers={"x-tenant-slug": "acme"}).json()["submission_id"]
        r = client.post(
            "/answer",
            json={"session_id": sid, "message": "hi"},
            headers={"x-tenant-slug": "globex"},
        )
    assert r.status_code == 404


def test_answer_finalized_session_returns_404(tmp_path, monkeypatch):
    """Finalized sessions reject new turns — KeyError → 404 (uniform surface)."""
    main = FakeEngine(["greeting"])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        intake.store.finalize_session(sid)
        r = client.post("/answer", json={"session_id": sid, "message": "late"})
    assert r.status_code == 404


def test_answer_concurrency_error_returns_409(tmp_path, monkeypatch):
    """Refinement-time concurrent modification → ConcurrencyError → 409."""
    main = FakeEngine([
        "greeting",
        # bad dates — triggers refinement
        {"policy_dates": {"effective_date": "2027-05-01",
                          "expiration_date": "2026-05-01"}},
    ])
    refiner = FakeEngine([{
        "business_name": "Acme",
        "policy_dates": {"effective_date": "2026-05-01",
                         "expiration_date": "2027-05-01"},
    }])
    monkeypatch.setenv("HARNESS_MAX_REFINES", "1")
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=main, refiner=refiner,
    )

    # Stub update_submission to raise ConcurrencyError when invoked with
    # expected_updated_at (refinement path).
    original = intake.store.update_submission

    def racing(sid, sub, *, tenant=None, expected_updated_at=None):
        if expected_updated_at is not None:
            raise ConcurrencyError("simulated concurrent write")
        return original(sid, sub, tenant=tenant)

    intake.store.update_submission = racing  # type: ignore[assignment]

    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post("/answer", json={"session_id": sid, "message": "bad dates"})
    assert r.status_code == 409
    assert "concurrent" in r.json()["detail"].lower()


# --- /finalize ---

def test_finalize_transitions_session(tmp_path, monkeypatch):
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post("/finalize", json={"session_id": sid})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["status"] == "finalized"
    assert intake.store.get_session(sid).status == "finalized"


def test_finalize_missing_session_returns_404(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(),
    )
    with TestClient(app) as client:
        r = client.post("/finalize", json={"session_id": "no-such-id"})
    assert r.status_code == 404


def test_finalize_idempotent_on_already_finalized(tmp_path, monkeypatch):
    """store.finalize_session is idempotent — API reflects that."""
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r1 = client.post("/finalize", json={"session_id": sid})
        r2 = client.post("/finalize", json={"session_id": sid})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert intake.store.get_session(sid).status == "finalized"


# --- Request schema validation ---

def test_answer_missing_required_field_returns_422(tmp_path, monkeypatch):
    """Pydantic validation — FastAPI auto-returns 422 on malformed body."""
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        r = client.post("/answer", json={"message": "missing session_id"})
    assert r.status_code == 422


# --- End-to-end multi-turn via HTTP ---

def test_multi_turn_via_http_accumulates_state(tmp_path, monkeypatch):
    main = FakeEngine([
        "greeting",
        valid_ca_dict(),          "What's your EIN?",       # turn 1
        {"ein": "99-9999999"},    "Perfect.",                # turn 2
    ])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        client.post("/answer", json={"session_id": sid, "message": "Acme Trucking"})
        client.post("/answer", json={"session_id": sid, "message": "EIN 99-9999999"})

    final = intake.store.get_session(sid).submission
    assert final.business_name == "Acme Trucking"
    assert final.ein == "99-9999999"


# ============================================================
# 11.c — read endpoints
# ============================================================

# --- GET /session/{id} ---

def test_get_session_detail_returns_full_state(tmp_path, monkeypatch):
    main = FakeEngine(["greeting"])
    app, intake = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        intake.store.update_submission(sid, CustomerSubmission(business_name="Acme"))
        r = client.get(f"/session/{sid}")

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["status"] == "active"
    assert body["submission"]["business_name"] == "Acme"
    assert "created_at" in body and "updated_at" in body


def test_get_session_detail_missing_returns_404(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=FakeEngine())
    with TestClient(app) as client:
        r = client.get("/session/no-such-id")
    assert r.status_code == 404


def test_get_session_detail_wrong_tenant_returns_404(tmp_path, monkeypatch):
    app, intake = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        sid = client.post(
            "/start-session", headers={"x-tenant-slug": "acme"},
        ).json()["submission_id"]
        r = client.get(f"/session/{sid}", headers={"x-tenant-slug": "globex"})
    assert r.status_code == 404


# --- GET /session/{id}/messages ---

def test_get_messages_returns_ordered_list(tmp_path, monkeypatch):
    main = FakeEngine([
        "greeting",
        valid_ca_dict(),            # extract (passes judge)
        "response 1",                # respond
    ])
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        client.post("/answer", json={"session_id": sid, "message": "hi"})
        r = client.get(f"/session/{sid}/messages")

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    msgs = body["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["role"] == "assistant"


def test_get_messages_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_MAX_REFINES", "0")
    main = FakeEngine([
        "greeting",
        {}, "r1",
        {}, "r2",
        {}, "r3",
    ])
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=main)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        for msg in ("m1", "m2", "m3"):
            client.post("/answer", json={"session_id": sid, "message": msg})
        r = client.get(f"/session/{sid}/messages?limit=2")

    assert len(r.json()["messages"]) == 2


def test_get_messages_missing_session_returns_404(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=FakeEngine())
    with TestClient(app) as client:
        r = client.get("/session/no-such-id/messages")
    assert r.status_code == 404


def test_get_messages_wrong_tenant_returns_404(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["greeting"]),
    )
    with TestClient(app) as client:
        sid = client.post(
            "/start-session", headers={"x-tenant-slug": "acme"},
        ).json()["submission_id"]
        r = client.get(
            f"/session/{sid}/messages",
            headers={"x-tenant-slug": "globex"},
        )
    assert r.status_code == 404


# --- GET /sessions ---

def test_list_sessions_returns_summaries(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["g1", "g2"]),
    )
    with TestClient(app) as client:
        client.post("/start-session")
        client.post("/start-session")
        r = client.get("/sessions")

    assert r.status_code == 200
    body = r.json()
    assert len(body["sessions"]) == 2
    assert "submission" not in body["sessions"][0]
    assert body["sessions"][0]["status"] == "active"


def test_list_sessions_filters_by_tenant(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["g1", "g2", "g3"]),
    )
    with TestClient(app) as client:
        client.post("/start-session", headers={"x-tenant-slug": "acme"})
        client.post("/start-session", headers={"x-tenant-slug": "acme"})
        client.post("/start-session", headers={"x-tenant-slug": "globex"})

        acme_only = client.get("/sessions", headers={"x-tenant-slug": "acme"}).json()
        admin_all = client.get("/sessions").json()

    assert len(acme_only["sessions"]) == 2
    assert len(admin_all["sessions"]) == 3


def test_list_sessions_filters_by_status(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["g1", "g2"]),
    )
    with TestClient(app) as client:
        sid1 = client.post("/start-session").json()["submission_id"]
        client.post("/start-session")
        client.post("/finalize", json={"session_id": sid1})

        active = client.get("/sessions?status=active").json()
        final = client.get("/sessions?status=finalized").json()

    assert len(active["sessions"]) == 1
    assert len(final["sessions"]) == 1
    assert final["sessions"][0]["session_id"] == sid1


def test_list_sessions_invalid_status_returns_422(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=FakeEngine())
    with TestClient(app) as client:
        r = client.get("/sessions?status=bogus")
    assert r.status_code == 422


def test_list_sessions_respects_limit(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(
        tmp_path, monkeypatch, main=FakeEngine(["g1", "g2", "g3", "g4", "g5"]),
    )
    with TestClient(app) as client:
        for _ in range(5):
            client.post("/start-session")
        r = client.get("/sessions?limit=2").json()
    assert len(r["sessions"]) == 2


def test_list_sessions_empty_returns_empty_array(tmp_path, monkeypatch):
    app, _ = _fastapi_with_engines(tmp_path, monkeypatch, main=FakeEngine())
    with TestClient(app) as client:
        r = client.get("/sessions")
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


# ============================================================
# 11.d — auth middleware + CORS
# ============================================================


def _fastapi_with_auth(
    tmp_path, monkeypatch, *,
    api_key: Optional[str] = "secret-key-xyz",
    auth_disabled: bool = False,
    chat_open: bool = False,
    allowed_origins: str = "*",
    allowed_origin_regex: Optional[str] = None,
):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    if api_key is not None:
        monkeypatch.setenv("INTAKE_API_KEY", api_key)
    else:
        monkeypatch.delenv("INTAKE_API_KEY", raising=False)
    if auth_disabled:
        monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    if chat_open:
        monkeypatch.setenv("ACCORD_CHAT_OPEN", "true")
    if allowed_origins != "*":
        monkeypatch.setenv("ALLOWED_ORIGINS", allowed_origins)
    if allowed_origin_regex is not None:
        monkeypatch.setenv("ALLOWED_ORIGIN_REGEX", allowed_origin_regex)

    settings = Settings()
    intake = build_intake_app(
        settings,
        engine=FakeEngine(["greeting", "greeting", "greeting"]),
        refiner_engine=FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake), intake


# --- /health always open ---

def test_health_accessible_without_api_key(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200


def test_health_accessible_when_server_has_no_key_configured(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key=None)
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200


# --- Protected endpoints require API key ---

def test_protected_endpoint_without_key_returns_401(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/start-session")
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]


def test_protected_endpoint_with_correct_key_succeeds(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key="secret-key-xyz")
    with TestClient(app) as client:
        r = client.post("/start-session", headers={"x-api-key": "secret-key-xyz"})
    assert r.status_code == 200


def test_protected_endpoint_with_wrong_key_returns_401(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key="secret-key-xyz")
    with TestClient(app) as client:
        r = client.post("/start-session", headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_missing_intake_api_key_returns_500_on_protected(tmp_path, monkeypatch):
    """Server without a key = misconfiguration, not 401."""
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key=None)
    with TestClient(app) as client:
        r = client.post("/start-session")
    assert r.status_code == 500
    assert "INTAKE_API_KEY" in r.json()["detail"]


# --- ACCORD_AUTH_DISABLED: global bypass ---

def test_accord_auth_disabled_opens_all_endpoints(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, api_key=None, auth_disabled=True,
    )
    with TestClient(app) as client:
        r1 = client.post("/start-session")
        r2 = client.get("/sessions")
    assert r1.status_code == 200
    assert r2.status_code == 200


# --- ACCORD_CHAT_OPEN: chat endpoints only ---

def test_chat_open_unlocks_chat_endpoints_only(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, api_key="secret", chat_open=True,
    )
    with TestClient(app) as client:
        r_chat = client.post("/start-session")
        r_read = client.get("/sessions")
    assert r_chat.status_code == 200
    assert r_read.status_code == 401   # read endpoints stay gated


def test_chat_open_still_allows_read_with_key(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, api_key="secret", chat_open=True,
    )
    with TestClient(app) as client:
        r = client.get("/sessions", headers={"x-api-key": "secret"})
    assert r.status_code == 200


def test_chat_open_gates_session_detail_endpoint(tmp_path, monkeypatch):
    """/session/{id} is a read endpoint — never anonymously open."""
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, api_key="secret", chat_open=True,
    )
    with TestClient(app) as client:
        # Create a session (chat endpoint, open)
        sid = client.post("/start-session").json()["submission_id"]
        # Read endpoint is gated even with chat_open
        r = client.get(f"/session/{sid}")
    assert r.status_code == 401


# --- Trace headers survive auth failures ---

def test_auth_failure_still_echoes_request_id(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/start-session", headers={"x-request-id": "trace-abc"},
        )
    assert r.status_code == 401
    assert r.headers["x-request-id"] == "trace-abc"


# --- CORS ---

def test_cors_preflight_returns_allow_origin(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch,
        allowed_origins="https://app.example.com",
    )
    with TestClient(app) as client:
        r = client.options(
            "/start-session",
            headers={
                "origin": "https://app.example.com",
                "access-control-request-method": "POST",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_cors_origin_regex_matches_multi_tenant_subdomains(tmp_path, monkeypatch):
    """v3's multi-tenant pattern: per-tenant subdomains."""
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch,
        allowed_origins="",
        allowed_origin_regex=r"^https?://[a-z0-9-]+\.example\.com$",
    )
    with TestClient(app) as client:
        r = client.options(
            "/start-session",
            headers={
                "origin": "https://acme.example.com",
                "access-control-request-method": "POST",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://acme.example.com"


# ============================================================
# P10.0.a — INTAKE_API_KEYS (key→tenant binding)
# ============================================================

import json


def _fastapi_with_bindings(
    tmp_path, monkeypatch, *,
    bindings: dict,
    admin_key: Optional[str] = None,
):
    """Build a FastAPI app with per-tenant bound keys (and optional admin key).

    bindings: {api_key: tenant_slug}. Serialized to INTAKE_API_KEYS (JSON).
    admin_key: optional shared INTAKE_API_KEY.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("INTAKE_API_KEYS", json.dumps(bindings))
    if admin_key is not None:
        monkeypatch.setenv("INTAKE_API_KEY", admin_key)
    else:
        monkeypatch.delenv("INTAKE_API_KEY", raising=False)

    settings = Settings()
    intake = build_intake_app(
        settings,
        engine=FakeEngine(["g1", "g2", "g3", "g4", "g5", "g6"]),
        refiner_engine=FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake), intake


def test_tenant_bound_key_with_matching_tenant_header_succeeds(
    tmp_path, monkeypatch,
):
    app, _ = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme-xyz": "acme"},
    )
    with TestClient(app) as client:
        r = client.post(
            "/start-session",
            headers={"x-api-key": "sk-acme-xyz", "x-tenant-slug": "acme"},
        )
    assert r.status_code == 200
    assert r.headers.get("x-tenant-slug") == "acme"


def test_tenant_bound_key_without_header_succeeds_and_uses_binding(
    tmp_path, monkeypatch,
):
    """No X-Tenant-Slug header = use the bound tenant from the key."""
    app, intake = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme-xyz": "acme"},
    )
    with TestClient(app) as client:
        body = client.post(
            "/start-session", headers={"x-api-key": "sk-acme-xyz"},
        ).json()
    session = intake.store.get_session(body["submission_id"], tenant="acme")
    assert session is not None
    assert session.tenant == "acme"


def test_tenant_bound_key_with_wrong_tenant_header_returns_403(
    tmp_path, monkeypatch,
):
    """Mismatched header is NOT silently ignored — this is the v3 leak."""
    app, _ = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme-xyz": "acme"},
    )
    with TestClient(app) as client:
        r = client.post(
            "/start-session",
            headers={"x-api-key": "sk-acme-xyz", "x-tenant-slug": "globex"},
        )
    assert r.status_code == 403
    assert "acme" in r.json()["detail"]
    assert "globex" in r.json()["detail"]


def test_tenant_bound_key_cannot_read_other_tenants_sessions(
    tmp_path, monkeypatch,
):
    """acme's key + globex's session_id must NOT leak globex's data."""
    app, intake = _fastapi_with_bindings(
        tmp_path, monkeypatch,
        bindings={"sk-acme-xyz": "acme", "sk-globex-abc": "globex"},
    )
    with TestClient(app) as client:
        # globex creates a session
        globex_sid = client.post(
            "/start-session", headers={"x-api-key": "sk-globex-abc"},
        ).json()["submission_id"]

        # acme's key tries to read globex's session — binding pins acme,
        # store tenant-isolates, KeyError → 404 (uniform tenant-leak-safe).
        r = client.get(
            f"/session/{globex_sid}", headers={"x-api-key": "sk-acme-xyz"},
        )
    assert r.status_code == 404


def test_admin_key_can_access_any_tenant(tmp_path, monkeypatch):
    """Admin key is backward-compat: caller picks tenant via header."""
    app, intake = _fastapi_with_bindings(
        tmp_path, monkeypatch,
        bindings={},
        admin_key="sk-admin-xyz",
    )
    with TestClient(app) as client:
        for slug in ("acme", "globex"):
            r = client.post(
                "/start-session",
                headers={"x-api-key": "sk-admin-xyz", "x-tenant-slug": slug},
            )
            assert r.status_code == 200

        all_sessions = client.get(
            "/sessions", headers={"x-api-key": "sk-admin-xyz"},
        ).json()
    assert len(all_sessions["sessions"]) == 2


def test_unknown_key_with_bindings_configured_returns_401(
    tmp_path, monkeypatch,
):
    """Unknown key + no admin fallback → 401, not 500 (server IS configured)."""
    app, _ = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme-xyz": "acme"},
    )
    with TestClient(app) as client:
        r = client.post(
            "/start-session", headers={"x-api-key": "sk-bogus"},
        )
    assert r.status_code == 401


def test_start_session_accepts_tenant_in_body_under_admin_key(
    tmp_path, monkeypatch,
):
    """v3 wire: admin key + body.tenant_slug — body picks the tenant."""
    app, intake = _fastapi_with_bindings(
        tmp_path, monkeypatch,
        bindings={},
        admin_key="sk-admin-xyz",
    )
    with TestClient(app) as client:
        body = client.post(
            "/start-session",
            headers={"x-api-key": "sk-admin-xyz"},
            json={"tenant_slug": "acme"},
        ).json()
    session = intake.store.get_session(body["submission_id"], tenant="acme")
    assert session is not None
    assert session.tenant == "acme"


def test_start_session_body_tenant_ignored_when_key_is_bound(
    tmp_path, monkeypatch,
):
    """Bound-key binding wins over body.tenant_slug — no cross-tenant hop."""
    app, intake = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme-xyz": "acme"},
    )
    with TestClient(app) as client:
        body = client.post(
            "/start-session",
            headers={"x-api-key": "sk-acme-xyz"},
            json={"tenant_slug": "globex"},
        ).json()
    # Body tenant was ignored — session is bound to acme.
    assert intake.store.get_session(body["submission_id"], tenant="acme") is not None
    assert intake.store.get_session(body["submission_id"], tenant="globex") is None


# ============================================================
# P10.0.b — auth logging + CORS fixes
# ============================================================

import logging as _logging


@pytest.fixture
def accord_caplog(caplog):
    """caplog that captures accord_ai.api records.

    configure_logging() sets propagate=False on the 'accord_ai' logger, so
    records from child loggers never reach pytest's root-attached caplog
    handler. We attach caplog's handler directly to 'accord_ai.api' — which
    configure_logging leaves alone (it only clears handlers on the root
    'accord_ai' logger) — so captures work regardless of propagation.
    """
    api_logger = _logging.getLogger("accord_ai.api")
    api_logger.addHandler(caplog.handler)
    original_level = api_logger.level
    api_logger.setLevel(_logging.DEBUG)
    try:
        yield caplog
    finally:
        api_logger.removeHandler(caplog.handler)
        api_logger.setLevel(original_level)


def test_invalid_key_logs_warning_with_path_and_ip(
    tmp_path, monkeypatch, accord_caplog,
):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key="secret")
    with TestClient(app) as client:
        client.post("/start-session", headers={"x-api-key": "wrong"})
    hits = [
        r for r in accord_caplog.records
        if "invalid key" in r.getMessage().lower()
    ]
    assert hits, "expected 'invalid key' warning"
    assert "/start-session" in hits[0].getMessage()
    assert "ip=" in hits[0].getMessage()


def test_misconfig_logs_error(tmp_path, monkeypatch, accord_caplog):
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key=None)
    with TestClient(app) as client:
        client.post("/start-session")
    hits = [
        r for r in accord_caplog.records
        if "misconfig" in r.getMessage().lower()
    ]
    assert hits, "expected 'misconfig' error"
    assert hits[0].levelno >= _logging.ERROR


def test_tenant_binding_violation_logs_warning(
    tmp_path, monkeypatch, accord_caplog,
):
    app, _ = _fastapi_with_bindings(
        tmp_path, monkeypatch, bindings={"sk-acme": "acme"},
    )
    with TestClient(app) as client:
        client.post(
            "/start-session",
            headers={"x-api-key": "sk-acme", "x-tenant-slug": "globex"},
        )
    hits = [
        r for r in accord_caplog.records
        if "tenant-binding violation" in r.getMessage()
    ]
    assert hits
    assert "bound=acme" in hits[0].getMessage()
    assert "claimed=globex" in hits[0].getMessage()


def test_x_forwarded_for_is_used_for_client_ip(
    tmp_path, monkeypatch, accord_caplog,
):
    """Behind a reverse proxy, X-Forwarded-For wins over direct peer."""
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch, api_key="secret")
    with TestClient(app) as client:
        client.post(
            "/start-session",
            headers={
                "x-api-key": "wrong",
                "x-forwarded-for": "203.0.113.42, 10.0.0.1",
            },
        )
    hits = [
        r for r in accord_caplog.records
        if "invalid key" in r.getMessage().lower()
    ]
    assert hits
    assert "ip=203.0.113.42" in hits[0].getMessage()


# --- CORS ---

def test_cors_wildcard_disables_credentials(
    tmp_path, monkeypatch, accord_caplog,
):
    """allow_origins='*' is incompatible with allow_credentials=True per spec."""
    app, _ = _fastapi_with_auth(tmp_path, monkeypatch)   # default "*"
    # Warning logged at construction time
    assert any(
        "CORS" in r.getMessage() and "wildcard" in r.getMessage().lower()
        for r in accord_caplog.records
    )
    # Preflight must NOT announce allow_credentials=true
    with TestClient(app) as client:
        r = client.options(
            "/start-session",
            headers={
                "origin": "https://app.example.com",
                "access-control-request-method": "POST",
            },
        )
    assert r.headers.get("access-control-allow-credentials") != "true"


def test_cors_non_wildcard_keeps_credentials_enabled(tmp_path, monkeypatch):
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, allowed_origins="https://app.example.com",
    )
    with TestClient(app) as client:
        r = client.options(
            "/start-session",
            headers={
                "origin": "https://app.example.com",
                "access-control-request-method": "POST",
            },
        )
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_cors_exposes_trace_headers_to_browser(tmp_path, monkeypatch):
    """CORS clients need explicit expose_headers to read X-Request-Id."""
    app, _ = _fastapi_with_auth(
        tmp_path, monkeypatch, allowed_origins="https://app.example.com",
    )
    with TestClient(app) as client:
        r = client.get(
            "/health",
            headers={"origin": "https://app.example.com"},
        )
    expose = (r.headers.get("access-control-expose-headers") or "").lower()
    assert "x-request-id" in expose
    assert "x-tenant-slug" in expose
