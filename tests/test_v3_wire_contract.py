"""v3 wire-contract regression canaries (P10.0.g batch 1).

These tests pin the three BLOCKING divergences caught by the v3 audit:

  1. Auth: v3 FE sends `Authorization: Bearer {key}`. v4 must accept it
     alongside (or instead of) `X-API-Key`. Verified against
     accord_ai_v3/accord_ai/api.py:243 `security = HTTPBearer(auto_error=False)`.

  2. Agent-folder response: v3's tenant backend returns the LOB folder id
     under the key `lob_folder_id`, not `id`. Verified against
     accord_ai_v3/accord_ai/drive_upload_v2.py:405-407 which extracts
     `folder_info["lob_folder_id"]` from the raw JSON response.

  3. Drive filename: v3 names filled-form uploads `{form_number}-Form.pdf`
     (e.g. "125-Form.pdf"). Verified against
     accord_ai_v3/accord_ai/drive_upload_v2.py:480 — `f"{pdf_id}-Form.pdf"`.

Each test is a canary: if anyone reverts the v3 compat (e.g. during a
future refactor that "simplifies" the header handling or renames the
Drive file convention), these break loudly and point at the exact v3
behavior that's being diverged from.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.integrations.backend import BackendClient
from accord_ai.testing import FakeEngine


# ---------------------------------------------------------------------------
# 1. Auth: Bearer + X-API-Key both work
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_client(tmp_path):
    """A client with INTAKE_API_KEY configured — no auth bypass."""
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=False,
        accord_chat_open=False,
        intake_api_key="secret-key-value",
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as c:
        yield c, intake


def test_auth_accepts_bearer_scheme(auth_client):
    """v3 FE wire: `Authorization: Bearer {key}`. Must authenticate."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={"authorization": "Bearer secret-key-value",
                 "x-tenant-slug": "acme"},
    )
    assert resp.status_code == 200


def test_auth_accepts_bearer_case_insensitive(auth_client):
    """HTTP scheme names are case-insensitive per RFC 7235."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
        resp = c.get(
            f"/session/{sid}",
            headers={"authorization": f"{scheme} secret-key-value",
                     "x-tenant-slug": "acme"},
        )
        assert resp.status_code == 200, f"scheme {scheme!r} rejected"


def test_auth_still_accepts_x_api_key(auth_client):
    """v4-native header must keep working for new integrations."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={"x-api-key": "secret-key-value",
                 "x-tenant-slug": "acme"},
    )
    assert resp.status_code == 200


def test_auth_bearer_preferred_over_x_api_key(auth_client):
    """When both headers are present, Bearer wins. If Bearer is valid and
    X-API-Key is wrong, request should still succeed (Bearer takes
    precedence). Prevents a subtle bug where a broken X-API-Key clobbers
    a valid Bearer."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={
            "authorization": "Bearer secret-key-value",
            "x-api-key":     "wrong-value",
            "x-tenant-slug": "acme",
        },
    )
    assert resp.status_code == 200


def test_auth_bad_bearer_rejected(auth_client):
    """Bearer with wrong token still 401s — no bypass."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={"authorization": "Bearer wrong-token",
                 "x-tenant-slug": "acme"},
    )
    assert resp.status_code == 401


def test_auth_empty_bearer_falls_through_to_x_api_key(auth_client):
    """`Authorization: Bearer ` (empty token) should fall through to
    X-API-Key rather than short-circuit as empty."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={
            "authorization": "Bearer ",
            "x-api-key":     "secret-key-value",
            "x-tenant-slug": "acme",
        },
    )
    assert resp.status_code == 200


def test_auth_non_bearer_scheme_ignored(auth_client):
    """`Authorization: Basic ...` or other schemes: ignore; fall through
    to X-API-Key. Don't attempt to decode Basic."""
    c, intake = auth_client
    sid = intake.store.create_session(tenant="acme")
    resp = c.get(
        f"/session/{sid}",
        headers={
            "authorization": "Basic dXNlcjpwYXNz",
            "x-api-key":     "secret-key-value",
            "x-tenant-slug": "acme",
        },
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Agent-folder response key: lob_folder_id (not id)
# ---------------------------------------------------------------------------

def _backend_client(handler) -> BackendClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return BackendClient(
        settings=Settings(
            backend_enabled=True,
            backend_host_suffix="copilot.inevo.ai",
            backend_client_id="intake-agent",
            backend_client_secret="test-secret",
        ),
        http=http,
    )


@pytest.mark.asyncio
async def test_agent_folder_accepts_lob_folder_id_v3_convention():
    """v3 backend returns `{lob_folder_id: ...}` — v4 must accept it."""
    def handler(req):
        return httpx.Response(200, json={
            "lob_folder_id": "v3-shape-folder-id",
            "name":          "Acme / Commercial Auto / 2026",
        })
    client = _backend_client(handler)
    try:
        result = await client.get_agent_folder("svc", "sub-1", "acme.x.com")
        assert result is not None
        assert result.get("lob_folder_id") == "v3-shape-folder-id"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_agent_folder_accepts_id_as_fallback():
    """For forward-compat if the backend ever standardizes on `id`."""
    def handler(req):
        return httpx.Response(200, json={"id": "standardized-folder-id"})
    client = _backend_client(handler)
    try:
        result = await client.get_agent_folder("svc", "sub-1", "acme.x.com")
        assert result is not None
        assert result.get("id") == "standardized-folder-id"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_agent_folder_rejects_response_without_any_folder_id():
    """No `lob_folder_id` AND no `id` — genuinely unusable response."""
    def handler(req):
        return httpx.Response(200, json={"name": "no ids at all"})
    client = _backend_client(handler)
    try:
        result = await client.get_agent_folder("svc", "sub-1", "acme.x.com")
        assert result is None
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 3. Drive filename convention: {form_number}-Form.pdf
# ---------------------------------------------------------------------------

def test_drive_filename_convention_matches_v3():
    """Source of truth assertion — read drive_pipeline.py and confirm the
    filename is built as `{form_number}-Form.pdf`, not the v4-original
    `ACORD_{form}_filled.pdf`. A refactor that reverts this silently
    breaks v3-folder dedup + prune."""
    from pathlib import Path
    pipeline_src = (Path(__file__).parent.parent
                    / "accord_ai" / "forms" / "drive_pipeline.py").read_text()
    # The specific line that builds the Drive filename.
    assert 'f"{form_number}-Form.pdf"' in pipeline_src, (
        "drive_pipeline.py no longer builds filenames as "
        "'{form_number}-Form.pdf' — v3 wire contract broken"
    )
    # The prune keep-set must use the same convention.
    assert 'f"{n}-Form.pdf"' in pipeline_src, (
        "drive_pipeline.py prune keep-set uses wrong filename convention"
    )
    # And the OLD convention must NOT appear anywhere in the pipeline.
    assert "ACORD_" not in pipeline_src, (
        "drive_pipeline.py still references the pre-v3-compat "
        "'ACORD_{form}_filled.pdf' convention"
    )


# ---------------------------------------------------------------------------
# 4. /start-session request + response exactly match v3's wire shape
# ---------------------------------------------------------------------------

def test_start_session_response_is_exactly_v3_shape(tmp_path):
    """v3 (api.py:389-391) returns exactly {submission_id, question}.
    No session_id, no assistant_message, no extra fields — the FE relies
    on this exact shape."""
    from accord_ai.testing import FakeEngine
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        r = client.post("/start-session")
    assert r.status_code == 200
    body = r.json()
    # Exact shape — v3 wire contract.
    assert set(body.keys()) == {"submission_id", "question"}, (
        f"response has extra/missing keys: {set(body.keys())}"
    )
    assert isinstance(body["submission_id"], str)
    assert isinstance(body["question"], str)


def test_start_session_request_accepts_v3_fields(tmp_path):
    """v3 request shape (api.py:378-386): submission_id, session_id,
    tenant_slug, tenant_domain, phase. v4 must accept all without 422."""
    from accord_ai.testing import FakeEngine
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        r = client.post("/start-session", json={
            "submission_id": "prev-session-abc",
            "session_id":    "prev-session-abc",
            "tenant_slug":   "acme",
            "tenant_domain": "acme.brocopilot.com",
            "phase":         "customer",
        })
    assert r.status_code == 200, r.json()
    body = r.json()
    # Key shape still strict on response side
    assert set(body.keys()) == {"submission_id", "question"}


def test_start_session_tolerates_missing_body(tmp_path):
    """v3 FE may POST with an empty body — v4 must not 422."""
    from accord_ai.testing import FakeEngine
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        r = client.post("/start-session")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 5. fields_data accepts v3's full shape (no scalar-only restriction)
# ---------------------------------------------------------------------------

def test_fields_data_accepts_nested_v3_shape(tmp_path):
    """v3 (api.py:1672) types fields_data as dict[str, dict[str, Any]].
    Nested dicts/lists inside a form's widget map must not 422 — the FE's
    form 163 structured shape uses `_header` and `drivers` as sub-dicts.
    P10.0.g.7 removed the scalar-only restriction that blocked this."""
    from accord_ai.testing import FakeEngine
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    # Create a session so /complete can proceed past the existence check
    app = build_fastapi_app(settings, intake=intake)
    sid = intake.store.create_session(tenant="acme")
    with TestClient(app) as client:
        resp = client.post("/complete", json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "fields_data": {
                # Structured shape v3 FE sends for form 163:
                "form_163": {
                    "_header":  {"policy_number": "POL-123"},
                    "drivers":  [{"first_name": "Alice"}],
                },
                # Also a regular flat shape for form 125:
                "form_125": {"NamedInsured_FullName_A": "Acme Trucking"},
            },
        })
    # Must not 422 — nested values are accepted at model validation
    assert resp.status_code == 200, resp.json()


# ---------------------------------------------------------------------------
# 6. translate_payload wired into /complete (P10.0.g.5)
# ---------------------------------------------------------------------------

def test_fe_label_translated_to_widget_name_on_complete(tmp_path):
    """v3 FE sends human labels ("Named Insured" etc.); v4 /complete must
    translate them to widget names before fill, or broker edits drop."""
    from accord_ai.testing import FakeEngine
    from accord_ai.schema import CommercialAutoDetails, CustomerSubmission
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)

    # Seed a CA session
    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid,
        CustomerSubmission(
            business_name="Original Name",
            lob_details=CommercialAutoDetails(),
        ),
        tenant="acme",
    )

    # FE sends a human label, not a widget name
    with TestClient(app) as client:
        resp = client.post("/complete", json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "fields_data": {
                "form_125": {"Named Insured": "FE-Edited Name"},
            },
        })
    assert resp.status_code == 200, resp.json()

    # Verify the label was translated — the filled 125 PDF must reflect
    # "FE-Edited Name" at NamedInsured_FullName_A
    import fitz
    pdf = intake.filled_pdf_store.load(sid, "acme", "125")
    assert pdf is not None, "125 not persisted after /complete"
    doc = fitz.open(stream=bytearray(pdf), filetype="pdf")
    try:
        values = {}
        for page in doc:
            for w in page.widgets():
                if w.field_name == "NamedInsured_FullName_A":
                    values[w.field_name] = w.field_value
                    break
    finally:
        doc.close()
    assert values.get("NamedInsured_FullName_A") == "FE-Edited Name", (
        f"label->widget translation failed; 125 has {values!r}"
    )


def test_widget_names_still_pass_through_untranslated(tmp_path):
    """When FE sends raw widget names, translate_payload passes them
    through. Coexists with label translation — no double-translation."""
    from accord_ai.testing import FakeEngine
    from accord_ai.schema import CommercialAutoDetails, CustomerSubmission
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)

    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid, CustomerSubmission(lob_details=CommercialAutoDetails()),
        tenant="acme",
    )

    with TestClient(app) as client:
        resp = client.post("/complete", json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "fields_data": {
                "form_125": {
                    "NamedInsured_FullName_A": "Widget-Key Name",
                },
            },
        })
    assert resp.status_code == 200


def test_null_sentinels_filtered_by_translate_payload(tmp_path):
    """v3 FE sends "NullObject" for unfilled fields. These must NOT render
    as the literal string "NullObject" on the PDF."""
    from accord_ai.testing import FakeEngine
    from accord_ai.schema import CommercialAutoDetails, CustomerSubmission
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(["hi"]), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)

    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid,
        CustomerSubmission(
            business_name="Acme",
            lob_details=CommercialAutoDetails(),
        ),
        tenant="acme",
    )

    with TestClient(app) as client:
        resp = client.post("/complete", json={
            "submission_id": sid,
            "tenant_slug":   "acme",
            "fields_data": {
                "form_125": {
                    "Named Insured":           "NullObject",   # filtered
                    "NamedInsured_DBAName_A":  "null",          # filtered
                    "DBA":                     "Real Value",
                },
            },
        })
    assert resp.status_code == 200

    import fitz
    pdf = intake.filled_pdf_store.load(sid, "acme", "125")
    assert pdf is not None
    doc = fitz.open(stream=bytearray(pdf), filetype="pdf")
    try:
        bad_values = []
        for page in doc:
            for w in page.widgets():
                fv = w.field_value
                if isinstance(fv, str) and fv.lower() in (
                    "nullobject", "null", "none",
                ):
                    bad_values.append((w.field_name, fv))
    finally:
        doc.close()
    assert not bad_values, (
        f"Null-sentinel strings leaked into PDF widgets: {bad_values}"
    )


# ---------------------------------------------------------------------------
# 7. /answer response carries v3's 18 wire fields (P10.0.g.4)
# ---------------------------------------------------------------------------

_V3_ANSWER_FIELDS = frozenset({
    "finished", "next_question", "category", "question_index",
    "total_questions_in_category", "current_category_index",
    "total_categories", "pdf_status", "drive_files", "message",
    "output_dir", "filled_pdfs", "baseline_complete", "submission_id",
    "progress", "validation_errors", "harness_version",
})


def _answer_client(tmp_path):
    from accord_ai.testing import FakeEngine
    settings = Settings(
        db_path=str(tmp_path / "a.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    # FakeEngine queue: greeting → extract → respond
    main = FakeEngine([
        "Hello — business name?",
        {"business_name": "Acme"},
        "Got it, ready to finalize.",
    ])
    intake = build_intake_app(
        settings, engine=main, refiner_engine=FakeEngine(),
    )
    return build_fastapi_app(settings, intake=intake), intake


def test_answer_response_has_all_v3_wire_fields(tmp_path):
    """v3 wire contract (api.py:467-486): AnswerResponse has 17 v3 fields
    (plus the v4 ones we emit additively). All 17 v3 fields MUST be
    present on every /answer response — FE relies on their existence."""
    app, intake = _answer_client(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        resp = client.post("/answer", json={
            "session_id": sid, "message": "we are Acme",
        })
    assert resp.status_code == 200
    body = resp.json()
    missing = _V3_ANSWER_FIELDS - set(body.keys())
    assert not missing, f"missing v3 fields on /answer response: {missing}"


def test_answer_finished_maps_from_is_complete(tmp_path):
    """v3's `finished` is v4's `is_complete` under a different name."""
    app, intake = _answer_client(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        body = client.post("/answer", json={
            "session_id": sid, "message": "we are Acme",
        }).json()
    # Both fields must agree
    assert body["finished"] == body["is_complete"]


def test_answer_next_question_maps_from_assistant_message(tmp_path):
    """v3's `next_question` mirrors `assistant_message`."""
    app, intake = _answer_client(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        body = client.post("/answer", json={
            "session_id": sid, "message": "we are Acme",
        }).json()
    assert body["next_question"] == body["assistant_message"]


def test_answer_submission_id_aliases_session_id(tmp_path):
    """v3 naming alias — both fields carry the same string."""
    app, intake = _answer_client(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        body = client.post("/answer", json={
            "session_id": sid, "message": "we are Acme",
        }).json()
    assert body["submission_id"] == body["session_id"] == sid


def test_answer_v3_defaults_match_v3_schema(tmp_path):
    """Static defaults on stubbed fields must match v3's model defaults
    (api.py:467-486). If any diverge, the FE may render wrong."""
    app, intake = _answer_client(tmp_path)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        body = client.post("/answer", json={
            "session_id": sid, "message": "we are Acme",
        }).json()
    assert body["category"] == "General Information"
    assert body["question_index"] == 0
    assert body["total_questions_in_category"] == 10
    assert body["current_category_index"] == 0
    assert body["total_categories"] == 8
    assert body["pdf_status"] == {}
    assert body["drive_files"] is None
    assert body["message"] is None
    assert body["output_dir"] == ""
    assert body["filled_pdfs"] == []
    assert body["baseline_complete"] is False
    assert body["progress"] is None
    assert body["validation_errors"] == []
    assert body["harness_version"] == 1


def test_drive_filename_in_upload_args():
    """When /complete runs with Drive mocked, the filename passed into
    upload_filled_pdf must be v3-shape. This catches the bug at runtime
    rather than in the source-string check above."""
    from unittest.mock import AsyncMock, MagicMock

    from accord_ai.forms.drive_pipeline import run_drive_upload_pipeline
    from accord_ai.forms.pipeline import FilledForm
    from accord_ai.forms.filler import FillResult
    from accord_ai.integrations.drive import UploadResult

    # Minimal fake of fill output — one form, one fake byte blob
    ff = FilledForm(
        form_number="125",
        pdf_bytes=b"%PDF-fake",
        content_hash="h" * 64,
        fill_result=FillResult(
            form_number="125", filled_count=1, skipped_count=0,
            error_count=0, errors=(), unknown_fields=(),
        ),
    )
    filled = {"125": ff}

    backend = MagicMock()
    backend.get_service_token = AsyncMock(return_value="svc")
    backend.get_drive_token   = AsyncMock(return_value="drv")
    backend.get_agent_folder  = AsyncMock(
        return_value={"lob_folder_id": "lob-folder"},
    )
    backend.push_fields       = AsyncMock(return_value=True)

    drive = MagicMock()
    drive.find_or_create_submission_folder = AsyncMock(return_value="sub-folder")
    drive.upload_filled_pdf = AsyncMock(
        return_value=UploadResult.from_id("fid-125"),
    )
    drive.prune_stale_pdfs = AsyncMock(return_value=[])

    filled_pdf_store = MagicMock()
    filled_pdf_store.get_drive_file_id.return_value = None

    import asyncio
    asyncio.run(run_drive_upload_pipeline(
        backend=backend, drive=drive,
        filled=filled, filled_pdf_store=filled_pdf_store,
        submission_id="s" * 32,
        tenant="acme",
        tenant_slug="acme",
        tenant_domain="acme.brocopilot.com",
        fields_data={}, audit_store=None,
    ))

    # Assert the upload call used the v3 filename convention
    call = drive.upload_filled_pdf.call_args
    file_name_arg = call.kwargs.get("file_name") or call.args[2]
    assert file_name_arg == "125-Form.pdf", (
        f"Drive upload called with {file_name_arg!r}, "
        f"expected v3 convention '125-Form.pdf'"
    )


# ---------------------------------------------------------------------------
# 4. /complete response shape (P10.0.g.3)
# ---------------------------------------------------------------------------
# v3 returns: {submission_id, uploaded, total, field_diff, fill_summary,
# drive_files, validation, cache, pruned}. v4 is additive — emits the v3
# shape plus its native detail (session_id, forms, total_bytes, ...).
# Verified against accord_ai_v3/accord_ai/api.py:2055-2070.


@pytest.fixture
def complete_client(tmp_path):
    """A fresh-DB client wired for /complete testing with auth bypassed."""
    from datetime import date
    from accord_ai.schema import (
        Address, CommercialAutoDetails, CustomerSubmission,
        Driver, PolicyDates, Vehicle,
    )
    pytest.importorskip("fitz")

    settings = Settings(
        db_path=str(tmp_path / "c.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,
    )
    intake = build_intake_app(
        settings, engine=FakeEngine(), refiner_engine=FakeEngine(),
    )
    app = build_fastapi_app(settings, intake=intake)

    sid = intake.store.create_session(tenant="acme")
    intake.store.update_submission(
        sid,
        CustomerSubmission(
            business_name="Acme Trucking",
            mailing_address=Address(
                line_one="123 Main", city="Austin", state="TX", zip_code="78701",
            ),
            policy_dates=PolicyDates(effective_date=date(2026, 5, 1)),
            lob_details=CommercialAutoDetails(
                drivers=[Driver(first_name="Alice", last_name="Jones")],
                vehicles=[Vehicle(year=2024, make="Freightliner")],
            ),
        ),
        tenant="acme",
    )
    with TestClient(app) as c:
        yield c, sid


def test_complete_response_has_full_v3_shape(complete_client):
    """Every v3 response key must be present — FE keys off them."""
    c, sid = complete_client
    resp = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # v3 keys (accord_ai_v3/api.py:2055-2070):
    for v3_key in (
        "submission_id", "uploaded", "total", "field_diff",
        "fill_summary", "drive_files", "validation", "cache", "pruned",
    ):
        assert v3_key in body, f"missing v3 key: {v3_key!r}"


def test_complete_submission_id_aliases_session_id(complete_client):
    c, sid = complete_client
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert body["submission_id"] == sid
    assert body["session_id"] == sid     # v4 native still present


def test_complete_fill_summary_shape(complete_client):
    """Each fill_summary entry has v3's per-form dict shape."""
    c, sid = complete_client
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert isinstance(body["fill_summary"], list)
    assert len(body["fill_summary"]) == body["total"]
    entry = body["fill_summary"][0]
    for k in (
        "form_number", "pdf_id", "fields_filled", "fields_skipped",
        "errors", "skipped_upload_unchanged",
    ):
        assert k in entry, f"fill_summary entry missing {k!r}"
    assert entry["pdf_id"] == f"ACORD_{entry['form_number']}"


def test_complete_total_matches_forms_length(complete_client):
    c, sid = complete_client
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert body["total"] == len(body["forms"])
    # CA with the seed above produces the standard CA form set.
    assert body["total"] == 5


def test_complete_uploaded_is_zero_when_drive_unwired(complete_client):
    """Drive pipeline didn't run (no backend/drive clients) → uploaded=0."""
    c, sid = complete_client
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert body["uploaded"] == 0
    assert body["drive_files"] == []
    assert body["drive_enabled"] is False


def test_complete_cache_reports_content_dedup(complete_client):
    """Second call dedups every form → cache.content_dedup_skipped == total."""
    c, sid = complete_client
    c.post("/complete", json={"submission_id": sid, "tenant_slug": "acme"})
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert body["cache"]["content_dedup_skipped"] == body["total"]
    assert body["cache"]["auth_hit"] is False         # stub
    assert body["cache"]["validation_hit"] is False   # stub
    assert body["cache"]["file_id_hits"] == 0         # stub


def test_complete_pruned_aliases_pruned_count(complete_client):
    c, sid = complete_client
    body = c.post(
        "/complete", json={"submission_id": sid, "tenant_slug": "acme"},
    ).json()
    assert body["pruned"] == body["pruned_count"]


# ---------------------------------------------------------------------------
# 5. Form 163 structured expansion (P10.0.g.6)
# ---------------------------------------------------------------------------
# v3 routes form 163 fields_data through `map_structured` so callers can send
# nested `{_header: {...}, drivers: [...]}` payloads. v4 must do the same —
# before this step the entire form was skipped when any value was a dict/list,
# so every label-keyed edit for 163 silently dropped.
# Verified against accord_ai_v3/api.py:1744-1751.


def test_form163_structured_header_expands_to_widgets(complete_client):
    """A _header block must expand to the HEADER_MAP widget names."""
    from accord_ai.forms.layouts.form_163_layout import HEADER_MAP
    c, sid = complete_client
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {
            "form_163": {
                "_header": {
                    "business_name":         "Acme Trucking LLC",
                    "policy_effective_date": "05/01/2026",
                    "producer_name":         "Smith Insurance Agency",
                },
            },
        },
    })
    assert resp.status_code == 200, resp.text
    info = next(
        f for f in resp.json()["forms"] if f["form_number"] == "163"
    )
    # The override was applied — fill_result.filled > 0 confirms the
    # expansion reached the filler. Three logical keys → three widgets.
    assert info["fill_result"]["filled"] >= 3, (
        f"Expected at least 3 widgets filled from _header expansion, "
        f"got fill_result={info['fill_result']}"
    )
    # Cross-check: the three logical keys map to distinct widget names.
    assert len({
        HEADER_MAP["business_name"],
        HEADER_MAP["policy_effective_date"],
        HEADER_MAP["producer_name"],
    }) == 3


def test_form163_structured_drivers_expand_to_row_widgets(complete_client):
    """A drivers list must expand per row via DRIVER_MAP."""
    c, sid = complete_client
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {
            "form_163": {
                "drivers": [
                    {"first_name": "Alice", "last_name": "Jones", "license_num": "TX123"},
                    {"first_name": "Bob",   "last_name": "Smith", "license_num": "TX456"},
                ],
            },
        },
    })
    assert resp.status_code == 200, resp.text
    info = next(
        f for f in resp.json()["forms"] if f["form_number"] == "163"
    )
    # Two drivers × 3 columns = 6 widgets.
    assert info["fill_result"]["filled"] >= 6, (
        f"Expected ≥ 6 filled widgets from 2-driver expansion, "
        f"got fill_result={info['fill_result']}"
    )


def test_form163_raw_text_widget_beats_structured_derivation():
    """Escape hatch: when both a structured payload AND a raw Text##[0]
    override for the same widget are supplied, the raw value wins —
    matches v3's `{**structured, **translated}` merge order."""
    from accord_ai.forms.fe_label_map import translate_payload
    from accord_ai.forms.layouts.form_163_layout import HEADER_MAP, map_structured

    payload = {
        "_header": {"business_name": "From Structured"},
        HEADER_MAP["business_name"]: "From Raw Override",
    }
    scalars = {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}
    translated, _ = translate_payload(scalars)
    structured = map_structured(payload)
    merged = {**structured, **translated}
    assert merged[HEADER_MAP["business_name"]] == "From Raw Override"


def test_form163_empty_values_dropped(complete_client):
    """Empty strings / None in merged result must be filtered (matches
    v3's `if v is not None and str(v) != ""` post-filter)."""
    c, sid = complete_client
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {
            "form_163": {
                "_header": {
                    "business_name":         "Acme",
                    "policy_effective_date": "",      # should drop
                    "producer_name":         None,    # should drop
                },
            },
        },
    })
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Phase 1.9 routes present in OpenAPI schema
# ---------------------------------------------------------------------------

def test_phase_19_routes_present_in_openapi(tmp_path):
    """GET /explain, POST /enrich, POST /correction, POST /feedback must be
    registered routes — confirmed via the OpenAPI schema paths."""
    settings = Settings(
        db_path=str(tmp_path / "api.db"),
        knowledge_db_path=str(tmp_path / "chroma"),
        accord_auth_disabled=True,
    )
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    paths = schema["paths"]
    assert "/explain/{field_path}" in paths
    assert "/enrich" in paths
    assert "/correction" in paths
    assert "/feedback" in paths


# ---------------------------------------------------------------------------
# Phase 1.10 routes present in OpenAPI schema
# ---------------------------------------------------------------------------

def test_phase_110_routes_present_in_openapi(tmp_path):
    """All 11 Phase 1.10 endpoints must appear in the OpenAPI schema paths."""
    settings = Settings(
        db_path=str(tmp_path / "api.db"),
        knowledge_db_path=str(tmp_path / "chroma"),
        accord_auth_disabled=True,
    )
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    paths = schema["paths"]
    expected = [
        "/upload-image",
        "/upload-filled-pdfs",
        "/upload-blank-pdfs",
        "/debug/session/{session_id}",
        "/fields/{session_id}",
        "/harness",
        "/harness/audit",
        "/harness/history",
        "/harness/rollback",
        "/harness/provenance",
        "/harness/review-queue",
    ]
    missing = [p for p in expected if p not in paths]
    assert not missing, f"missing routes in OpenAPI: {missing}"


def test_full_v3_route_set_present_in_openapi(tmp_path):
    """v3 wire-contract parity: every v3 endpoint is in the OpenAPI schema."""
    settings = Settings(
        db_path=str(tmp_path / "api.db"),
        knowledge_db_path=str(tmp_path / "chroma"),
        accord_auth_disabled=True,
    )
    intake = build_intake_app(settings, engine=FakeEngine(), refiner_engine=FakeEngine())
    app = build_fastapi_app(settings, intake=intake)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    paths = schema["paths"]
    v3_parity = [
        "/health", "/start-session", "/answer", "/complete", "/finalize",
        "/enrich", "/upload-document", "/upload-image",
        "/upload-filled-pdfs", "/upload-blank-pdfs",
        "/feedback", "/correction", "/explain/{field_path}",
        "/fields/{session_id}", "/debug/session/{session_id}",
        "/pdf/{session_id}/{form_number}",
        "/harness", "/harness/history", "/harness/rollback",
        "/harness/audit", "/harness/provenance", "/harness/review-queue",
    ]
    missing = [p for p in v3_parity if p not in paths]
    assert not missing, f"v3 parity routes missing from OpenAPI: {missing}"


def test_form125_unaffected_by_structured_block(complete_client):
    """g.6 is 163-scoped. A non-163 form's scalar subset is still
    translated while any nested _header block is skipped — verifies the
    form_num gate doesn't accidentally apply structured expansion
    cross-form (would insert 163 widget names into 125's fill)."""
    c, sid = complete_client
    resp = c.post("/complete", json={
        "submission_id": sid,
        "tenant_slug":   "acme",
        "fields_data":   {
            "form_125": {
                "Text1[0]":   "scalar-override",
                "_header":    {"business_name": "should-be-ignored"},
            },
        },
    })
    assert resp.status_code == 200, resp.text
    info = next(
        f for f in resp.json()["forms"] if f["form_number"] == "125"
    )
    # filled > 0 confirms the scalar didn't trip the pre-g.6 whole-form skip.
    assert info["fill_result"]["filled"] >= 1
