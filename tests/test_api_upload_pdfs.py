"""Tests for POST /upload-filled-pdfs and POST /upload-blank-pdfs (Phase 1.10)."""
import io
import pytest
from fastapi.testclient import TestClient

from accord_ai.api import build_fastapi_app
from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.llm.fake_engine import FakeEngine


def _make_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("KNOWLEDGE_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    settings = Settings()
    engine = FakeEngine(["Hello, let's start."])
    intake = build_intake_app(settings, engine=engine, refiner_engine=FakeEngine())
    return build_fastapi_app(settings, intake=intake)


def _fake_pdf(form="125") -> bytes:
    return b"%PDF-1.4 FAKE CONTENT for form " + form.encode()


# ---------------------------------------------------------------------------
# /upload-filled-pdfs
# ---------------------------------------------------------------------------

def test_upload_filled_pdfs_two_valid_one_invalid(tmp_path, monkeypatch):
    """2 valid PDFs → uploaded, 1 non-PDF file with valid form filename → failed."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post(
            "/upload-filled-pdfs",
            data={"session_id": sid},
            files=[
                ("files", ("125-Form.pdf", io.BytesIO(_fake_pdf("125")), "application/pdf")),
                ("files", ("127-Form.pdf", io.BytesIO(_fake_pdf("127")), "application/pdf")),
                # Valid filename pattern but invalid PDF content
                ("files", ("129-Form.pdf", io.BytesIO(b"NOT PDF CONTENT"), "application/pdf")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["uploaded"]) == 2
    assert len(body["failed"]) == 1
    uploaded_forms = {u["form_number"] for u in body["uploaded"]}
    assert "125" in uploaded_forms
    assert "127" in uploaded_forms
    assert body["failed"][0]["form_number"] == "129"
    assert "PDF" in body["failed"][0]["error"]


def test_upload_filled_pdfs_unknown_session_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/upload-filled-pdfs",
            data={"session_id": "no-such-session"},
            files=[
                ("files", ("125-Form.pdf", io.BytesIO(_fake_pdf()), "application/pdf")),
            ],
        )
    assert r.status_code == 404


def test_upload_filled_pdfs_drive_not_configured_still_succeeds(tmp_path, monkeypatch):
    """When Drive is not configured, PDFs are stored locally (no Drive metadata)."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post(
            "/upload-filled-pdfs",
            data={"session_id": sid},
            files=[
                ("files", ("125-Form.pdf", io.BytesIO(_fake_pdf()), "application/pdf")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["uploaded"]) == 1
    assert body["uploaded"][0]["drive_file_id"] is None
    assert body["uploaded"][0]["drive_url"] is None


# ---------------------------------------------------------------------------
# /upload-blank-pdfs
# ---------------------------------------------------------------------------

def test_upload_blank_pdfs_known_lob(tmp_path, monkeypatch):
    """Known LOB returns 200 with stub response."""
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post(
            "/upload-blank-pdfs",
            json={"session_id": sid, "lob": "commercial_auto"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "uploaded" in body
    assert "failed" in body
    assert "Phase 1.5" in body["note"]


def test_upload_blank_pdfs_unknown_lob_returns_422(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/start-session").json()["submission_id"]
        r = client.post(
            "/upload-blank-pdfs",
            json={"session_id": sid, "lob": "nonexistent_lob"},
        )
    assert r.status_code == 422
    assert "LOB" in r.json()["detail"]


def test_upload_blank_pdfs_unknown_session_returns_404(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/upload-blank-pdfs",
            json={"session_id": "no-such-session", "lob": "commercial_auto"},
        )
    assert r.status_code == 404
