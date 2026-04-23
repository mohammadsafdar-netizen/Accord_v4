"""Tests for FilledPdfStore (P10.A.4 storage half)."""
from __future__ import annotations

import json

import pytest

from accord_ai.forms import FilledPdfStore


@pytest.fixture
def store(tmp_path):
    return FilledPdfStore(tmp_path / "pdfs")


SID_A = "a" * 32
SID_B = "b" * 32


# --- Save / load roundtrip ---------------------------------------------------

def test_save_and_load_roundtrip(store):
    assert store.save(SID_A, "acme", "125", b"%PDF-fake125", "hash125") is True
    assert store.load(SID_A, "acme", "125") == b"%PDF-fake125"


def test_load_missing_returns_none(store):
    assert store.load(SID_A, "acme", "125") is None


def test_list_forms_sorted_numerically(store):
    store.save(SID_A, "acme", "163", b"p163", "h163")
    store.save(SID_A, "acme", "125", b"p125", "h125")
    store.save(SID_A, "acme", "130", b"p130", "h130")
    assert store.list_forms(SID_A, "acme") == ["125", "130", "163"]


def test_manifest_returns_defensive_copy(store):
    store.save(SID_A, "acme", "125", b"p", "hash1")
    m = store.manifest(SID_A, "acme")
    m["999"] = "mutated"
    assert store.manifest(SID_A, "acme") == {"125": "hash1"}


# --- Dedup -------------------------------------------------------------------

def test_save_with_same_hash_is_skipped(store):
    assert store.save(SID_A, "acme", "125", b"p1", "hash_x") is True
    assert store.save(SID_A, "acme", "125", b"p1", "hash_x") is False


def test_save_with_new_hash_overwrites(store):
    store.save(SID_A, "acme", "125", b"v1", "hash_v1")
    assert store.save(SID_A, "acme", "125", b"v2", "hash_v2") is True
    assert store.load(SID_A, "acme", "125") == b"v2"
    assert store.manifest(SID_A, "acme")["125"] == "hash_v2"


def test_dedup_miss_when_pdf_deleted_out_of_band(store, tmp_path):
    """If someone rm'd the PDF but the manifest still points to it,
    next save should rewrite (not silently dedup to nothing)."""
    store.save(SID_A, "acme", "125", b"v1", "hash_v1")
    store._pdf_path(SID_A, "acme", "125").unlink()
    assert store.save(SID_A, "acme", "125", b"v1", "hash_v1") is True
    assert store.load(SID_A, "acme", "125") == b"v1"


# --- Tenant isolation --------------------------------------------------------

def test_tenant_isolation(store):
    store.save(SID_A, "acme",   "125", b"acme-125", "h1")
    store.save(SID_A, "globex", "125", b"globex-125", "h2")
    assert store.load(SID_A, "acme",   "125") == b"acme-125"
    assert store.load(SID_A, "globex", "125") == b"globex-125"


def test_wrong_tenant_load_returns_none(store):
    store.save(SID_A, "acme", "125", b"p", "h")
    assert store.load(SID_A, "globex", "125") is None
    assert store.list_forms(SID_A, "globex") == []


def test_no_tenant_bucket(store):
    store.save(SID_A, None, "125", b"p", "h")
    assert store.load(SID_A, None, "125") == b"p"
    assert store.list_forms(SID_A, None) == ["125"]


def test_reserved_tenant_slug_rejected(store):
    with pytest.raises(ValueError, match="reserved"):
        store.save(SID_A, "_no_tenant", "125", b"p", "h")


# --- Input validation --------------------------------------------------------

def test_invalid_tenant_rejected(store):
    for bad in ("UPPERCASE", "has spaces", "../../etc/passwd", "t.e.n.a.n.t"):
        with pytest.raises(ValueError, match="invalid tenant"):
            store.save(SID_A, bad, "125", b"p", "h")


def test_invalid_session_id_rejected(store):
    for bad in ("not-a-uuid", "../traverse", "", "g" * 32):
        with pytest.raises(ValueError, match="invalid session_id"):
            store.save(bad, "acme", "125", b"p", "h")


def test_invalid_form_number_rejected(store):
    for bad in ("12", "12345", "abc", "../125", ""):
        with pytest.raises(ValueError, match="invalid form_number"):
            store.save(SID_A, "acme", bad, b"p", "h")


# --- Manifest durability -----------------------------------------------------

def test_corrupted_manifest_treated_as_empty(store, caplog):
    """configure_logging() disables propagate on `accord_ai`, so records
    from `accord_ai.forms.storage` don't reach the root-attached caplog
    handler in test-suite ordering. Attach directly to the logger."""
    import logging
    storage_logger = logging.getLogger("accord_ai.forms.storage")
    storage_logger.addHandler(caplog.handler)
    original_level = storage_logger.level
    storage_logger.setLevel(logging.DEBUG)
    try:
        store.save(SID_A, "acme", "125", b"p1", "h1")
        store._manifest_path(SID_A, "acme").write_text("not valid json{{")

        # Next save should not dedup against an unreadable manifest.
        assert store.save(SID_A, "acme", "125", b"p2", "h2") is True
        # Manifest got rewritten from scratch; only the latest form remains.
        assert store.manifest(SID_A, "acme") == {"125": "h2"}
        assert any(
            "unreadable" in r.getMessage() for r in caplog.records
        )
    finally:
        storage_logger.removeHandler(caplog.handler)
        storage_logger.setLevel(original_level)


def test_manifest_is_sorted(store):
    """Stable on-disk ordering → reproducible diffs if someone checks it in."""
    store.save(SID_A, "acme", "163", b"p", "h163")
    store.save(SID_A, "acme", "125", b"p", "h125")
    text = store._manifest_path(SID_A, "acme").read_text()
    data = json.loads(text)
    assert list(data.keys()) == sorted(data.keys())


# --- Clear -------------------------------------------------------------------

def test_clear_session_removes_everything(store):
    store.save(SID_A, "acme", "125", b"p", "h")
    store.save(SID_A, "acme", "126", b"p", "h")
    store.clear_session(SID_A, "acme")
    assert store.list_forms(SID_A, "acme") == []
    assert store.load(SID_A, "acme", "125") is None


def test_clear_session_idempotent(store):
    store.clear_session(SID_A, "acme")     # no-op on nonexistent session
    store.save(SID_A, "acme", "125", b"p", "h")
    store.clear_session(SID_A, "acme")
    store.clear_session(SID_A, "acme")     # second call also fine
