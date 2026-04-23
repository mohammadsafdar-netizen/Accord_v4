"""Tests for FilledPdfStore Drive file ID tracking (P10.C.7).

Covers the extended manifest shape:
    {form: {"content_hash": ..., "drive_file_id": ...}}

Plus the two new methods (get_drive_file_id / set_drive_file_id) and the
extended save() signature with the keyword-only drive_file_id param.
"""
from __future__ import annotations

import json
import logging

import pytest

from accord_ai.forms import FilledPdfStore


SID_A = "a" * 32
SID_B = "b" * 32


@pytest.fixture
def store(tmp_path):
    return FilledPdfStore(tmp_path / "pdfs")


@pytest.fixture
def storage_caplog(caplog):
    """caplog that reliably captures accord_ai.forms.storage records.

    Mirrors the audit_caplog / accord_caplog pattern elsewhere in the suite:
    configure_logging() sets propagate=False on the 'accord_ai' logger, so
    child records never reach pytest's root-attached caplog handler. Attach
    directly to 'accord_ai.forms.storage' to capture regardless.
    """
    storage_logger = logging.getLogger("accord_ai.forms.storage")
    storage_logger.addHandler(caplog.handler)
    original_level = storage_logger.level
    storage_logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        storage_logger.removeHandler(caplog.handler)
        storage_logger.setLevel(original_level)


# --- Legacy manifest compatibility -----------------------------------------

def test_legacy_manifest_auto_promotes_on_read(store):
    """Old-shape manifest ({form: hash_string}) must read cleanly.

    get_drive_file_id returns None for legacy entries (no ID stored), and
    the next save() must persist the new dict shape on disk.
    """
    # Seed a legacy manifest directly.
    sess_dir = store._session_dir(SID_A, "acme")
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / "manifest.json").write_text(
        json.dumps({"125": "legacy_hash_125"})
    )

    # Legacy entries advertise no Drive ID.
    assert store.get_drive_file_id(SID_A, "acme", "125") is None

    # A save (new bytes → dedup miss) rewrites manifest in the new shape.
    assert store.save(
        SID_A, "acme", "125", b"new-bytes", "new_hash",
        drive_file_id="drive-abc",
    ) is True

    raw = json.loads(store._manifest_path(SID_A, "acme").read_text())
    assert raw == {
        "125": {"content_hash": "new_hash", "drive_file_id": "drive-abc"},
    }


def test_legacy_manifest_promote_logs_once(store, storage_caplog):
    """Reading a legacy manifest emits exactly one INFO log per read.

    Deduplicates across caplog capture paths — if configure_logging
    hasn't detached propagation yet, the same record can appear on both
    the root caplog handler AND our directly-attached one. What we care
    about is that _read_manifest emitted exactly one log *event*.
    """
    sess_dir = store._session_dir(SID_A, "acme")
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / "manifest.json").write_text(
        json.dumps({"125": "h125", "126": "h126"})
    )

    storage_caplog.clear()
    # Trigger a read.
    store.manifest(SID_A, "acme")

    legacy_logs = [
        r for r in storage_caplog.records
        if "legacy manifest auto-promoted" in r.getMessage()
        and r.levelno == logging.INFO
    ]
    # Dedup by identity — a single LogRecord may be captured by multiple
    # handlers (root + our attached one) depending on propagation state.
    unique_events = {id(r) for r in legacy_logs}
    # Every record references the same underlying event (same created ts).
    unique_timestamps = {r.created for r in legacy_logs}

    assert legacy_logs, "expected a legacy-promote INFO log"
    assert len(unique_timestamps) == 1, (
        f"expected a single emission, got {len(unique_timestamps)} distinct "
        f"timestamps across {len(legacy_logs)} captured records"
    )
    assert len(unique_events) <= 2, (
        "more than 2 captures implies something beyond caplog duplication"
    )


# --- get_drive_file_id ------------------------------------------------------

def test_get_drive_file_id_unknown_form_returns_none(store):
    store.save(SID_A, "acme", "125", b"p", "h", drive_file_id="drv-125")
    assert store.get_drive_file_id(SID_A, "acme", "999") is None


def test_get_drive_file_id_missing_session_returns_none(store):
    # No save → session dir doesn't exist.
    assert store.get_drive_file_id(SID_A, "acme", "125") is None


def test_get_drive_file_id_validates_inputs(store):
    with pytest.raises(ValueError, match="invalid session_id"):
        store.get_drive_file_id("not-a-uuid", "acme", "125")
    with pytest.raises(ValueError, match="invalid form_number"):
        store.get_drive_file_id(SID_A, "acme", "12")
    with pytest.raises(ValueError, match="invalid tenant"):
        store.get_drive_file_id(SID_A, "UPPERCASE", "125")


# --- set_drive_file_id ------------------------------------------------------

def test_set_drive_file_id_preserves_content_hash(store):
    store.save(SID_A, "acme", "125", b"p", "hash_abc")
    store.set_drive_file_id(SID_A, "acme", "125", "drv-xyz")

    assert store.get_drive_file_id(SID_A, "acme", "125") == "drv-xyz"
    # content_hash still visible via legacy manifest() accessor
    assert store.manifest(SID_A, "acme") == {"125": "hash_abc"}


def test_set_drive_file_id_creates_entry_when_form_missing(store):
    # No prior save.
    store.set_drive_file_id(SID_A, "acme", "125", "drv-only")

    raw = json.loads(store._manifest_path(SID_A, "acme").read_text())
    assert raw == {
        "125": {"content_hash": "", "drive_file_id": "drv-only"},
    }
    assert store.get_drive_file_id(SID_A, "acme", "125") == "drv-only"


def test_set_drive_file_id_validates_inputs(store):
    with pytest.raises(ValueError, match="invalid session_id"):
        store.set_drive_file_id("bad", "acme", "125", "drv")
    with pytest.raises(ValueError, match="invalid form_number"):
        store.set_drive_file_id(SID_A, "acme", "abc", "drv")
    with pytest.raises(ValueError, match="invalid tenant"):
        store.set_drive_file_id(SID_A, "Bad Slug", "125", "drv")


# --- save() with drive_file_id ---------------------------------------------

def test_save_with_drive_file_id_records_it(store):
    store.save(SID_A, "acme", "125", b"p1", "h1", drive_file_id="drv-1")

    raw = json.loads(store._manifest_path(SID_A, "acme").read_text())
    assert raw == {
        "125": {"content_hash": "h1", "drive_file_id": "drv-1"},
    }
    assert store.get_drive_file_id(SID_A, "acme", "125") == "drv-1"


def test_save_dedup_hit_updates_drive_file_id(store):
    # First save — bytes + original Drive ID.
    assert store.save(
        SID_A, "acme", "125", b"same-bytes", "same_hash",
        drive_file_id="drv-original",
    ) is True

    # Second save with identical bytes/hash but NEW Drive ID (re-upload).
    # Bytes stay put (dedup hit → returns False), but the new ID persists.
    assert store.save(
        SID_A, "acme", "125", b"same-bytes", "same_hash",
        drive_file_id="drv-updated",
    ) is False

    assert store.get_drive_file_id(SID_A, "acme", "125") == "drv-updated"


def test_save_dedup_hit_without_drive_file_id_preserves_existing(store):
    store.save(SID_A, "acme", "125", b"p", "h", drive_file_id="drv-keep")
    # Dedup hit, no drive_file_id supplied → must not clear.
    assert store.save(SID_A, "acme", "125", b"p", "h") is False
    assert store.get_drive_file_id(SID_A, "acme", "125") == "drv-keep"


def test_save_new_content_with_drive_file_id_after_dedup(store):
    # v1.
    store.save(SID_A, "acme", "125", b"v1", "hv1", drive_file_id="drv-v1")
    # v2: new bytes, new hash, new Drive ID.
    assert store.save(
        SID_A, "acme", "125", b"v2", "hv2", drive_file_id="drv-v2",
    ) is True

    assert store.load(SID_A, "acme", "125") == b"v2"
    raw = json.loads(store._manifest_path(SID_A, "acme").read_text())
    assert raw == {
        "125": {"content_hash": "hv2", "drive_file_id": "drv-v2"},
    }


def test_save_dedup_hit_same_drive_file_id_is_noop(store):
    """A dedup hit with the same drive_file_id should not rewrite manifest.

    Not strictly in the spec, but tangential: re-writing manifest on every
    identical /complete call would churn the file pointlessly.
    """
    store.save(SID_A, "acme", "125", b"p", "h", drive_file_id="drv-1")
    mtime_before = store._manifest_path(SID_A, "acme").stat().st_mtime_ns
    # Identical call — should neither write PDF nor rewrite manifest.
    assert store.save(
        SID_A, "acme", "125", b"p", "h", drive_file_id="drv-1",
    ) is False
    mtime_after = store._manifest_path(SID_A, "acme").stat().st_mtime_ns
    assert mtime_before == mtime_after


# --- Tenant isolation for Drive IDs ----------------------------------------

def test_get_drive_file_id_tenant_isolated(store):
    store.save(SID_A, "acme",   "125", b"p", "h", drive_file_id="drv-acme")
    store.save(SID_A, "globex", "125", b"p", "h", drive_file_id="drv-globex")

    assert store.get_drive_file_id(SID_A, "acme",   "125") == "drv-acme"
    assert store.get_drive_file_id(SID_A, "globex", "125") == "drv-globex"
    # Unknown tenant on a known form: isolated, returns None.
    assert store.get_drive_file_id(SID_A, "other", "125") is None


# --- On-disk shape ----------------------------------------------------------

def test_manifest_on_disk_uses_new_shape(store):
    store.save(SID_A, "acme", "125", b"p", "h125", drive_file_id="drv-125")
    store.save(SID_A, "acme", "126", b"p", "h126")  # no Drive ID

    raw = json.loads(store._manifest_path(SID_A, "acme").read_text())
    assert raw == {
        "125": {"content_hash": "h125", "drive_file_id": "drv-125"},
        "126": {"content_hash": "h126", "drive_file_id": None},
    }
    # Every value is a dict (not a legacy string).
    assert all(isinstance(v, dict) for v in raw.values())


def test_manifest_json_sorted_by_form_number(store):
    """Stable on-disk ordering — reproducible diffs if checked in."""
    store.save(SID_A, "acme", "163", b"p", "h163", drive_file_id="d163")
    store.save(SID_A, "acme", "125", b"p", "h125", drive_file_id="d125")
    store.save(SID_A, "acme", "130", b"p", "h130")

    text = store._manifest_path(SID_A, "acme").read_text()
    data = json.loads(text)
    assert list(data.keys()) == sorted(data.keys())


# --- Legacy manifest() accessor still returns {form: hash} -----------------

def test_manifest_accessor_returns_legacy_shape_for_back_compat(store):
    """The existing manifest() accessor must keep returning {form: hash}
    so callers from before P10.C.7 keep working unchanged."""
    store.save(SID_A, "acme", "125", b"p", "h125", drive_file_id="drv-1")
    store.save(SID_A, "acme", "126", b"p", "h126")

    m = store.manifest(SID_A, "acme")
    assert m == {"125": "h125", "126": "h126"}
    # Defensive copy.
    m["999"] = "mutated"
    assert store.manifest(SID_A, "acme") == {"125": "h125", "126": "h126"}
