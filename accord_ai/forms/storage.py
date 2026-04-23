"""Filesystem storage for filled PDFs, per-tenant, per-session.

Layout:
    {root}/
        {tenant or _no_tenant}/
            {session_id}/
                manifest.json                    # see shape below
                acord_{form_number}_filled.pdf

Manifest shape (current):
    {
        "125": {"content_hash": "sha256...", "drive_file_id": "abc123"},
        "126": {"content_hash": "sha256...", "drive_file_id": null}
    }

Manifest shape (legacy, pre P10.C.7):
    {"125": "sha256...", "126": "sha256..."}

Legacy manifests are auto-promoted in-memory on read (string → dict with
drive_file_id=None); the new shape is written to disk only on the next
_write_manifest call (so a read-only tool can still inspect legacy data).

manifest.json is the dedup key: a repeat fill_submission with identical
submission data produces identical hashes, which compare equal to the
manifest entry, so we skip the disk write entirely. (Byte-stability is
enforced upstream by PyMuPDF's no_new_id=True — see forms.filler.)

Tenant isolation is path-level, enforced the same way SessionStore
enforces it at row level: caller MUST pass tenant on every read/write.
No global escape hatch, no cross-tenant listing. An attacker with a
guessed session_id from tenant A cannot read tenant B's PDFs.

One caveat: a literal tenant string "_no_tenant" would collide with the
None-tenant bucket. That value is rejected upfront.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from accord_ai.logging_config import get_logger

_logger = get_logger("forms.storage")

_NO_TENANT_DIR = "_no_tenant"

# Session IDs are 32-hex UUIDs per SessionStore. Form numbers are 3-4 digits.
# Both validated here so we never join user input into a filesystem path
# without checking shape first — defense in depth against traversal.
_SESSION_ID_RE  = re.compile(r"^[a-f0-9]{32}$")
_FORM_NUMBER_RE = re.compile(r"^\d{3,4}$")
_TENANT_RE      = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _tenant_dir_name(tenant: Optional[str]) -> str:
    if tenant is None:
        return _NO_TENANT_DIR
    if tenant == _NO_TENANT_DIR:
        raise ValueError(
            f"tenant slug {_NO_TENANT_DIR!r} is reserved — would collide "
            "with the no-tenant bucket"
        )
    if not _TENANT_RE.match(tenant):
        raise ValueError(f"invalid tenant slug: {tenant!r}")
    return tenant


def _validated_session(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return session_id


def _validated_form(form_number: str) -> str:
    if not _FORM_NUMBER_RE.match(form_number):
        raise ValueError(f"invalid form_number: {form_number!r}")
    return form_number


def _normalize_entry(value: Any) -> Optional[Dict[str, Optional[str]]]:
    """Promote a raw manifest value to the normalized dict shape.

    - str → {"content_hash": <str>, "drive_file_id": None} (legacy auto-promote)
    - dict with content_hash → normalized dict (coerces strings, keeps None)
    - anything else → None (caller treats as missing/corrupt)
    """
    if isinstance(value, str):
        return {"content_hash": value, "drive_file_id": None}
    if isinstance(value, dict):
        ch = value.get("content_hash")
        did = value.get("drive_file_id")
        return {
            "content_hash": "" if ch is None else str(ch),
            "drive_file_id": None if did is None else str(did),
        }
    return None


class FilledPdfStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # --- Paths --------------------------------------------------------------

    def _session_dir(self, session_id: str, tenant: Optional[str]) -> Path:
        return (
            self._root
            / _tenant_dir_name(tenant)
            / _validated_session(session_id)
        )

    def _pdf_path(
        self, session_id: str, tenant: Optional[str], form_number: str,
    ) -> Path:
        return (
            self._session_dir(session_id, tenant)
            / f"acord_{_validated_form(form_number)}_filled.pdf"
        )

    def _manifest_path(self, session_id: str, tenant: Optional[str]) -> Path:
        return self._session_dir(session_id, tenant) / "manifest.json"

    # --- Manifest -----------------------------------------------------------

    def _read_manifest(
        self, session_id: str, tenant: Optional[str],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Load manifest, auto-promoting legacy string values in-memory.

        We do NOT rewrite legacy manifests on read — tools that only read
        (e.g. inspection scripts) shouldn't cause surprise writes. The next
        _write_manifest call (triggered by a save/set_drive_file_id) will
        persist the new shape.

        A single INFO log fires per legacy detection so ops can track the
        rolling migration of existing sessions to the new shape.
        """
        p = self._manifest_path(session_id, tenant)
        if not p.is_file():
            return {}
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning(
                "manifest unreadable — treating as empty (session=%s err=%s)",
                session_id, exc,
            )
            return {}

        if not isinstance(raw, dict):
            return {}

        out: Dict[str, Dict[str, Optional[str]]] = {}
        legacy_detected = False
        for k, v in raw.items():
            entry = _normalize_entry(v)
            if entry is None:
                continue
            if isinstance(v, str):
                legacy_detected = True
            out[str(k)] = entry

        if legacy_detected:
            _logger.info(
                "legacy manifest auto-promoted in-memory "
                "(session=%s forms=%d) — new shape persists on next write",
                session_id, len(out),
            )
        return out

    def _write_manifest(
        self,
        session_id: str,
        tenant: Optional[str],
        manifest: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        p = self._manifest_path(session_id, tenant)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to tmp + rename. Manifest drives dedup, so
        # a partial write here would cause phantom "no change" dedups.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        )
        tmp.replace(p)

    def _manifest_entry(
        self,
        session_id: str,
        tenant: Optional[str],
        form_number: str,
    ) -> Optional[Dict[str, Optional[str]]]:
        """Return the normalized entry for a form, or None if missing.

        Pure read — does not rewrite the on-disk manifest even when the
        source was legacy. Intended for internal callers that want the
        auto-promoted view without triggering persistence.
        """
        _validated_form(form_number)
        manifest = self._read_manifest(session_id, tenant)
        return manifest.get(form_number)

    # --- Public API ---------------------------------------------------------

    def save(
        self,
        session_id: str,
        tenant: Optional[str],
        form_number: str,
        pdf_bytes: bytes,
        content_hash: str,
        *,
        drive_file_id: Optional[str] = None,
    ) -> bool:
        """Write the PDF iff content_hash differs from the manifest entry.

        Returns True if written, False if skipped by dedup. A fresh session
        with no manifest yet always writes (miss → write). If the manifest
        points at a hash but the on-disk PDF is missing (out-of-band delete),
        we re-write — don't silently dedup to nothing.

        drive_file_id semantics (why the branch below is not symmetric):

        - Dedup miss (content changed): record content_hash AND drive_file_id
          (even if None — the new content has no known Drive ID yet).
        - Dedup hit (content unchanged): do NOT rewrite bytes, but DO update
          the stored drive_file_id if a non-None value is supplied. The common
          case is a second /complete on the same submission: bytes are stable
          but a fresh Drive upload just produced a new file_id we need to
          persist. A None drive_file_id on a dedup hit preserves the existing
          stored value — so callers that don't know the ID can no-op safely.
        """
        manifest = self._read_manifest(session_id, tenant)
        existing = manifest.get(form_number)
        existing_hash = existing["content_hash"] if existing else None
        existing_drive = existing["drive_file_id"] if existing else None

        pdf_path = self._pdf_path(session_id, tenant, form_number)
        dedup_hit = (
            existing_hash == content_hash
            and pdf_path.is_file()
        )

        if dedup_hit:
            _logger.debug(
                "dedup hit: session=%s form=%s hash=%s",
                session_id, form_number, content_hash[:12],
            )
            # Persist a new drive_file_id if one was supplied and differs.
            # Preserve the existing ID when caller passes None.
            if drive_file_id is not None and drive_file_id != existing_drive:
                manifest[form_number] = {
                    "content_hash": content_hash,
                    "drive_file_id": drive_file_id,
                }
                self._write_manifest(session_id, tenant, manifest)
            return False

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_bytes)

        manifest[form_number] = {
            "content_hash": content_hash,
            "drive_file_id": drive_file_id,
        }
        self._write_manifest(session_id, tenant, manifest)
        _logger.debug(
            "wrote: session=%s form=%s bytes=%d",
            session_id, form_number, len(pdf_bytes),
        )
        return True

    def load(
        self, session_id: str, tenant: Optional[str], form_number: str,
    ) -> Optional[bytes]:
        """Return PDF bytes or None if absent. Tenant-scoped — wrong tenant
        is indistinguishable from missing, by design."""
        p = self._pdf_path(session_id, tenant, form_number)
        if not p.is_file():
            return None
        return p.read_bytes()

    def list_forms(
        self, session_id: str, tenant: Optional[str],
    ) -> List[str]:
        """Form numbers present in the manifest, sorted numerically.

        Returns [] for unknown session or unknown tenant — uniform surface,
        same as SessionStore.get_session(wrong_tenant) → None.
        """
        manifest = self._read_manifest(session_id, tenant)
        return sorted(manifest.keys(), key=lambda s: (len(s), s))

    def manifest(
        self, session_id: str, tenant: Optional[str],
    ) -> Dict[str, str]:
        """Full {form_number: content_hash} map — defensive copy.

        NOTE: preserves the legacy return shape (form → hash) so existing
        callers keep working. For the Drive ID, use get_drive_file_id.
        """
        return {
            k: ("" if v["content_hash"] is None else str(v["content_hash"]))
            for k, v in self._read_manifest(session_id, tenant).items()
        }

    def get_drive_file_id(
        self, session_id: str, tenant: Optional[str], form_number: str,
    ) -> Optional[str]:
        """Return the stored Drive file ID for (session, tenant, form).

        Returns None when:
          - session/tenant unknown
          - form has no manifest entry
          - entry is legacy (pre P10.C.7) so no drive_file_id was stored
          - entry exists but drive_file_id is explicitly null

        Validates session_id / form_number / tenant using the same regex
        helpers as save() so a bad input fails loudly instead of silently
        returning None.
        """
        _validated_session(session_id)
        _validated_form(form_number)
        _tenant_dir_name(tenant)  # raises on reserved/invalid slugs
        entry = self._manifest_entry(session_id, tenant, form_number)
        if entry is None:
            return None
        return entry["drive_file_id"]

    def set_drive_file_id(
        self,
        session_id: str,
        tenant: Optional[str],
        form_number: str,
        drive_file_id: str,
    ) -> None:
        """Persist drive_file_id for a form, preserving content_hash.

        If the form has no manifest entry yet (e.g. Drive upload completed
        before the local save() call for some reason — we still want to
        remember the ID), a fresh entry is created with content_hash=""
        (empty-string sentinel meaning "hash unknown"). The next real save()
        will then overwrite content_hash with the actual value and keep
        this drive_file_id only if the caller re-supplies it — see save()'s
        dedup-miss branch which records (hash, provided_drive_id_or_None).

        Validates session_id / form_number / tenant.
        """
        _validated_session(session_id)
        _validated_form(form_number)
        _tenant_dir_name(tenant)
        manifest = self._read_manifest(session_id, tenant)
        existing = manifest.get(form_number)
        if existing is None:
            manifest[form_number] = {
                "content_hash": "",
                "drive_file_id": drive_file_id,
            }
        else:
            manifest[form_number] = {
                "content_hash": existing["content_hash"] or "",
                "drive_file_id": drive_file_id,
            }
        self._write_manifest(session_id, tenant, manifest)

    def clear_session(
        self, session_id: str, tenant: Optional[str],
    ) -> None:
        """Wipe a session's directory. Idempotent."""
        d = self._session_dir(session_id, tenant)
        if d.is_dir():
            shutil.rmtree(d)
