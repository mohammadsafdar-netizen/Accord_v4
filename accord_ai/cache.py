"""Process-wide TTL caches — port of accord_ai_v3/accord_ai/cache.py.

Bounded TTL caches for auth bundles, Drive file IDs, content hashes, and
validation reports. Thread-safe mutations, async per-submission locks.

Changes from v3:
  - cachetools.TTLCache replaced with a minimal _TTLDict (same interface,
    no extra dependency).
  - ttl_cached() async decorator added for wrapping standalone functions.
  - clear_cache() helper for test teardown.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Internal TTL dict (replaces cachetools.TTLCache)
# ---------------------------------------------------------------------------

class _TTLDict:
    """Minimal TTL-aware dict. Thread-unsafe — callers hold a lock."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: Dict[Any, tuple] = {}  # key -> (value, expires_at)

    def get(self, key: Any, default: Any = None) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return default
        value, exp = entry
        if time.monotonic() > exp:
            del self._store[key]
            return default
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if len(self._store) >= self._maxsize and key not in self._store:
            try:
                oldest = next(iter(self._store))
                del self._store[oldest]
            except StopIteration:
                pass
        self._store[key] = (value, time.monotonic() + self._ttl)

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def pop(self, key: Any, default: Any = None) -> Any:
        entry = self._store.pop(key, None)
        if entry is None:
            return default
        return entry[0]

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# ttl_cached decorator
# ---------------------------------------------------------------------------

def ttl_cached(
    ttl_seconds: float,
    key: Callable[..., Any],
) -> Callable:
    """Async function decorator that caches results by a computed key.

    Args:
        ttl_seconds: How long a cached result stays valid.
        key: Callable(*args, **kwargs) -> hashable cache key. Receives the
             same arguments as the decorated function.

    Usage::

        @ttl_cached(ttl_seconds=1800, key=lambda tenant, domain: tenant)
        async def get_service_token(tenant: str, domain: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        _cache: Dict[Any, tuple] = {}  # cache_key -> (result, expires_at)
        _lock = threading.Lock()

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = key(*args, **kwargs)
            now = time.monotonic()
            with _lock:
                entry = _cache.get(cache_key)
                if entry is not None and now < entry[1]:
                    return entry[0]
            result = await fn(*args, **kwargs)
            with _lock:
                _cache[cache_key] = (result, time.monotonic() + ttl_seconds)
            return result

        wrapper._ttl_cache = _cache  # exposed for clear_cache()
        wrapper._ttl_lock = _lock
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Dataclasses (v3 parity)
# ---------------------------------------------------------------------------

@dataclass
class TenantAuth:
    service_token: str
    drive_token: str
    lob_folder_id: str
    fetched_at: float


@dataclass
class AuthBundle:
    service_token: str
    drive_token: str
    lob_folder_id: str
    sub_folder_id: str
    fetched_at: float


# ---------------------------------------------------------------------------
# CompleteCache
# ---------------------------------------------------------------------------

class CompleteCache:
    """Per-submission caching for /complete refresh path. Thread-safe."""

    _MAX_LOCKS = 1000

    def __init__(
        self,
        *,
        auth_ttl: float = 600.0,
        sub_folder_ttl: float = 3600.0,
        file_id_ttl: float = 3600.0,
        content_hash_ttl: float = 3600.0,
        validation_ttl: float = 300.0,
        auth_size: int = 500,
        sub_folder_size: int = 5000,
        file_id_size: int = 10000,
        content_hash_size: int = 10000,
        validation_size: int = 500,
    ) -> None:
        self._tenant_auth   = _TTLDict(maxsize=auth_size,          ttl=auth_ttl)
        self._sub_folders   = _TTLDict(maxsize=sub_folder_size,    ttl=sub_folder_ttl)
        self._file_ids      = _TTLDict(maxsize=file_id_size,       ttl=file_id_ttl)
        self._content_hashes = _TTLDict(maxsize=content_hash_size, ttl=content_hash_ttl)
        self._validation    = _TTLDict(maxsize=validation_size,     ttl=validation_ttl)
        self._mu = threading.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_mu = threading.Lock()

    # --- tenant-scoped auth (service_token, drive_token, lob_folder_id) ---

    def get_tenant_auth(self, tenant_slug: str) -> Optional[TenantAuth]:
        with self._mu:
            return self._tenant_auth.get(tenant_slug)

    def set_tenant_auth(self, tenant_slug: str, auth: TenantAuth) -> None:
        with self._mu:
            self._tenant_auth[tenant_slug] = auth

    def invalidate_tenant_auth(self, tenant_slug: str) -> None:
        with self._mu:
            self._tenant_auth.pop(tenant_slug, None)

    # --- submission-scoped sub_folder_id ---

    def get_sub_folder_id(
        self, tenant_slug: str, submission_id: str,
    ) -> Optional[str]:
        key = (tenant_slug, submission_id)
        with self._mu:
            return self._sub_folders.get(key)

    def set_sub_folder_id(
        self, tenant_slug: str, submission_id: str, folder_id: str,
    ) -> None:
        key = (tenant_slug, submission_id)
        with self._mu:
            self._sub_folders[key] = folder_id

    def invalidate_sub_folder_id(
        self, tenant_slug: str, submission_id: str,
    ) -> None:
        key = (tenant_slug, submission_id)
        with self._mu:
            self._sub_folders.pop(key, None)

    # --- auth bundle (backward-compat: synthesize from split halves) ---

    def get_auth(self, tenant_slug: str, submission_id: str) -> Optional[AuthBundle]:
        ta = self.get_tenant_auth(tenant_slug)
        sub = self.get_sub_folder_id(tenant_slug, submission_id)
        if ta is None or not sub:
            return None
        return AuthBundle(
            service_token=ta.service_token,
            drive_token=ta.drive_token,
            lob_folder_id=ta.lob_folder_id,
            sub_folder_id=sub,
            fetched_at=ta.fetched_at,
        )

    def set_auth(
        self, tenant_slug: str, submission_id: str, bundle: AuthBundle,
    ) -> None:
        self.set_tenant_auth(
            tenant_slug,
            TenantAuth(
                service_token=bundle.service_token,
                drive_token=bundle.drive_token,
                lob_folder_id=bundle.lob_folder_id,
                fetched_at=bundle.fetched_at,
            ),
        )
        self.set_sub_folder_id(tenant_slug, submission_id, bundle.sub_folder_id)

    def invalidate_auth(self, tenant_slug: str, submission_id: str) -> None:
        self.invalidate_tenant_auth(tenant_slug)
        self.invalidate_sub_folder_id(tenant_slug, submission_id)

    # --- file_id ---

    def get_file_id(self, submission_id: str, pdf_id: str) -> Optional[str]:
        key = (submission_id, pdf_id)
        with self._mu:
            return self._file_ids.get(key)

    def set_file_id(self, submission_id: str, pdf_id: str, file_id: str) -> None:
        key = (submission_id, pdf_id)
        with self._mu:
            self._file_ids[key] = file_id

    def invalidate_file_id(self, submission_id: str, pdf_id: str) -> None:
        key = (submission_id, pdf_id)
        with self._mu:
            self._file_ids.pop(key, None)

    # --- content hash ---

    def get_content_hash(self, submission_id: str, pdf_id: str) -> Optional[str]:
        key = (submission_id, pdf_id)
        with self._mu:
            return self._content_hashes.get(key)

    def set_content_hash(
        self, submission_id: str, pdf_id: str, sha256: str,
    ) -> None:
        key = (submission_id, pdf_id)
        with self._mu:
            self._content_hashes[key] = sha256

    # --- validation report ---

    def get_validation(self, entities_hash: str) -> Optional[dict]:
        with self._mu:
            return self._validation.get(entities_hash)

    def set_validation(self, entities_hash: str, report: dict) -> None:
        with self._mu:
            self._validation[entities_hash] = report

    # --- per-submission asyncio.Lock ---

    def get_lock(self, submission_id: str) -> asyncio.Lock:
        with self._locks_mu:
            lock = self._locks.get(submission_id)
            if lock is not None:
                return lock
            if len(self._locks) >= self._MAX_LOCKS:
                self._evict_unlocked()
            lock = asyncio.Lock()
            self._locks[submission_id] = lock
            return lock

    def _evict_unlocked(self) -> None:
        try:
            stale = [sid for sid, lk in self._locks.items() if not lk.locked()]
            for sid in stale:
                self._locks.pop(sid, None)
                if len(self._locks) < self._MAX_LOCKS:
                    break
        except Exception:
            pass

    def clear(self) -> None:
        """Evict all cached entries. Used in test teardown."""
        with self._mu:
            self._tenant_auth.clear()
            self._sub_folders.clear()
            self._file_ids.clear()
            self._content_hashes.clear()
            self._validation.clear()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def hash_entities(entities: dict) -> str:
    """Stable sha256 hex of an entities dict."""
    payload = json.dumps(entities, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_bytes(data: bytes) -> str:
    """sha256 hex of bytes."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton: Optional[CompleteCache] = None
_singleton_lock = threading.Lock()


def get_complete_cache() -> CompleteCache:
    """Process-wide CompleteCache singleton (double-checked locking)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CompleteCache()
    return _singleton


def clear_cache() -> None:
    """Evict all entries from the process-wide cache. For test teardown."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.clear()
