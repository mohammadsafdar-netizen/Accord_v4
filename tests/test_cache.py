"""Tests for accord_ai.cache — TTL caches and CompleteCache."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from accord_ai.cache import (
    AuthBundle,
    CompleteCache,
    TenantAuth,
    _TTLDict,
    clear_cache,
    get_complete_cache,
    hash_bytes,
    hash_entities,
    ttl_cached,
)


# ---------------------------------------------------------------------------
# _TTLDict unit tests
# ---------------------------------------------------------------------------

def test_ttldict_basic_get_set():
    d = _TTLDict(maxsize=10, ttl=60.0)
    d["k"] = "v"
    assert d.get("k") == "v"


def test_ttldict_miss_returns_default():
    d = _TTLDict(maxsize=10, ttl=60.0)
    assert d.get("missing") is None
    assert d.get("missing", "default") == "default"


def test_ttldict_expiry(monkeypatch):
    d = _TTLDict(maxsize=10, ttl=10.0)
    d["k"] = "v"
    # Advance monotonic clock past TTL
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 20.0)
    assert d.get("k") is None


def test_ttldict_pop():
    d = _TTLDict(maxsize=10, ttl=60.0)
    d["k"] = "v"
    assert d.pop("k") == "v"
    assert d.get("k") is None


def test_ttldict_clear():
    d = _TTLDict(maxsize=10, ttl=60.0)
    d["a"] = 1
    d["b"] = 2
    d.clear()
    assert d.get("a") is None
    assert d.get("b") is None


# ---------------------------------------------------------------------------
# ttl_cached decorator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttl_hit():
    """Second call within TTL returns cached value without invoking fn."""
    call_count = 0

    @ttl_cached(ttl_seconds=60.0, key=lambda x: x)
    async def fetch(x: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result:{x}"

    r1 = await fetch("a")
    r2 = await fetch("a")

    assert r1 == r2 == "result:a"
    assert call_count == 1, f"expected 1 call, got {call_count}"


@pytest.mark.asyncio
async def test_ttl_expiry(monkeypatch):
    """After TTL expires the underlying function is called again."""
    call_count = 0

    @ttl_cached(ttl_seconds=10.0, key=lambda x: x)
    async def fetch(x: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result:{call_count}"

    r1 = await fetch("a")
    assert call_count == 1

    # Advance monotonic clock past TTL
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 20.0)

    r2 = await fetch("a")
    assert call_count == 2, f"expected 2 calls after expiry, got {call_count}"
    assert r2 != r1


@pytest.mark.asyncio
async def test_different_keys_isolated():
    """Cache entries for different keys don't collide."""
    call_count: dict = {}

    @ttl_cached(ttl_seconds=60.0, key=lambda x: x)
    async def fetch(x: str) -> str:
        call_count[x] = call_count.get(x, 0) + 1
        return f"result:{x}"

    await fetch("a")
    await fetch("b")
    await fetch("a")  # should hit cache for "a"

    assert call_count.get("a") == 1
    assert call_count.get("b") == 1


@pytest.mark.asyncio
async def test_cache_clear():
    """clear_cache() evicts all entries; subsequent call re-invokes function."""
    call_count = 0

    @ttl_cached(ttl_seconds=60.0, key=lambda x: x)
    async def fetch(x: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result:{x}"

    await fetch("a")
    assert call_count == 1

    # Manually clear the decorator's internal cache dict
    fetch._ttl_cache.clear()

    await fetch("a")
    assert call_count == 2, "expected re-invocation after cache clear"


# ---------------------------------------------------------------------------
# CompleteCache
# ---------------------------------------------------------------------------

def test_complete_cache_tenant_auth_roundtrip():
    cache = CompleteCache()
    auth = TenantAuth(
        service_token="tok",
        drive_token="dtok",
        lob_folder_id="folder123",
        fetched_at=time.time(),
    )
    cache.set_tenant_auth("acme", auth)
    result = cache.get_tenant_auth("acme")
    assert result is auth


def test_complete_cache_auth_bundle_synthesize():
    cache = CompleteCache()
    bundle = AuthBundle(
        service_token="tok",
        drive_token="dtok",
        lob_folder_id="folder123",
        sub_folder_id="sub456",
        fetched_at=time.time(),
    )
    cache.set_auth("acme", "sub1", bundle)
    result = cache.get_auth("acme", "sub1")
    assert result is not None
    assert result.service_token == "tok"
    assert result.sub_folder_id == "sub456"


def test_complete_cache_invalidate():
    cache = CompleteCache()
    ta = TenantAuth("t", "d", "f", time.time())
    cache.set_tenant_auth("acme", ta)
    cache.invalidate_tenant_auth("acme")
    assert cache.get_tenant_auth("acme") is None


def test_complete_cache_file_id():
    cache = CompleteCache()
    cache.set_file_id("sub1", "pdf1", "drive_file_abc")
    assert cache.get_file_id("sub1", "pdf1") == "drive_file_abc"
    cache.invalidate_file_id("sub1", "pdf1")
    assert cache.get_file_id("sub1", "pdf1") is None


def test_complete_cache_clear():
    cache = CompleteCache()
    ta = TenantAuth("t", "d", "f", time.time())
    cache.set_tenant_auth("acme", ta)
    cache.set_file_id("sub1", "pdf1", "fid")
    cache.clear()
    assert cache.get_tenant_auth("acme") is None
    assert cache.get_file_id("sub1", "pdf1") is None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_complete_cache_singleton():
    c1 = get_complete_cache()
    c2 = get_complete_cache()
    assert c1 is c2


def test_clear_cache_helper():
    """clear_cache() empties the process-wide singleton stores."""
    cache = get_complete_cache()
    ta = TenantAuth("t", "d", "f", time.time())
    cache.set_tenant_auth("tenant-x", ta)

    clear_cache()

    assert cache.get_tenant_auth("tenant-x") is None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def test_hash_entities_stable():
    h1 = hash_entities({"a": 1, "b": 2})
    h2 = hash_entities({"b": 2, "a": 1})
    assert h1 == h2


def test_hash_bytes():
    assert len(hash_bytes(b"hello")) == 64
