"""Tests for accord_ai.http_client — shared AsyncClient singleton."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

import accord_ai.http_client as http_mod
from accord_ai.http_client import _reset_for_tests, close_client, get_client


@pytest.fixture(autouse=True)
def reset_singleton():
    """Drop the shared client before and after every test."""
    _reset_for_tests()
    yield
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Singleton identity
# ---------------------------------------------------------------------------

def test_singleton_identity():
    """Two calls to get_client() must return the exact same object."""
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2


# ---------------------------------------------------------------------------
# Transport reuse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_reused_across_requests():
    """Requests made via get_client() share the same transport instance."""
    transport = httpx.MockTransport(handler=lambda req: httpx.Response(200))
    original_create = http_mod._CLIENT  # None at this point

    # Patch the client creation to inject a MockTransport
    real_client = httpx.AsyncClient(transport=transport)
    with patch.object(http_mod, "_CLIENT", real_client):
        c1 = get_client()
        c2 = get_client()
        assert c1 is c2
        assert c1._transport is transport

    await real_client.aclose()


# ---------------------------------------------------------------------------
# Lifespan close
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_close():
    """close_client() shuts down the shared client and clears the singleton."""
    c = get_client()
    assert http_mod._CLIENT is c

    closed = []
    original_aclose = c.aclose

    async def tracked_aclose():
        closed.append(True)
        await original_aclose()

    c.aclose = tracked_aclose  # type: ignore[method-assign]

    await close_client()

    assert closed, "aclose() was not called on the shared client"
    assert http_mod._CLIENT is None, "singleton was not cleared after close"


@pytest.mark.asyncio
async def test_close_client_idempotent():
    """Calling close_client() twice must not raise."""
    get_client()
    await close_client()
    await close_client()  # second call: _CLIENT is already None, must be a no-op


def test_close_client_before_create():
    """close_client() before any get_client() call must not raise."""
    import asyncio
    asyncio.get_event_loop().run_until_complete(close_client())
