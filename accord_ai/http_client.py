"""Process-wide shared httpx.AsyncClient — port of accord_ai_v3/accord_ai/http_client.py.

One pooled client replaces the previous per-call
`async with httpx.AsyncClient(...)` pattern. Connection reuse avoids
TLS handshake on every Drive / backend hop.

Lifecycle:
  - get_client() creates on first call (double-checked lock, thread-safe).
  - close_client() is called from the FastAPI lifespan on shutdown.
  - _reset_for_tests() drops the singleton between test cases.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import httpx

_CLIENT: Optional[httpx.AsyncClient] = None
_CLIENT_LOCK = threading.Lock()


def _tls_verify() -> bool:
    raw = os.environ.get("BACKEND_TLS_VERIFY", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient; create on first use."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=5.0, read=15.0, write=10.0, pool=5.0,
                    ),
                    verify=_tls_verify(),
                    limits=httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=30,
                    ),
                )
    return _CLIENT


async def close_client() -> None:
    """Close the shared client if created. Idempotent."""
    global _CLIENT
    if _CLIENT is not None:
        c = _CLIENT
        _CLIENT = None
        try:
            await c.aclose()
        except Exception:
            pass


def _reset_for_tests() -> None:
    """Test-only: drop the cached client so a fresh one is built."""
    global _CLIENT
    _CLIENT = None
