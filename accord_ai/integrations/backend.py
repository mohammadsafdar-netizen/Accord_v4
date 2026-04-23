"""Insurance Backend (copilot.inevo.ai) client.

Responsibilities in this step:
  * Derive the per-tenant base URL from a slug or a tenant domain, with
    SSRF defenses (no localhost, no private IPs, no malformed hosts).
  * Exchange client credentials for a service JWT (OAuth2 service-token
    endpoint), caching the result in-memory for `backend_token_ttl_s`.

Not yet: push_fields / read_fields / status (land in 10.C.2) and any
Drive operations (10.C.3+).

Design:
  * httpx.AsyncClient is constructor-injected. Production code calls
    build_backend_client(settings), which wires a real client. Tests use
    httpx.MockTransport and instantiate BackendClient directly.
  * All exceptions are swallowed and logged. Token methods return None
    on any failure — callers treat None as "backend unavailable" and
    degrade. Backend flakiness must never block the local /complete path.
"""
from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

import httpx

from accord_ai.http_client import get_client as _get_http_client

from accord_ai.audit import record_audit_event
from accord_ai.config import Settings
from accord_ai.core.store import SessionStore
from accord_ai.logging_config import get_logger

_logger = get_logger("integrations.backend")


# ---------------------------------------------------------------------------
# Tenant-URL derivation + SSRF guard
# ---------------------------------------------------------------------------

# First label of the tenant reference is the canonical slug.
# Accepts: "acme", "acme.copilot.inevo.ai", "acme.brocopilot.com"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# Full-domain shape check. Length cap from DNS RFC (254 incl. trailing dot).
_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.\-]{1,253}$")

_BLOCKED_LITERALS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]",
})


class BackendSSRFError(ValueError):
    """Raised when a tenant domain/slug would point the backend call at a
    private, loopback, or otherwise unsafe host. Surfaces as `None` to the
    caller via get_service_token — never as a raw exception."""


def _slug_from(tenant_ref: str) -> str:
    if not tenant_ref:
        return ""
    return tenant_ref.split(".", 1)[0].strip().lower()


def _validate_tenant_ref(tenant_ref: str) -> str:
    """Return the normalized slug, or raise BackendSSRFError."""
    ref = (tenant_ref or "").strip().lower()
    if not ref:
        raise BackendSSRFError("empty tenant reference")
    if ref in _BLOCKED_LITERALS:
        raise BackendSSRFError(f"blocked host literal: {ref!r}")

    # If it looks like a bare IP, reject loopback / private / link-local.
    # Parse and classify in separate steps — BackendSSRFError inherits
    # from ValueError, so raising it inside the try/except would be
    # swallowed by the parse-failure branch.
    try:
        ip = ipaddress.ip_address(ref)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_loopback or ip.is_private
        or ip.is_link_local or ip.is_reserved
    ):
        raise BackendSSRFError(f"blocked IP: {ref!r}")

    if "." in ref and not _DOMAIN_RE.match(ref):
        raise BackendSSRFError(f"malformed domain: {ref!r}")

    slug = _slug_from(ref)
    if not _SLUG_RE.match(slug):
        raise BackendSSRFError(f"invalid tenant slug: {slug!r}")
    return slug


def backend_base_url(tenant_ref: str, settings: Settings) -> str:
    """Build https://{slug}.{suffix}. Raises BackendSSRFError on bad input."""
    slug = _validate_tenant_ref(tenant_ref)
    return f"https://{slug}.{settings.backend_host_suffix}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class _TokenEntry:
    token: str
    expires_at: float       # unix epoch seconds


class BackendClient:
    """Thin OAuth2-ish client. Safe to share across asyncio tasks: the
    token cache is guarded by a std-library Lock (cache ops are nanoseconds
    — a Lock is simpler than asyncio.Lock and cross-context safe)."""

    def __init__(
        self,
        *,
        settings: Settings,
        http: httpx.AsyncClient,
        audit_store: Optional[SessionStore] = None,
    ) -> None:
        self._settings = settings
        self._http = http
        self._audit_store = audit_store

        self._token_cache: Dict[str, _TokenEntry] = {}
        self._token_lock = Lock()

    async def aclose(self) -> None:
        pass  # shared client lifecycle is managed by the FastAPI lifespan

    # --- URL helpers exposed for later steps ---

    def base_url(self, tenant_ref: str) -> str:
        return backend_base_url(tenant_ref, self._settings)

    # --- Service token ---

    async def get_service_token(
        self, tenant_slug: str, tenant_domain: str,
    ) -> Optional[str]:
        """Return a cached service JWT for the tenant, or None on any failure.

        Path:   POST {base_url}/api/v1/auth/oauth/service-token
        Body:   {client_id, client_secret, tenant_slug}
        Header: X-Forwarded-Host: {tenant_domain}
        """
        if not self._settings.backend_enabled:
            return None
        secret = self._settings.backend_client_secret
        if secret is None:
            _logger.warning(
                "backend: disabled — BACKEND_CLIENT_SECRET not set",
            )
            return None

        cache_key = tenant_slug or tenant_domain
        now = time.time()
        with self._token_lock:
            entry = self._token_cache.get(cache_key)
            if entry is not None and now < entry.expires_at:
                return entry.token

        try:
            base = self.base_url(tenant_slug or tenant_domain)
        except BackendSSRFError as exc:
            _logger.warning("backend: %s", exc)
            self._audit(
                "backend.token_issued", tenant_slug,
                {"status": "ssrf_blocked"},
            )
            return None

        try:
            resp = await self._http.post(
                f"{base}/api/v1/auth/oauth/service-token",
                json={
                    "client_id":     self._settings.backend_client_id,
                    "client_secret": secret.get_secret_value(),
                    "tenant_slug":   tenant_slug,
                },
                headers={"x-forwarded-host": tenant_domain},
                timeout=self._settings.backend_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("backend: token request failed err=%s", exc)
            self._audit("backend.token_issued", tenant_slug, {
                "status": "error", "error_class": type(exc).__name__,
            })
            return None

        if resp.status_code != 200:
            _logger.warning("backend: token http=%s", resp.status_code)
            self._audit("backend.token_issued", tenant_slug, {
                "status": "http_error", "http_status": resp.status_code,
            })
            return None

        try:
            token = resp.json().get("access_token")
        except ValueError:
            token = None
        if not isinstance(token, str) or not token:
            _logger.warning("backend: token response missing access_token")
            self._audit(
                "backend.token_issued", tenant_slug,
                {"status": "missing_token"},
            )
            return None

        # ttl=0 means "disable cache": entry expires immediately (now+0)
        # so the next call always refetches. No floor — honor the setting.
        ttl = int(self._settings.backend_token_ttl_s)
        with self._token_lock:
            self._token_cache[cache_key] = _TokenEntry(
                token=token, expires_at=now + ttl,
            )
        self._audit("backend.token_issued", tenant_slug, {
            "status": "ok", "ttl_s": ttl,
        })
        return token

    # --- Submission field sync (P10.C.2) ---

    async def push_fields(
        self,
        token: str,
        submission_id: str,
        tenant_domain: str,
        fields_data: dict,
        *,
        completion_percentage: float = 0.0,
        source: str = "chat_backend",
    ) -> bool:
        """POST submission fields to the backend. Returns True on 200/201.

        Non-fatal: any failure logs + audits + returns False. Callers treat
        False as "backend unavailable" and don't surface it to the user —
        extraction continues, we retry on next turn.
        """
        if not self._settings.backend_enabled:
            return False
        if not token:
            return False

        try:
            base = self.base_url(tenant_domain)
        except BackendSSRFError as exc:
            _logger.warning("backend.push_fields: %s", exc)
            self._audit(
                "backend.push_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status": "ssrf_blocked",
                },
            )
            return False

        body = {
            "fields_data":           fields_data,
            "source":                source,
            "completion_percentage": round(float(completion_percentage), 1),
        }
        try:
            resp = await self._http.post(
                f"{base}/api/v1/submissions/{submission_id}/fields",
                json=body,
                headers={
                    "authorization":    f"Bearer {token}",
                    "x-forwarded-host": tenant_domain,
                },
                timeout=self._settings.backend_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("backend.push_fields: network err=%s", exc)
            self._audit(
                "backend.push_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status": "error",
                    "error_class": type(exc).__name__,
                },
            )
            return False

        fields_count = (
            len(fields_data) if isinstance(fields_data, dict) else 0
        )
        if resp.status_code in (200, 201):
            self._audit(
                "backend.push_fields", _slug_from(tenant_domain), {
                    "submission_id":  submission_id,
                    "status":         "ok",
                    "http_status":    resp.status_code,
                    "fields_count":   fields_count,
                    "completion_pct": round(float(completion_percentage), 1),
                    "source":         source,
                },
            )
            return True

        _logger.warning("backend.push_fields: http=%s", resp.status_code)
        self._audit(
            "backend.push_fields", _slug_from(tenant_domain), {
                "submission_id": submission_id,
                "status":        "http_error",
                "http_status":   resp.status_code,
            },
        )
        return False

    async def read_fields(
        self,
        token: str,
        submission_id: str,
        tenant_domain: str,
    ) -> Optional[dict]:
        """GET existing submission fields from the backend.

        Returns the fields_data dict on success, None on any failure or
        when the response shape is unexpected. Callers use None → "no
        prior data to pre-populate", not "backend is broken" — safe to
        fall through to a fresh session.
        """
        if not self._settings.backend_enabled:
            return None
        if not token:
            return None

        try:
            base = self.base_url(tenant_domain)
        except BackendSSRFError as exc:
            _logger.warning("backend.read_fields: %s", exc)
            self._audit(
                "backend.read_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status": "ssrf_blocked",
                },
            )
            return None

        try:
            resp = await self._http.get(
                f"{base}/api/v1/submissions/{submission_id}/fields",
                headers={
                    "authorization":    f"Bearer {token}",
                    "x-forwarded-host": tenant_domain,
                },
                timeout=self._settings.backend_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("backend.read_fields: network err=%s", exc)
            self._audit(
                "backend.read_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "error",
                    "error_class":   type(exc).__name__,
                },
            )
            return None

        if resp.status_code != 200:
            _logger.warning(
                "backend.read_fields: http=%s", resp.status_code,
            )
            self._audit(
                "backend.read_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "http_error",
                    "http_status":   resp.status_code,
                },
            )
            return None

        try:
            payload = resp.json()
        except ValueError:
            _logger.warning("backend.read_fields: non-JSON response")
            self._audit(
                "backend.read_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "non_json",
                },
            )
            return None

        fields_data = (
            payload.get("fields_data") if isinstance(payload, dict) else None
        )
        if not isinstance(fields_data, dict):
            self._audit(
                "backend.read_fields", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "missing_fields_data",
                },
            )
            return None

        self._audit(
            "backend.read_fields", _slug_from(tenant_domain), {
                "submission_id": submission_id,
                "status":        "ok",
                "fields_count":  len(fields_data),
            },
        )
        return fields_data

    # --- Drive integration (backend-routed endpoints, P10.C.3) ---

    async def get_drive_token(
        self,
        service_token: str,
        submission_id: str,
        tenant_domain: str,
    ) -> Optional[str]:
        """Exchange a service token for a Drive access token.

        GET {base}/api/v1/drive/service-token?submission_id=...

        The backend holds Google's domain-wide delegation; we never hit
        Google's OAuth endpoints directly.
        """
        if not self._settings.backend_enabled or not service_token:
            return None
        try:
            base = self.base_url(tenant_domain)
        except BackendSSRFError as exc:
            _logger.warning("backend.drive_token: %s", exc)
            self._audit(
                "backend.drive_token", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status": "ssrf_blocked",
                },
            )
            return None

        try:
            resp = await self._http.get(
                f"{base}/api/v1/drive/service-token",
                params={"submission_id": submission_id},
                headers={
                    "authorization":    f"Bearer {service_token}",
                    "x-forwarded-host": tenant_domain,
                },
                timeout=self._settings.backend_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("backend.drive_token: network err=%s", exc)
            self._audit(
                "backend.drive_token", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "error",
                    "error_class":   type(exc).__name__,
                },
            )
            return None

        if resp.status_code != 200:
            _logger.warning(
                "backend.drive_token: http=%s", resp.status_code,
            )
            self._audit(
                "backend.drive_token", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "http_error",
                    "http_status":   resp.status_code,
                },
            )
            return None

        try:
            token = resp.json().get("access_token")
        except ValueError:
            token = None
        if not isinstance(token, str) or not token:
            self._audit(
                "backend.drive_token", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "missing_token",
                },
            )
            return None

        self._audit(
            "backend.drive_token", _slug_from(tenant_domain), {
                "submission_id": submission_id,
                "status":        "ok",
            },
        )
        return token

    async def get_agent_folder(
        self,
        service_token: str,
        submission_id: str,
        tenant_domain: str,
    ) -> Optional[dict]:
        """Fetch the tenant's agent/LOB Drive folder for this submission.

        GET {base}/api/v1/drive/agent-folder?submission_id=...
        Returns the full payload dict ({id, name, ...}) or None.
        """
        if not self._settings.backend_enabled or not service_token:
            return None
        try:
            base = self.base_url(tenant_domain)
        except BackendSSRFError as exc:
            _logger.warning("backend.agent_folder: %s", exc)
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status": "ssrf_blocked",
                },
            )
            return None

        try:
            resp = await self._http.get(
                f"{base}/api/v1/drive/agent-folder",
                params={"submission_id": submission_id},
                headers={
                    "authorization":    f"Bearer {service_token}",
                    "x-forwarded-host": tenant_domain,
                },
                timeout=self._settings.backend_timeout_s,
            )
        except httpx.HTTPError as exc:
            _logger.warning("backend.agent_folder: network err=%s", exc)
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "error",
                    "error_class":   type(exc).__name__,
                },
            )
            return None

        if resp.status_code != 200:
            _logger.warning(
                "backend.agent_folder: http=%s", resp.status_code,
            )
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "http_error",
                    "http_status":   resp.status_code,
                },
            )
            return None

        try:
            payload = resp.json()
        except ValueError:
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "non_json",
                },
            )
            return None
        # v3 wire contract (verified against drive_upload_v2.py:405-407):
        # the backend returns the LOB/agent folder id under the key
        # `lob_folder_id`, NOT `id`. Accept `id` as a fallback in case a
        # future backend revision standardizes on it, but require at least
        # one to be present.
        if not isinstance(payload, dict):
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "non_dict_payload",
                },
            )
            return None
        folder_id = payload.get("lob_folder_id") or payload.get("id")
        if not folder_id:
            self._audit(
                "backend.agent_folder", _slug_from(tenant_domain), {
                    "submission_id": submission_id,
                    "status":        "missing_folder_id",
                },
            )
            return None

        self._audit(
            "backend.agent_folder", _slug_from(tenant_domain), {
                "submission_id": submission_id,
                "status":        "ok",
                "folder_id":     folder_id,
            },
        )
        return payload

    # --- Internal ---

    def _audit(
        self, event_type: str, tenant: Optional[str], payload: dict,
    ) -> None:
        if self._audit_store is None:
            return
        record_audit_event(
            self._audit_store, event_type, tenant=tenant, payload=payload,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend_client(
    settings: Settings,
    *,
    audit_store: Optional[SessionStore] = None,
) -> Optional[BackendClient]:
    """Return a ready-to-use BackendClient, or None when disabled."""
    if not settings.backend_enabled:
        return None
    http = _get_http_client()
    return BackendClient(
        settings=settings, http=http, audit_store=audit_store,
    )
