"""HTTP API for Accord AI intake.

Endpoints:
    GET  /health
    POST /start-session
    POST /answer
    POST /finalize
    GET  /session/{id}
    GET  /session/{id}/messages
    GET  /sessions

Wire compatibility (P10.0.a):
    * v3 convention: tenant supplied via `X-Tenant-Slug` header (or, on
      `/start-session`, body field `tenant_slug`). A single `INTAKE_API_KEY`
      acts as an admin/shared key — caller picks the tenant.
    * v4 upgrade (opt-in): set `INTAKE_API_KEYS` to a JSON dict mapping
      `{api_key: tenant_slug}`. Each key is *bound* to a tenant; the
      binding wins over the header, and a mismatched header is rejected
      with 403. This closes the v3 leak where any admin-key holder could
      target any tenant.

The "effective tenant" produced by auth is what the middleware writes
into request.state and contextvars, and what handlers pass to the store
— never the raw header. This makes tenant isolation security-relevant,
not just a logging hint.

Error mapping:
    KeyError                     -> 404  (tenant-leak-safe)
    ConcurrencyError             -> 409
    LobTransitionError           -> 400
    ExtractionOutputError /
      RefinerOutputError         -> 502  (bad upstream LLM output)
    openai.RateLimitError        -> 503
    openai.APITimeoutError       -> 504
    openai.APIConnectionError    -> 503
    openai.AuthenticationError   -> 502
    other openai.APIError        -> 502
"""
from __future__ import annotations

import asyncio
import hmac
import re
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple, Union

import openai
from fastapi import Body, Depends, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.cors import CORSMiddleware

from accord_ai.app import IntakeApp, build_intake_app
from accord_ai.audit import (
    COMPLETE_OVERRIDES_APPLIED,
    SUBMISSION_COMPLETED,
    record_audit_event,
)
from accord_ai.config import Settings
from accord_ai.conversation.controller import TurnResult
from accord_ai.core.diff import LobTransitionError
from accord_ai.core.store import ConcurrencyError
from accord_ai.extraction.extractor import ExtractionOutputError
from accord_ai.forms import fill_submission
from accord_ai.forms.drive_pipeline import (
    DRIVE_STATUS_OVERWRITTEN,
    DRIVE_STATUS_SKIPPED,
    DRIVE_STATUS_UPLOADED,
    DriveUploadOutcome,
    run_drive_upload_pipeline,
)
from accord_ai.extraction.ocr import (
    DocKind as _OcrDocKind,
    OCRConfigError,
    OCRReadError,
    ocr_document,
)
from accord_ai.harness.refiner import RefinerOutputError
from accord_ai.http_client import close_client, get_client
from accord_ai.cache import hash_bytes
from accord_ai.logging_config import configure_logging, get_logger, redact_pii_text
from accord_ai.request_context import clear_context, new_request_id, set_context

_logger = get_logger("api")


# ---------------------------------------------------------------------------
# SFT transcript capture helpers (Phase 2.7)
# ---------------------------------------------------------------------------


def _reconstruct_turns(session_id: str, store, tenant: Optional[str]):
    """Approximate per-turn SFT turns from stored messages + final submission.

    No per-turn extraction snapshots are persisted, so extracted_diff uses the
    final cumulative submission for every turn. Known approximation; per-turn
    diffs can be added in Phase 4.5 once we store extraction deltas.
    """
    from accord_ai.feedback.transcript_capture import Turn

    session = store.get_session(session_id, tenant=tenant)
    if session is None:
        return []
    messages = store.get_messages(session_id, tenant=tenant)
    user_texts = [m.content for m in messages if m.role == "user"]
    if not user_texts:
        return []
    submission_dict = session.submission.model_dump(mode="json", exclude_none=True)
    return [Turn(user_text=t, extracted_diff=submission_dict) for t in user_texts]


async def _maybe_capture_transcript(
    *,
    session_id: str,
    tenant: Optional[str],
    store,
    transcript_capture,
    validation_results,
) -> None:
    from accord_ai.feedback.eligibility import is_session_sft_eligible

    correction_count = store.count_corrections_for_session(session_id, tenant=tenant)
    eligibility = is_session_sft_eligible(
        session_id=session_id,
        tenant=tenant or "",
        status="finalized",
        validation_results=validation_results,
        correction_count=correction_count,
    )
    if not eligibility.eligible:
        _logger.debug("sft_skip session=%s reason=%s", session_id, eligibility.reason)
        return
    turns = _reconstruct_turns(session_id, store, tenant)
    result = transcript_capture.capture(
        tenant=tenant or "default",
        session_id=session_id,
        turns=turns,
    )
    _logger.info(
        "sft_capture session=%s count=%d path=%s", session_id, result.count, result.path
    )


def _client_ip(request: Request) -> str:
    """Best-effort client IP for auth telemetry.

    X-Forwarded-For wins when behind a reverse proxy (first entry = original
    client). Falls back to the direct peer. Spoofable unless the upstream is
    trusted — treat as informational, not security-bearing.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


# Paths that bypass auth entirely (load balancers, health probes)
_OPEN_PATHS = frozenset({"/health"})
# Chat endpoints — unlocked when ACCORD_CHAT_OPEN=true (v3 convention).
# Read endpoints (/sessions, /session/{id}, /session/{id}/messages) are
# intentionally NOT in this set — they expose stored data.
_CHAT_PATHS = frozenset({
    "/start-session", "/answer", "/finalize", "/complete",
    "/upload-document", "/upload-image", "/upload-filled-pdfs", "/upload-blank-pdfs",
    "/enrich", "/correction", "/feedback",
})
# /pdf/{sid}/{form} intentionally NOT in _CHAT_PATHS — it exposes stored
# data, same reasoning as /session/{id}. Read endpoints stay gated.


def _resolve_auth(
    request: Request, settings: Settings,
) -> Tuple[Optional[JSONResponse], Optional[str], bool]:
    """Authorize the request and resolve the effective tenant.

    Returns (error_response, effective_tenant, is_admin):
      * error_response is None on success; a JSONResponse with 401/403/500
        on failure.
      * effective_tenant is the tenant the handlers should use for all
        store/controller operations. For admin-key / auth-disabled paths
        this is whatever the X-Tenant-Slug header says (caller-picked);
        for a per-tenant bound key this is the *binding*, which wins
        over any header hint.
      * is_admin is True when the admin (shared) key was used or when auth
        is disabled. Handlers that restrict access to admin-only routes
        check request.state.is_admin rather than re-parsing the key.
    """
    path = request.url.path
    header_tenant = request.headers.get("x-tenant-slug")

    if path in _OPEN_PATHS:
        return None, header_tenant, False
    if settings.accord_auth_disabled:
        # Auth disabled → treat as admin for diagnostic/test access.
        return None, header_tenant, True
    if settings.accord_chat_open and path in _CHAT_PATHS:
        return None, header_tenant, False

    # No auth configured at all → misconfiguration, not 401
    if settings.intake_api_key is None and not settings.intake_api_keys:
        _logger.error(
            "auth misconfig: no INTAKE_API_KEY or INTAKE_API_KEYS set "
            "(path=%s ip=%s)",
            path, _client_ip(request),
        )
        return (
            JSONResponse(
                status_code=500,
                content={
                    "detail":
                        "server not configured (INTAKE_API_KEY or "
                        "INTAKE_API_KEYS unset)",
                },
            ),
            None,
            False,
        )

    # v3 wire contract (verified api.py:243 — `HTTPBearer()`): the FE
    # authenticates via `Authorization: Bearer {api_key}`. v4 also accepts
    # the native `X-API-Key: {api_key}` header for new integrations. Bearer
    # takes precedence when both are present.
    provided = ""
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        provided = auth_header[7:].strip()
    if not provided:
        provided = request.headers.get("x-api-key") or ""

    # Per-tenant bound keys first. A bound key + mismatched header = 403
    # (security-relevant, NOT 401: the key is valid, but the caller is
    # attempting to target a tenant it doesn't own).
    for key, bound_tenant in settings.intake_api_keys.items():
        if hmac.compare_digest(provided, key):
            if header_tenant is None or header_tenant == bound_tenant:
                return None, bound_tenant, False  # tenant key, not admin
            _logger.warning(
                "auth: tenant-binding violation "
                "(bound=%s claimed=%s path=%s ip=%s)",
                bound_tenant, header_tenant, path, _client_ip(request),
            )
            return (
                JSONResponse(
                    status_code=403,
                    content={
                        "detail":
                            f"API key is bound to tenant {bound_tenant!r}, "
                            f"X-Tenant-Slug={header_tenant!r}",
                    },
                ),
                None,
                False,
            )

    # Admin (shared) key — caller controls tenant via header / body.
    if settings.intake_api_key is not None:
        expected = settings.intake_api_key.get_secret_value()
        if hmac.compare_digest(provided, expected):
            return None, header_tenant, True  # admin key

    _logger.warning(
        "auth: invalid key (path=%s ip=%s)",
        path, _client_ip(request),
    )
    return (
        JSONResponse(
            status_code=401,
            content={"detail": "invalid or missing X-API-Key"},
        ),
        None,
        False,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    """v3 wire contract (verified accord_ai_v3/api.py:378-386):
        submission_id — optional, caller can pre-assign the session id
        session_id    — backward-compat alias for submission_id
        tenant_slug   — tenant routing (also accepted via X-Tenant-Slug header)
        tenant_domain — tenant domain for backend subdomain routing (used by
                        /complete to mint service tokens; pre-populated here
                        so the session remembers it across turns)
        phase         — v3-specific phase marker, default "customer"
    """
    submission_id: Optional[str] = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$",
    )
    session_id:    Optional[str] = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$",
    )
    tenant_slug:   str = Field(default="", max_length=64)
    tenant_domain: str = Field(default="", max_length=256)
    phase:         str = Field(default="customer", max_length=32)

    @property
    def effective_id(self) -> Optional[str]:
        """v3 accepts either name; submission_id wins when both present."""
        return self.submission_id or self.session_id or None


class StartSessionResponse(BaseModel):
    """v3 wire contract (verified accord_ai_v3/api.py:389-391): exactly
    {submission_id, question}. No extra fields."""
    submission_id: str
    question: str


class AnswerRequest(BaseModel):
    session_id: str
    # Cap matches CompleteRequest.message — prevents unbounded-payload DoS
    # against the LLM (a valid API key + 10MB message = cost explosion or
    # local-model stall). 50k chars ~= 12k tokens, well beyond any real turn.
    message: str = Field(
        ..., min_length=1, max_length=50_000,
        description="The user's message for this turn.",
    )


class VerdictModel(BaseModel):
    passed: bool
    reasons: List[str]
    failed_paths: List[str]


# --- v3 wire-compat sub-models for /answer response (P10.0.g.4) ---
#
# Minimal subset of v3's progress/tracking shape. Populated with safe
# defaults so the FE can render without NPE; real progress/pdf-status
# tracking is a separate step (controller/pipeline plumbing).

class SectionProgress(BaseModel):
    """v3 SectionProgress (api.py:412-420) — unused by us yet, stubbed."""
    id: str
    name: str
    status: str                   # not_started | in_progress | complete | locked
    pct: float = 0.0
    filled: int = 0
    total: int = 0
    missing: List[str] = Field(default_factory=list)
    note: str = ""


class FormProgress(BaseModel):
    form_number: str
    name: str
    status: str                   # pending | ready
    pct: float = 0.0
    filled_fields: int = 0
    total_fields: int = 0


class CurrentFlow(BaseModel):
    id: str
    name: str
    position: int
    total_flows: int


class GateStatus(BaseModel):
    reached: bool = False
    passed: bool = False


class LOBInfo(BaseModel):
    id: str
    name: str
    confidence: float = 0.0


class Progress(BaseModel):
    """v3 Progress shape (api.py:445-455). Emitted as None for now — we
    don't track section-level progress yet. FE must tolerate null."""
    overall_pct:       float = 0.0
    total_fields:      int = 0
    filled_fields:     int = 0
    remaining_fields:  int = 0
    current_flow:      Optional[CurrentFlow] = None
    sections:          List[SectionProgress] = Field(default_factory=list)
    forms:             List[FormProgress] = Field(default_factory=list)
    lobs:              List[LOBInfo] = Field(default_factory=list)
    gate:              GateStatus = Field(default_factory=GateStatus)


class FilledPDF(BaseModel):
    """v3 FilledPDF (api.py:458-461): inline base64 PDFs returned on /answer
    when the baseline fills. Emitted as [] for now — baseline detection
    isn't wired."""
    form_number: str
    filename: str
    data: str   # base64-encoded PDF bytes


class DriveFile(BaseModel):
    pdf_id: str
    file_id: str
    view_url: str
    web_content_link: str


class TurnResultResponse(BaseModel):
    """v3-wire-compat /answer response.

    Emits v3's full 18-field AnswerResponse shape (api.py:467-486) PLUS
    the v4-native fields (session_id, submission, verdict, is_complete)
    additively. FE can read either naming; v4 tests still assert on v4
    fields. Matches the additive pattern used for /complete.

    Intentional semantic mapping:
      finished          ← is_complete
      next_question     ← assistant_message
      submission_id     ← session_id (alias)

    Stubbed v3 fields (real tracking in follow-up steps):
      category / question_index / total_questions_in_category: static
          defaults matching v3 (we don't track question categories yet)
      pdf_status: {} (no per-form fill tracking on /answer yet; /complete
          is where fills happen)
      drive_files: None (Drive uploads run on /complete, not /answer)
      filled_pdfs: [] (inline base64 PDFs on /answer is a v3-era
          feature — real work needs a mid-conversation fill loop)
      baseline_complete: False (no baseline-fields detection yet)
      progress: None (no section-progress tracker yet)
      validation_errors: [] (validator module not yet wired)
      harness_version: 1 (fixed; v4's harness doesn't version like v3)
      message: None
      output_dir: ""
    """
    # --- v4-native fields ---
    session_id: str
    submission: dict = Field(..., description="CustomerSubmission as JSON-dumped dict.")
    verdict: VerdictModel
    assistant_message: str
    is_complete: bool
    # --- v3 wire-compat fields ---
    finished:                     bool = False
    next_question:                Optional[str] = None
    category:                     str = "General Information"
    question_index:               int = 0
    total_questions_in_category:  int = 10
    current_category_index:       int = 0
    total_categories:             int = 8
    pdf_status:                   Dict[str, str] = Field(default_factory=dict)
    drive_files:                  Optional[List[DriveFile]] = None
    message:                      Optional[str] = None
    output_dir:                   str = ""
    filled_pdfs:                  List[FilledPDF] = Field(default_factory=list)
    baseline_complete:            bool = False
    submission_id:                str = ""
    progress:                     Optional[Progress] = None
    validation_errors:            List[str] = Field(default_factory=list)
    harness_version:              int = 1


class FinalizeRequest(BaseModel):
    session_id: str


class FinalizeResponse(BaseModel):
    session_id: str
    status: str = Field(..., description="Final status — 'finalized'.")


class UploadDocumentResponse(BaseModel):
    """Response for POST /upload-document (fleet roster upload)."""
    session_id: str
    filename: str
    drivers_added: int
    drivers_updated: int
    vehicles_added: int
    vehicles_updated: int
    header_row_idx: int
    warnings: List[str]


# --- /complete (P10.A.5) ---

# v3 wire contract (verified accord_ai_v3/api.py:1672-1675):
#     fields_data: dict[str, dict[str, Any]] = {}
# No caps, no value-type restrictions — v3 relies on upstream nginx/uvicorn
# request-size limits for DoS defense.
#
# v4 keeps the same `Any` inner type (v3 FE may send nested structures, e.g.
# form 163's `_header`/`drivers` structured shape, and those must round-trip)
# but adds generous per-dimension caps as defense-in-depth. Caps are sized
# well above any realistic v3 payload:
#   * max forms — 32 (real FE sends up to 10 for all LOBs combined)
#   * max widgets per form — 2000 (ACORD 160 has ~1135; 2000 is ~2x max)
#   * max value length — 100_000 (operations_description is ~2KB in practice)
# Payloads that breach these caps are almost certainly malicious; anything
# legitimate flows through without tripping them.
_FIELDS_DATA_MAX_FORMS            = 32
_FIELDS_DATA_MAX_WIDGETS_PER_FORM = 2000
_FIELDS_DATA_MAX_VALUE_LEN        = 100_000


class CompleteRequest(BaseModel):
    """v3 wire-compatible (verified accord_ai_v3/api.py:1665-1675).

    `fields_data` carries FE-editor overrides keyed by form_number
    (``"form_125"`` or the bare ``"125"`` shape). Inner value type is
    `Any` so v3's structured shapes (form 163 `_header` / `drivers`)
    round-trip without rejection. Flat widget_name → scalar entries
    merge with mapper output; overrides win; empty-string clears.

    Size caps are generous but not unlimited — see
    _FIELDS_DATA_MAX_* constants.
    """
    submission_id: str = Field(..., min_length=1, max_length=64)
    tenant_slug:   str = Field(default="",  max_length=64)
    tenant_domain: str = Field(default="",  max_length=256)
    message:       str = Field(default="",  max_length=50_000)
    source:        str = Field(default="",  max_length=64)
    agent_id:      str = Field(default="",  max_length=64)
    # `Any` inner value matches v3 exactly. Caps applied via validator below.
    fields_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("fields_data")
    @classmethod
    def _cap_fields_data(
        cls, v: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Generous caps — v3-realistic payloads always pass; reject only
        the pathological ones."""
        if not v:
            return v
        if len(v) > _FIELDS_DATA_MAX_FORMS:
            raise ValueError(
                f"fields_data exceeds max forms "
                f"({len(v)} > {_FIELDS_DATA_MAX_FORMS})"
            )
        for form_key, widgets in v.items():
            if not isinstance(widgets, dict):
                # v3 requires inner-dict shape; non-dict is structurally wrong.
                raise ValueError(
                    f"fields_data[{form_key!r}] must be a dict, "
                    f"got {type(widgets).__name__}"
                )
            if len(widgets) > _FIELDS_DATA_MAX_WIDGETS_PER_FORM:
                raise ValueError(
                    f"fields_data[{form_key!r}] exceeds max widgets "
                    f"({len(widgets)} > {_FIELDS_DATA_MAX_WIDGETS_PER_FORM})"
                )
            for widget_name, value in widgets.items():
                if (
                    isinstance(value, str)
                    and len(value) > _FIELDS_DATA_MAX_VALUE_LEN
                ):
                    raise ValueError(
                        f"fields_data[{form_key!r}][{widget_name!r}] "
                        f"exceeds max value length "
                        f"({len(value)} > {_FIELDS_DATA_MAX_VALUE_LEN})"
                    )
        return v


class CompleteFormInfo(BaseModel):
    """Per-form metadata returned from /complete.

    The `drive_*` fields are populated when the Drive upload pipeline ran
    for this form; when Drive is disabled / skipped they carry the defaults
    ("skipped" + Nones) so response consumers never see missing keys.
    """
    form_number:    str
    content_hash:   str
    byte_length:    int
    dedup_skipped:  bool = Field(
        description=(
            "True when the filled PDF's content hash matched the previously "
            "stored hash — no new bytes were written to disk."
        ),
    )
    fill_result:    dict   # FillResult.to_dict() shape
    drive_status:   str = Field(
        default=DRIVE_STATUS_SKIPPED,
        description=(
            "Drive upload outcome for this form — one of: "
            "'skipped' (pipeline not run), 'uploaded' (new file), "
            "'overwritten' (PATCH over existing), 'failed' "
            "(Drive API returned non-success), 'auth_failed' "
            "(401 after re-exchange retry)."
        ),
    )
    drive_file_id:  Optional[str] = Field(
        default=None,
        description="Google Drive file id, when upload succeeded.",
    )
    drive_view_url: Optional[str] = Field(
        default=None,
        description="Browser view URL for the uploaded file.",
    )


class CompleteResponse(BaseModel):
    """Top-level /complete response.

    Wire-compatible with v3 (``accord_ai_v3/api.py:2055-2070``) — the FE was
    built against that shape, so v4 emits every v3 key plus the native v4
    detail fields. v3 keys first; v4 additions follow and are ignorable by
    v3-era consumers.

    v3 shape: ``{submission_id, uploaded, total, field_diff, fill_summary,
    drive_files, validation, cache, pruned}``. `field_diff` / `validation` /
    `cache` are stubs for now — the data plumbing lands in a follow-up
    step; callers that rely on them today would get the same empty shape
    they already see on v3 when validation is skipped / cache is cold.

    `drive_enabled` is True when the pipeline attempted Drive work (Drive +
    Backend clients wired, tenant_domain supplied). It stays True even
    when subsequent steps short-circuited (e.g. token exchange failed) —
    per-form `drive_status` explains what actually happened.
    """
    # --- v3 wire shape ----------------------------------------------------
    submission_id:   str = Field(
        description="v3 compat: same value as session_id. v3 FEs key off this.",
    )
    uploaded:        int = Field(
        default=0,
        description=(
            "Count of forms whose Drive upload succeeded this call. "
            "(`drive_status` ∈ {'uploaded', 'overwritten'})."
        ),
    )
    total:           int = Field(
        default=0,
        description="Total number of forms processed (len(forms)).",
    )
    field_diff:      Dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Per-form field-level diff vs. the previous /complete call. "
            "Stub in v4: empty dict until we wire the content-delta tracker."
        ),
    )
    fill_summary:    List[dict] = Field(
        default_factory=list,
        description=(
            "Per-form fill counts: "
            "[{form_number, pdf_id, fields_filled, fields_skipped, errors, "
            "skipped_upload_unchanged?}]."
        ),
    )
    drive_files:     List[dict] = Field(
        default_factory=list,
        description=(
            "Per-uploaded-form Drive descriptors: "
            "[{pdf_id, file_id, view_url}]. Empty when Drive was skipped."
        ),
    )
    validation:      dict = Field(
        default_factory=dict,
        description=(
            "Validation pipeline result. Stub in v4 (empty dict); wired when "
            "the validator module lands."
        ),
    )
    cache:           dict = Field(
        default_factory=dict,
        description=(
            "Cache-hit telemetry for this call: "
            "{auth_hit, validation_hit, content_dedup_skipped, file_id_hits}."
        ),
    )
    pruned:          int = Field(
        default=0,
        description=(
            "Count of stale files pruned from the submission folder "
            "(same as pruned_count below — v3 FEs read `pruned`)."
        ),
    )

    # --- v4 native detail (additive; v3 FEs ignore these) -----------------
    session_id:      str
    forms:           List[CompleteFormInfo]
    total_bytes:     int
    total_written:   int     # forms that actually hit disk (dedup misses)
    drive_enabled:   bool = Field(
        default=False,
        description=(
            "True when the Drive upload pipeline was attempted. False when "
            "backend/drive clients aren't wired or tenant_domain was empty."
        ),
    )
    drive_folder_id: Optional[str] = Field(
        default=None,
        description="Per-submission Drive folder id, when resolved.",
    )
    pruned_count:    int = Field(
        default=0,
        description=(
            "Count of stale files successfully deleted from the submission "
            "folder — non-zero when an LOB drop-shift left orphan PDFs."
        ),
    )


# --- 11.c: read-endpoint models ---

class SessionDetailResponse(BaseModel):
    session_id: str
    tenant: Optional[str]
    status: str
    created_at: str
    updated_at: str
    submission: dict


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: str


class MessagesResponse(BaseModel):
    session_id: str
    messages: List[MessageResponse]


class SessionSummaryResponse(BaseModel):
    session_id: str
    tenant: Optional[str]
    status: str
    created_at: str
    updated_at: str


class SessionsResponse(BaseModel):
    sessions: List[SessionSummaryResponse]


# --- Phase 1.9 models (/explain, /enrich, /correction, /feedback) ----------

class ExplainSource(BaseModel):
    title: str
    snippet: str
    score: float


class ExplainResponse(BaseModel):
    field: str
    explanation: str
    sources: List[ExplainSource]


class EnrichRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


class EnrichResponse(BaseModel):
    status: str = "ok"
    validators_run: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    cached: bool = False


class CorrectionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    turn: int = Field(..., ge=0)
    field_path: str = Field(..., min_length=1, max_length=256)
    wrong_value: Any = None
    correct_value: Any = None
    explanation: Optional[str] = Field(default=None, max_length=2000)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    turn: int = Field(..., ge=0)
    rating: int = Field(..., description="Star rating 1–5.")
    comment: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("rating")
    @classmethod
    def _validate_rating(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


class CapturedResponse(BaseModel):
    captured: bool = True
    id: str


class DPOExportRequest(BaseModel):
    force: bool = False


# --- Phase 1.10 models (/upload-image, /upload-filled-pdfs, /upload-blank-pdfs,
#     /debug, /fields, /harness family) ----------------------------------------

class UploadImageResponse(BaseModel):
    status: str = "ok"
    kind: str
    extracted: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class UploadedPdfInfo(BaseModel):
    form_number: str
    drive_file_id: Optional[str] = None
    drive_url: Optional[str] = None


class FailedPdfInfo(BaseModel):
    form_number: str
    error: str


class UploadFilledPdfsResponse(BaseModel):
    uploaded: List[UploadedPdfInfo]
    failed: List[FailedPdfInfo]


class UploadBlankPdfsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    lob: str = Field(..., min_length=1, max_length=64)
    forms: Optional[List[str]] = None


class UploadBlankPdfsResponse(BaseModel):
    uploaded: List[UploadedPdfInfo]
    failed: List[FailedPdfInfo]
    note: str = "Blank PDF seeding ships in Phase 1.5"


class DebugTurn(BaseModel):
    turn_idx: int
    role: str
    content: str
    created_at: str


class DebugSessionResponse(BaseModel):
    session: dict
    submission: dict
    turns: List[DebugTurn]
    harness_version: str
    last_refiner_stage: str


class FieldsResponse(BaseModel):
    """Session submission in FE-facing shape — all CustomerSubmission fields."""

    class Config:
        extra = "allow"


class HarnessResponse(BaseModel):
    version: str
    critical_fields_per_lob: Dict[str, List[str]]
    active_rules: List[str]
    refiner_harness_enabled: bool


class HarnessAuditResponse(BaseModel):
    session_id: Optional[str] = None
    refinement_count: int = 0
    judge_pass_rate: Optional[float] = None
    most_failed_paths: List[str] = Field(default_factory=list)
    note: str = "Per-session harness metrics ship in Phase D"


class HarnessHistoryResponse(BaseModel):
    versions: List[dict]
    note: str


class HarnessProvenanceResponse(BaseModel):
    entries: List[dict] = Field(default_factory=list)
    note: str


class HarnessReviewQueueResponse(BaseModel):
    queue: List[dict] = Field(default_factory=list)
    note: str


# Regex to extract a form number from FE filename conventions:
# "125-Form.pdf", "acord_125.pdf", "125.pdf" → "125"
_FORM_FROM_FILENAME_RE = re.compile(
    r"(?:^|[_-])(\d{2,3})(?:[_-]Form)?\.pdf$", re.IGNORECASE,
)

_IMAGE_ALLOWED_EXT = frozenset({"jpg", "jpeg", "png", "heic", "webp"})
_IMAGE_ALLOWED_KIND = frozenset({
    "drivers_license", "insurance_card", "vehicle_registration",
})


# ---------------------------------------------------------------------------
# OCR merge helper
# ---------------------------------------------------------------------------

def _merge_ocr_into_submission(sub, fields, kind, merge_drivers_fn, merge_vehicles_fn):
    """Merge OCR-extracted fields into a CustomerSubmission.

    Works on submission dict (like fleet_ingest) for clean discriminated-union
    handling. Delegates to Phase-1.4 merge_drivers / merge_vehicles.
    Returns a new CustomerSubmission.
    """
    from accord_ai.schema import CustomerSubmission, Driver, PriorInsurance, Vehicle
    from accord_ai.extraction.ocr import (
        DriverLicenseFields, InsuranceCardFields, RegistrationFields,
    )

    sub_dict = sub.model_dump(mode="json")
    lob = sub_dict.get("lob_details") or {}

    if kind == "drivers_license" and isinstance(fields, DriverLicenseFields):
        if lob.get("lob") != "commercial_auto":
            return sub  # DL only meaningful for CA; no-op elsewhere
        lob.setdefault("drivers", [])
        existing_d = [Driver.model_validate(d) for d in lob["drivers"]]
        new_driver = Driver(
            first_name=fields.first_name,
            last_name=fields.last_name,
            date_of_birth=fields.date_of_birth,
            license_number=fields.license_number,
            license_state=fields.license_state,
        )
        merged = merge_drivers_fn(existing_d, [new_driver])
        lob["drivers"] = [d.model_dump(exclude_none=True) for d in merged]
        sub_dict["lob_details"] = lob
        return CustomerSubmission.model_validate(sub_dict)

    if kind == "vehicle_registration" and isinstance(fields, RegistrationFields):
        # LOB guard already done in the endpoint handler.
        lob.setdefault("vehicles", [])
        existing_v = [Vehicle.model_validate(v) for v in lob["vehicles"]]
        new_vehicle = Vehicle(
            vin=fields.vin,
            year=fields.year,
            make=fields.make,
            model=fields.model,
            registration_state=fields.registration_state,
        )
        merged = merge_vehicles_fn(existing_v, [new_vehicle])
        lob["vehicles"] = [v.model_dump(exclude_none=True) for v in merged]
        sub_dict["lob_details"] = lob
        return CustomerSubmission.model_validate(sub_dict)

    if kind == "insurance_card" and isinstance(fields, InsuranceCardFields):
        # prior_insurance lives on WorkersCompDetails; for other LOBs we skip.
        if lob.get("lob") != "workers_comp":
            return sub
        prior = PriorInsurance(
            carrier_name=fields.carrier,
            policy_number=fields.policy_number,
            effective_date=fields.effective_date,
            expiration_date=fields.expiration_date,
        )
        existing = list(lob.get("prior_insurance") or [])
        existing.insert(0, prior.model_dump(exclude_none=True))
        lob["prior_insurance"] = existing
        sub_dict["lob_details"] = lob
        return CustomerSubmission.model_validate(sub_dict)

    return sub


# ---------------------------------------------------------------------------
# Module-level TTL-cached retrieval helper (shared across requests)
# ---------------------------------------------------------------------------

from accord_ai.cache import ttl_cached  # noqa: E402 — after models for clarity


@ttl_cached(ttl_seconds=300, key=lambda settings, tenant, field: (tenant, field))
async def _explain_retrieve(settings, tenant: str, field: str):
    """Embed `field` and retrieve top-5 knowledge hits. Cached 300 s per (tenant, field)."""
    from accord_ai.knowledge import build_retriever
    retriever = build_retriever(settings, collection_name=tenant or "default")
    return await retriever.retrieve(field, k=5)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _turn_result_to_response(session_id: str, result: TurnResult) -> TurnResultResponse:
    """Build the v4 response with v3 wire-compat fields populated additively.

    v3 field semantics:
      finished        = is_complete
      next_question   = assistant_message (None when empty, matches v3)
      submission_id   = session_id (v3 naming alias)
      All other v3 fields ship with v3's documented defaults — real
      tracking (pdf_status / progress / filled_pdfs / baseline_complete /
      validation_errors) lands in follow-up steps.
    """
    assistant_msg = result.assistant_message
    return TurnResultResponse(
        # v4-native fields (existing tests keep working).
        session_id=session_id,
        submission=result.submission.model_dump(mode="json"),
        verdict=VerdictModel(
            passed=result.verdict.passed,
            reasons=list(result.verdict.reasons),
            failed_paths=list(result.verdict.failed_paths),
        ),
        assistant_message=assistant_msg,
        is_complete=result.is_complete,
        # v3 wire-compat fields.
        finished=result.is_complete,
        next_question=assistant_msg if assistant_msg else None,
        submission_id=session_id,
        # All other v3 fields take their defaults from the model definition.
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

async def _prime_engines(intake: IntakeApp) -> None:
    """Fire one extractor + one responder call against a dummy submission.

    vLLM compiles the ``guided_json`` xgrammar and warms the chat-template
    prefix cache on the first call — 5-10 s of latency that would
    otherwise land on the first real user turn. After warmup, those
    costs amortize to ~zero.

    Errors are swallowed and logged; a warmup failure should never take
    down the API (the real request path has its own error handling,
    and we'd rather the FE see a first-turn retry than a 502 at startup).
    """
    try:
        from accord_ai.harness.judge import JudgeVerdict
        from accord_ai.schema import CustomerSubmission

        blank = CustomerSubmission()
        _logger.info("warmup: starting dummy extract + responder calls")
        # Extractor: triggers xgrammar compilation + full schema prompt prefill.
        await intake.controller._extractor.extract(
            user_message="__warmup__ (ignore)",
            current_submission=blank,
        )
        # Responder: triggers the no-guided-json path warmup too.
        await intake.responder.respond(
            submission=blank,
            verdict=JudgeVerdict(passed=True),
        )
        _logger.info("warmup: complete")
    except Exception as exc:
        # Any failure here is non-fatal. Log and continue.
        _logger.warning("warmup: skipped/failed (%s: %s)", type(exc).__name__, exc)


def build_fastapi_app(
    settings: Optional[Settings] = None,
    *,
    intake: Optional[IntakeApp] = None,
) -> FastAPI:
    settings = settings or Settings()
    intake = intake or build_intake_app(settings)

    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Startup: initialise shared HTTP client + kick off warmup.
        get_client()  # eagerly create so first real request doesn't pay init cost
        warmup_task: Optional[asyncio.Task] = None
        if settings.warmup_on_boot:
            warmup_task = asyncio.create_task(_prime_engines(intake))
        yield
        # Shutdown: cancel in-flight warmup, then close shared HTTP client.
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except (asyncio.CancelledError, Exception):
                pass
        await close_client()

    app = FastAPI(
        title="Accord AI Intake",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.intake = intake
    app.state.settings = settings

    # Validation engine — built once at startup, validators list may be empty
    # when ENABLE_EXTERNAL_VALIDATION=false (default).
    from accord_ai.validation.engine import build_engine as _build_validation_engine
    from accord_ai.cache import _TTLDict
    from accord_ai.feedback.collector import CorrectionCollector, PIIFilter
    app.state.validation_engine = _build_validation_engine(settings)
    app.state.correction_collector = CorrectionCollector(
        db_path=settings.db_path,
        pii_filter=PIIFilter(),
    )
    from accord_ai.feedback.dpo import DPOManager
    app.state.dpo_manager = DPOManager(
        db_path=settings.db_path,
        output_dir=settings.training_data_dir,
        threshold=settings.dpo_threshold,
    )
    # Bounded TTL caches keyed by (session_id, submission_hash).
    # maxsize=1000 prevents unbounded growth under load; _TTLDict evicts on
    # overflow and on stale read. Locks required — _TTLDict is not thread-safe.
    _enrich_lock = threading.Lock()
    _enrich_cache = _TTLDict(maxsize=1000, ttl=600.0)
    _review_lock = threading.Lock()
    _review_cache = _TTLDict(maxsize=1000, ttl=600.0)

    # --- Rate limiting (slowapi) -------------------------------------------
    # Per-IP keyed via _client_ip (X-Forwarded-For aware) so real clients
    # behind ngrok/nginx are limited, not the proxy. Limits bake in at
    # factory time from Settings — runtime env changes don't re-arm.
    # `enabled=False` → unlimited (default for tests + local dev).
    # headers_enabled requires response: Response kwarg on every endpoint
    # (for success-path X-RateLimit-* injection). We keep endpoint signatures
    # clean; the 429 still carries Retry-After via _rate_limit_exceeded_handler,
    # which is the part clients actually need.
    limiter = Limiter(
        key_func=_client_ip,
        enabled=settings.rate_limit_enabled,
    )
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    async def _rate_limit_handler(
        request: Request, exc: RateLimitExceeded,
    ) -> JSONResponse:
        # slowapi's default handler only adds Retry-After when the Limiter
        # was built with headers_enabled=True. We keep headers_enabled off
        # (because that also requires every endpoint to take `response:
        # Response` as a kwarg), but still want well-behaved clients to
        # see Retry-After and back off. 60s matches the per-minute
        # granularity of the slowapi limits we configure.
        return JSONResponse(
            status_code=429,
            content={"detail": f"rate limit exceeded: {exc.detail}"},
            headers={"retry-after": "60"},
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    answer_limit = f"{settings.rate_limit_answer_per_minute}/minute"
    complete_limit = f"{settings.rate_limit_complete_per_minute}/minute"
    start_session_limit = (
        f"{settings.rate_limit_start_session_per_minute}/minute"
    )

    # ---- Middleware: per-request context + auth ----
    # Combined so auth failures still carry request_id (log correlation).
    # Auth resolves the *effective* tenant (bound-key wins over header);
    # that's what goes into context + request.state — never the raw header.
    @app.middleware("http")
    async def context_and_auth_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or new_request_id()
        header_tenant = request.headers.get("x-tenant-slug")

        # Provisional context so that auth-path logs carry a request_id
        # even if auth fails and we never reach the handler.
        set_context(request_id=request_id, tenant=header_tenant)
        try:
            auth_response, effective_tenant, is_admin = _resolve_auth(request, settings)
            if auth_response is not None:
                auth_response.headers["x-request-id"] = request_id
                if header_tenant is not None:
                    auth_response.headers["x-tenant-slug"] = header_tenant
                return auth_response

            # Auth succeeded — overwrite context with the security-correct
            # effective tenant, then make it available to handlers via state.
            set_context(request_id=request_id, tenant=effective_tenant)
            request.state.effective_tenant = effective_tenant
            request.state.is_admin = is_admin

            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            if effective_tenant is not None:
                response.headers["x-tenant-slug"] = effective_tenant
            return response
        finally:
            clear_context()

    # ---- Exception handlers ----
    @app.exception_handler(KeyError)
    async def _handle_key_error(request: Request, exc: KeyError):
        # Tenant-leak-safe — missing / wrong-tenant / terminal all look the same
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(ConcurrencyError)
    async def _handle_concurrency(request: Request, exc: ConcurrencyError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(LobTransitionError)
    async def _handle_lob_transition(request: Request, exc: LobTransitionError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ExtractionOutputError)
    async def _handle_extraction_output(request: Request, exc: ExtractionOutputError):
        return JSONResponse(
            status_code=502,
            content={"detail": "upstream LLM produced invalid output (extraction)"},
        )

    @app.exception_handler(RefinerOutputError)
    async def _handle_refiner_output(request: Request, exc: RefinerOutputError):
        return JSONResponse(
            status_code=502,
            content={"detail": "upstream LLM produced invalid output (refinement)"},
        )

    @app.exception_handler(openai.RateLimitError)
    async def _handle_rate_limit(request: Request, exc: openai.RateLimitError):
        headers = {}
        retry_after = getattr(exc, "response", None) and exc.response.headers.get("retry-after")
        if retry_after:
            headers["retry-after"] = str(retry_after)
        return JSONResponse(
            status_code=503,
            content={"detail": "upstream LLM rate-limited"},
            headers=headers,
        )

    @app.exception_handler(openai.APITimeoutError)
    async def _handle_timeout(request: Request, exc: openai.APITimeoutError):
        return JSONResponse(status_code=504, content={"detail": "upstream LLM timed out"})

    @app.exception_handler(openai.APIConnectionError)
    async def _handle_connection(request: Request, exc: openai.APIConnectionError):
        return JSONResponse(status_code=503, content={"detail": "upstream LLM unreachable"})

    @app.exception_handler(openai.AuthenticationError)
    async def _handle_auth(request: Request, exc: openai.AuthenticationError):
        _logger.error("upstream LLM authentication failed — check LLM_API_KEY config")
        return JSONResponse(
            status_code=502,
            content={"detail": "upstream LLM authentication misconfigured (server-side)"},
        )

    @app.exception_handler(openai.APIError)
    async def _handle_api_error(request: Request, exc: openai.APIError):
        return JSONResponse(status_code=502, content={"detail": "upstream LLM error"})

    # ---- Routes ----
    def _get_intake(request: Request) -> IntakeApp:
        return request.app.state.intake

    def _auth_tenant(request: Request) -> Optional[str]:
        """Return the effective tenant set by the auth middleware.

        For /health and other OPEN_PATHS the middleware still populates
        state with whatever the header hinted (or None). Handlers that
        care about tenant isolation should ONLY use this helper — never
        read X-Tenant-Slug directly.
        """
        return getattr(request.state, "effective_tenant", None)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
        }

    @app.post("/start-session", response_model=StartSessionResponse)
    @limiter.limit(start_session_limit)
    async def start_session(
        request: Request,
        req: StartSessionRequest = Body(default_factory=StartSessionRequest),
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        # Fall back to body-supplied tenant only on admin/auth-disabled
        # paths where the middleware didn't bind a tenant. For bound keys
        # the binding already won — the body field is ignored.
        if tenant is None and req.tenant_slug:
            tenant = req.tenant_slug

        # v3 accepts a caller-supplied submission_id for resuming — our store
        # doesn't support pre-assigned ids yet, so we ignore it and always
        # mint fresh. Logged so it's visible if a FE does rely on resume.
        if req.effective_id:
            _logger.info(
                "start-session: caller-supplied id %r ignored (resume not yet supported)",
                req.effective_id,
            )

        sid = intake.store.create_session(tenant=tenant)
        session = intake.store.get_session(sid, tenant=tenant)
        # Initial greeting — responder against empty submission + failing verdict
        verdict = intake.judge.evaluate(session.submission)
        greeting = await intake.responder.respond(
            submission=session.submission,
            verdict=verdict,
        )
        return StartSessionResponse(submission_id=sid, question=greeting)

    @app.post("/answer", response_model=TurnResultResponse)
    @limiter.limit(answer_limit)
    async def answer(
        req: AnswerRequest,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        result = await intake.controller.process_turn(
            session_id=req.session_id,
            user_message=req.message,
            tenant=tenant,
        )
        return _turn_result_to_response(req.session_id, result)

    @app.post("/finalize", response_model=FinalizeResponse)
    async def finalize(
        req: FinalizeRequest,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        session = intake.store.get_session(req.session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {req.session_id}")
        validation_results = await request.app.state.validation_engine.run_all(
            session.submission
        )
        intake.store.finalize_session(req.session_id, tenant=tenant)
        if intake.transcript_capture is not None:
            await _maybe_capture_transcript(
                session_id=req.session_id,
                tenant=tenant,
                store=intake.store,
                transcript_capture=intake.transcript_capture,
                validation_results=validation_results,
            )
        return FinalizeResponse(session_id=req.session_id, status="finalized")

    _UPLOAD_MAX_BYTES = 10 * 1024 * 1024          # 10 MB
    _UPLOAD_ALLOWED_EXT = frozenset({"xlsx", "xls", "xlsm", "csv"})

    @app.post("/upload-document", response_model=UploadDocumentResponse)
    @limiter.limit(answer_limit)
    async def upload_document(
        request: Request,
        session_id: str = Form(..., min_length=1, max_length=64),
        file: UploadFile = File(...),
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Parse an Excel or CSV fleet roster and merge vehicles+drivers into a session.

        Fleet upload requires the session's LOB to be commercial_auto (or unset,
        in which case it is set automatically). Returns 422 if the session is
        already locked to a different LOB.
        """
        from accord_ai.extraction.fleet_ingest import (
            merge_fleet_into_submission,
            parse_fleet_sheet,
        )
        from accord_ai.schema import CustomerSubmission

        tenant = _auth_tenant(request)

        # Extension check before reading bytes (cheap)
        filename = file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _UPLOAD_ALLOWED_EXT:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": (
                        f"unsupported file type {ext!r}; "
                        f"allowed: {sorted(_UPLOAD_ALLOWED_EXT)}"
                    )
                },
            )

        # Read with size cap — avoids zip-bomb memory pressure
        content = await file.read(_UPLOAD_MAX_BYTES + 1)
        if len(content) > _UPLOAD_MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"file too large (max {_UPLOAD_MAX_BYTES // 1_048_576} MB)"
                    )
                },
            )

        # Parse fleet sheet
        fleet = await asyncio.to_thread(parse_fleet_sheet, content, filename)

        # Session existence + tenant check
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        # Merge into submission dict (raises ValueError on wrong LOB)
        sub_dict = session.submission.model_dump(mode="json")
        try:
            counts = merge_fleet_into_submission(sub_dict, fleet)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": str(exc)},
            )

        # Persist updated submission
        new_submission = CustomerSubmission.model_validate(sub_dict)
        intake.store.update_submission(session_id, new_submission, tenant=tenant)

        _logger.info(
            "upload-document: session=%s file=%s vehicles=%d+%d drivers=%d+%d",
            session_id, filename,
            counts["vehicles_added"], counts["vehicles_updated"],
            counts["drivers_added"], counts["drivers_updated"],
        )

        return UploadDocumentResponse(
            session_id=session_id,
            filename=filename,
            drivers_added=counts["drivers_added"],
            drivers_updated=counts["drivers_updated"],
            vehicles_added=counts["vehicles_added"],
            vehicles_updated=counts["vehicles_updated"],
            header_row_idx=fleet.header_row_idx,
            warnings=fleet.warnings,
        )

    @app.post("/complete", response_model=CompleteResponse)
    @limiter.limit(complete_limit)
    async def complete(
        req: CompleteRequest,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Fill every form required by the submission's LOB; write
        deduplicated bytes to the filled-PDF store; return metadata.

        v3 wire-compat: accepts `submission_id` (we call it session_id
        internally), plus `tenant_slug` and a handful of telemetry
        fields. The `fields_data` override payload is merged into mapper
        output before filling (P10.0.f.3) — keyed by form_number, each
        value a widget→override map. Caller values win over the mapper;
        empty-string values clear a widget.
        """
        tenant = _auth_tenant(request)
        # Body tenant_slug is honored only when the middleware didn't bind
        # one (admin key / auth disabled). Bound-key tenant always wins.
        if tenant is None and req.tenant_slug:
            tenant = req.tenant_slug

        # Session existence + tenant check up front. KeyError → 404 via
        # the global handler; tenant-leak-safe (wrong-tenant looks identical
        # to missing).
        session = intake.store.get_session(req.submission_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {req.submission_id}")

        # v3 wire contract (verified accord_ai_v3/api.py:1723 + fe_label_map.py):
        # the FE sends fields_data keyed by HUMAN LABEL ("Named Insured",
        # "Driver First Name A") mixed with raw widget names. v3 routes
        # every form through `translate_payload` which:
        #   * passes widget-shaped keys (Text##[0] or PascalCase_*) through
        #   * maps known labels to widget names via LABEL_TO_WIDGET
        #   * filters null sentinels ("NullObject", "null", etc.)
        # Without this translation, every label-keyed edit silently drops.
        #
        # Form 163 layers a STRUCTURED shape on top (nested _header dict +
        # drivers list). `map_structured` expands those into flat widget
        # names; the two dicts merge with translated-wins so a raw
        # Text##[0] override beats the derived expansion. v3 parity:
        # accord_ai_v3/api.py:1744-1751.
        from accord_ai.forms.fe_label_map import translate_payload
        from accord_ai.forms.layouts.form_163_layout import map_structured
        from accord_ai.forms.pipeline import _FORM_KEY_RE

        overrides_in: Dict[str, Dict[str, object]] = {}
        for form_key, widget_map in req.fields_data.items():
            if not isinstance(widget_map, dict):
                continue
            m = _FORM_KEY_RE.match(str(form_key))
            form_num = m.group(1) if m else None

            # Structured payloads (_header / drivers) contain dict/list
            # values that str()-ify to garbage in translate_payload's
            # passthrough branch. Filter to scalars before translating, so
            # Text##[0] + label-keyed scalars still get translated while
            # the nested structures flow through map_structured below.
            scalar_values = {
                k: v for k, v in widget_map.items()
                if not isinstance(v, (dict, list))
            }
            translated, unknown = translate_payload(scalar_values)
            if unknown:
                _logger.info(
                    "complete: form=%s %d unknown-label keys dropped "
                    "(sample=%s)",
                    form_key, len(unknown), unknown[:5],
                )

            if form_num == "163":
                # Run structured expansion on the RAW payload (not scalars
                # only) — map_structured reads _header / drivers nested
                # values. translated keys win: a caller who overrode
                # Text14[0] directly beats the address-composition logic.
                structured = map_structured(widget_map)
                merged = {**structured, **translated}
                overrides_in[form_key] = {
                    k: v for k, v in merged.items()
                    if v is not None and str(v) != ""
                }
            else:
                overrides_in[form_key] = translated
        overrides_arg = overrides_in or None

        # Offload PDF work — fill_submission is CPU-bound (PyMuPDF).
        filled = await asyncio.to_thread(
            fill_submission,
            session.submission,
            field_overrides=overrides_arg,
        )

        # Audit the override payload shape — emitted ONLY when fields_data
        # is non-empty so we don't spam the log on every vanilla /complete.
        # Payload captures the form keys + per-form override counts +
        # count of "clear this widget" (empty-string) entries, which is
        # enough to trace "did broker X blank out a field" without leaking
        # the corrected values themselves.
        if overrides_in:
            override_counts = {
                form_key: len(widgets)
                for form_key, widgets in overrides_in.items()
            }
            empty_overrides = sum(
                1
                for widgets in overrides_in.values()
                for v in widgets.values()
                if isinstance(v, str) and v == ""
            )
            record_audit_event(
                intake.store,
                COMPLETE_OVERRIDES_APPLIED,
                session_id=req.submission_id,
                tenant=tenant,
                payload={
                    "form_numbers":    list(overrides_in.keys()),
                    "override_counts": override_counts,
                    "empty_overrides": empty_overrides,
                },
            )

        # --- Local save first ---
        # save() without drive_file_id records content_hash + drive_file_id=None
        # on a miss. The Drive pipeline below calls set_drive_file_id to stamp
        # the uploaded IDs onto those same manifest entries — doing save() AFTER
        # the pipeline would clobber them back to None on a dedup miss.
        saved_flags: Dict[str, bool] = {}
        total_bytes = 0
        total_written = 0
        for form_number, ff in filled.items():
            wrote = await asyncio.to_thread(
                intake.filled_pdf_store.save,
                req.submission_id, tenant, form_number,
                ff.pdf_bytes, ff.content_hash,
            )
            saved_flags[form_number] = wrote
            total_bytes += len(ff.pdf_bytes)
            if wrote:
                total_written += 1

        # --- Drive upload pipeline (optional) ---
        # Every failure mode resolves to a drive_status on the per-form
        # outcome; the pipeline never raises. Local /pdf fallback serves
        # files when Drive is unreachable.
        drive_outcome: Optional[DriveUploadOutcome] = None
        if (
            intake.backend_client is not None
            and intake.drive_client is not None
        ):
            drive_outcome = await run_drive_upload_pipeline(
                backend=intake.backend_client,
                drive=intake.drive_client,
                filled_pdf_store=intake.filled_pdf_store,
                submission_id=req.submission_id,
                tenant=tenant,
                tenant_domain=req.tenant_domain,
                tenant_slug=req.tenant_slug or (tenant or ""),
                filled=filled,
                fields_data=req.fields_data,
                audit_store=intake.store,
            )

        # --- Aggregate per-form metadata for the response ---
        drive_by_form: Dict[str, tuple] = {}
        if drive_outcome is not None:
            for o in drive_outcome.form_outcomes:
                drive_by_form[o.form_number] = (
                    o.drive_status, o.drive_file_id, o.drive_view_url,
                )

        infos: List[CompleteFormInfo] = []
        for form_number, ff in filled.items():
            drive_status, drive_file_id, drive_view_url = drive_by_form.get(
                form_number,
                (DRIVE_STATUS_SKIPPED, None, None),
            )
            wrote = saved_flags[form_number]
            infos.append(CompleteFormInfo(
                form_number=ff.form_number,
                content_hash=ff.content_hash,
                byte_length=len(ff.pdf_bytes),
                dedup_skipped=not wrote,
                fill_result=ff.fill_result.to_dict(),
                drive_status=drive_status,
                drive_file_id=drive_file_id,
                drive_view_url=drive_view_url,
            ))

        record_audit_event(
            intake.store,
            SUBMISSION_COMPLETED,
            session_id=req.submission_id,
            tenant=tenant,
            payload={
                "forms":         [f.form_number for f in infos],
                "total_bytes":   total_bytes,
                "total_written": total_written,
                # Pre-declared so future FE-override audits can grep here.
                "fields_data_keys": list(req.fields_data.keys()),
            },
        )

        # --- v3 wire-compat projections (P10.0.g.3) ------------------------
        # Per-form fill summary list matching v3's api.py:1900-1928 shape:
        # one dict per form with filled/skipped counts + dedup flag. v3 FEs
        # render progress bars off this; v4 keeps it derivable from FillResult
        # + saved_flags so we avoid parallel state.
        fill_summary = [
            {
                "form_number":               info.form_number,
                "pdf_id":                    f"ACORD_{info.form_number}",
                "fields_filled":             info.fill_result.get("filled", 0),
                "fields_skipped":            info.fill_result.get("skipped", 0),
                "errors":                    info.fill_result.get("error_messages", [])[:3],
                "skipped_upload_unchanged":  info.dedup_skipped,
            }
            for info in infos
        ]
        # Drive uploaded-file descriptors. v3 returns a list of dicts
        # {pdf_id, file_id, view_url}; skip forms that weren't uploaded.
        drive_files = [
            {
                "pdf_id":   f"ACORD_{info.form_number}",
                "file_id":  info.drive_file_id,
                "view_url": info.drive_view_url,
            }
            for info in infos
            if info.drive_file_id is not None
        ]
        uploaded_count = sum(
            1 for info in infos
            if info.drive_status in (
                DRIVE_STATUS_UPLOADED, DRIVE_STATUS_OVERWRITTEN,
            )
        )
        dedup_skipped_count = sum(1 for info in infos if info.dedup_skipped)

        return CompleteResponse(
            # v3 shape:
            submission_id=req.submission_id,
            uploaded=uploaded_count,
            total=len(infos),
            field_diff={},          # stub — delta tracker not yet wired
            fill_summary=fill_summary,
            drive_files=drive_files,
            validation={},          # stub — validation module not yet wired
            cache={
                "auth_hit":              False,
                "validation_hit":        False,
                "content_dedup_skipped": dedup_skipped_count,
                "file_id_hits":          0,
            },
            pruned=(
                drive_outcome.pruned_count
                if drive_outcome is not None else 0
            ),
            # v4 native detail:
            session_id=req.submission_id,
            forms=infos,
            total_bytes=total_bytes,
            total_written=total_written,
            drive_enabled=(
                drive_outcome.drive_enabled
                if drive_outcome is not None else False
            ),
            drive_folder_id=(
                drive_outcome.drive_folder_id
                if drive_outcome is not None else None
            ),
            pruned_count=(
                drive_outcome.pruned_count
                if drive_outcome is not None else 0
            ),
        )

    @app.get("/pdf/{session_id}/{form_number}")
    async def get_filled_pdf(
        session_id: str,
        form_number: str,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Serve a filled PDF. Tenant-scoped — wrong tenant is
        indistinguishable from missing, and session existence is checked
        BEFORE the filesystem so an attacker with a valid key cannot
        enumerate other tenants' session IDs by watching 200 vs 404.
        """
        tenant = _auth_tenant(request)

        # Session-level existence + tenant check first. Uniform 404 surface.
        if intake.store.get_session(session_id, tenant=tenant) is None:
            raise KeyError(f"session not found: {session_id}")

        # FilledPdfStore validates session_id/form_number shape; a malformed
        # path segment raises ValueError, not FileNotFoundError. Map to 404.
        try:
            pdf_bytes = intake.filled_pdf_store.load(
                session_id, tenant, form_number,
            )
        except ValueError:
            raise KeyError(f"pdf not found: {session_id}/{form_number}")

        if pdf_bytes is None:
            raise KeyError(f"pdf not found: {session_id}/{form_number}")

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "content-disposition":
                    f'inline; filename="acord_{form_number}_filled.pdf"',
            },
        )

    @app.get("/session/{session_id}", response_model=SessionDetailResponse)
    async def get_session_detail(
        session_id: str,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return SessionDetailResponse(
            session_id=session.session_id,
            tenant=session.tenant,
            status=session.status,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            submission=session.submission.model_dump(mode="json"),
        )

    @app.get(
        "/session/{session_id}/messages", response_model=MessagesResponse
    )
    async def get_session_messages(
        session_id: str,
        request: Request,
        limit: Optional[int] = None,
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        # Existence + tenant check — uniform 404 on missing / wrong-tenant
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        messages = intake.store.get_messages(
            session_id, tenant=tenant, limit=min(limit, 1000) if limit else 1000,
        )
        return MessagesResponse(
            session_id=session_id,
            messages=[
                MessageResponse(
                    message_id=m.message_id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at.isoformat(),
                )
                for m in messages
            ],
        )

    @app.get("/sessions", response_model=SessionsResponse)
    async def list_sessions_endpoint(
        request: Request,
        status: Optional[Literal["active", "finalized", "expired"]] = None,
        limit: int = Query(default=50, ge=1, le=500),
        intake: IntakeApp = Depends(_get_intake),
    ):
        tenant = _auth_tenant(request)
        summaries = intake.store.list_sessions(
            tenant=tenant, status=status, limit=limit,
        )
        return SessionsResponse(
            sessions=[
                SessionSummaryResponse(
                    session_id=s.session_id,
                    tenant=s.tenant,
                    status=s.status,
                    created_at=s.created_at.isoformat(),
                    updated_at=s.updated_at.isoformat(),
                )
                for s in summaries
            ],
        )

    # ---- Phase 1.9 endpoints (/explain, /enrich, /correction, /feedback) ----

    @app.get("/explain/{field_path}", response_model=ExplainResponse)
    async def explain_field(
        field_path: str,
        request: Request,
    ):
        """Return a knowledge-base explanation for a submission field.

        field_path must be a known top-level field of CustomerSubmission.
        Uses a 300-second TTL cache keyed by (tenant, field_path).
        """
        from accord_ai.schema import CustomerSubmission

        # Validate that the root segment names a real field.
        root = field_path.split(".")[0]
        if root not in CustomerSubmission.model_fields:
            return JSONResponse(
                status_code=404,
                content={"detail": f"unknown field: {field_path!r}"},
            )

        tenant = _auth_tenant(request) or "default"
        hits = await _explain_retrieve(settings, tenant, field_path)

        if not hits:
            return ExplainResponse(
                field=field_path,
                explanation=f"No knowledge-base entries found for '{field_path}'.",
                sources=[],
            )

        top = hits[0]
        explanation = top.metadata.get("explanation") or top.document[:500]
        sources = [
            ExplainSource(
                title=str(h.metadata.get("title") or h.doc_id),
                snippet=h.document[:300],
                score=float(h.distance),
            )
            for h in hits
        ]
        return ExplainResponse(field=field_path, explanation=explanation, sources=sources)

    @app.post("/enrich", response_model=EnrichResponse)
    async def enrich(
        req: EnrichRequest,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Run the validation engine against the current session submission.

        Results are cached 10 min per (session_id, submission_hash) — rapid
        back-to-back /enrich calls on an unchanged submission are free.
        """
        import json

        tenant = _auth_tenant(request)
        session = intake.store.get_session(req.session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {req.session_id}")

        sub = session.submission
        sub_hash = hash_bytes(
            json.dumps(sub.model_dump(mode="json"), sort_keys=True).encode()
        )
        cache_key = f"{req.session_id}:{sub_hash}"

        with _enrich_lock:
            cached_results = _enrich_cache.get(cache_key)
        if cached_results is not None:
            return EnrichResponse(
                validators_run=len(cached_results),
                results=cached_results,
                cached=True,
            )

        engine = request.app.state.validation_engine
        results = await engine.run_all(sub)

        results_json = [r.model_dump(mode="json") for r in results]
        with _enrich_lock:
            _enrich_cache[cache_key] = results_json

        return EnrichResponse(
            validators_run=len(results),
            results=results_json,
            cached=False,
        )

    @app.get("/review/{session_id}")
    async def get_review(
        session_id: str,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Finalize review payload for a session.

        Runs the full validation engine and transforms results into a FE-ready
        ReviewPayload grouped by: conflicts, compliance checks, warnings, info,
        and prefill suggestions.

        Finalized sessions skip re-running validators (returns cached payload or
        snapshot from submission state only — prevents wasted API quota).
        Cached 10 min per (session_id, submission_hash). Safe to poll.
        """
        import json

        from accord_ai.validation.review import build_review_payload

        tenant = _auth_tenant(request)
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        sub = session.submission
        sub_hash = hash_bytes(
            json.dumps(sub.model_dump(mode="json"), sort_keys=True).encode()
        )
        cache_key = f"{session_id}:{sub_hash}"

        with _review_lock:
            cached_payload = _review_cache.get(cache_key)
        if cached_payload is not None:
            return cached_payload

        if session.status == "finalized":
            # Don't re-run validators for completed sessions — burn no API quota.
            # Build payload from submission state only (conflicts still visible).
            payload = build_review_payload(session_id=session_id, submission=sub, results=[])
            with _review_lock:
                _review_cache[cache_key] = payload
            return payload

        engine = request.app.state.validation_engine
        results = await engine.run_all(sub)

        payload = build_review_payload(session_id=session_id, submission=sub, results=results)
        with _review_lock:
            _review_cache[cache_key] = payload
        return payload

    @app.post("/review/{session_id}/resolve")
    async def resolve_conflict(session_id: str, request: Request):
        """Stub — conflict resolution ships in Phase 3 (conversation controller)."""
        from fastapi import HTTPException
        raise HTTPException(status_code=501, detail="conflict resolution ships in Phase 3")

    @app.post("/correction", response_model=CapturedResponse)
    async def correction(
        req: CorrectionRequest,
        request: Request,
    ):
        """Persist a field correction record to SQLite via CorrectionCollector."""
        from accord_ai.feedback.collector import CorrectionCollector

        tenant = _auth_tenant(request) or "default"
        collector: CorrectionCollector = request.app.state.correction_collector
        record_id = collector.record_correction(
            tenant=tenant,
            session_id=req.session_id,
            turn=req.turn,
            field_path=req.field_path,
            wrong_value=req.wrong_value,
            correct_value=req.correct_value,
            explanation=req.explanation,
        )
        return CapturedResponse(captured=True, id=record_id)

    @app.post("/feedback", response_model=CapturedResponse)
    async def feedback(
        req: FeedbackRequest,
        request: Request,
    ):
        """Persist a turn-level feedback rating to SQLite via CorrectionCollector."""
        from accord_ai.feedback.collector import CorrectionCollector

        tenant = _auth_tenant(request) or "default"
        collector: CorrectionCollector = request.app.state.correction_collector
        record_id = collector.record_feedback(
            tenant=tenant,
            session_id=req.session_id,
            turn=req.turn,
            rating=req.rating,
            comment=req.comment,
        )
        return CapturedResponse(captured=True, id=record_id)

    # ---- Phase 2.3 — DPO admin endpoints ------------------------------------

    @app.post("/admin/dpo/export/{tenant}")
    async def dpo_export(
        tenant: str,
        request: Request,
        body: Optional[DPOExportRequest] = None,
    ):
        """Trigger DPO training-pair export for a tenant (admin key required)."""
        from accord_ai.feedback.dpo import DPOManager

        from fastapi import HTTPException

        is_admin = getattr(request.state, "is_admin", False)
        if not is_admin:
            return JSONResponse(
                status_code=403,
                content={"detail": "admin key required for /admin endpoints"},
            )

        mgr: DPOManager = request.app.state.dpo_manager
        force = body.force if body is not None else False

        if not force and not mgr.eligible_for_training(tenant):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "below_threshold",
                    "pending": mgr.count_pending(tenant),
                    "threshold": mgr._threshold,
                },
            )

        result = mgr.export(tenant)
        return {
            "tenant": result.tenant,
            "path": str(result.path) if result.path else None,
            "count": result.count,
            "eligible_for_training": result.count > 0,
        }

    @app.get("/admin/dpo/status/{tenant}")
    async def dpo_status(
        tenant: str,
        request: Request,
    ):
        """Return pending/graduated counts and last export info (admin key required)."""
        from accord_ai.feedback.dpo import DPOManager

        is_admin = getattr(request.state, "is_admin", False)
        if not is_admin:
            return JSONResponse(
                status_code=403,
                content={"detail": "admin key required for /admin endpoints"},
            )

        mgr: DPOManager = request.app.state.dpo_manager
        return mgr.status(tenant)

    # ---- Phase 1.10 endpoints ------------------------------------------------

    # 9.1 — /upload-image (Phase 1.5 — real OCR implementation)
    @app.post("/upload-image", response_model=UploadImageResponse)
    @limiter.limit(answer_limit)
    async def upload_image(
        request: Request,
        kind: str = Form(...),
        session_id: str = Form(..., min_length=1, max_length=64),
        file: UploadFile = File(...),
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Extract structured fields from an uploaded image via OCR + LLM parse."""
        from accord_ai.core.vehicle_merge import merge_drivers, merge_vehicles
        from accord_ai.schema import CustomerSubmission

        tenant = _auth_tenant(request)

        if kind not in _IMAGE_ALLOWED_KIND:
            return JSONResponse(
                status_code=422,
                content={"detail": f"invalid kind {kind!r}; allowed: {sorted(_IMAGE_ALLOWED_KIND)}"},
            )
        filename = file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in _IMAGE_ALLOWED_EXT:
            return JSONResponse(
                status_code=422,
                content={"detail": f"unsupported image type {ext!r}; allowed: {sorted(_IMAGE_ALLOWED_EXT)}"},
            )
        content = await file.read(_UPLOAD_MAX_BYTES + 1)
        if len(content) > _UPLOAD_MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "file too large (max 10 MB)"},
            )

        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        # LOB guard — vehicle_registration only makes sense for CA
        if kind == "vehicle_registration":
            lob = None
            if session.submission.lob_details is not None:
                lob = getattr(session.submission.lob_details, "lob", None)
            if lob is not None and lob != "commercial_auto":
                return JSONResponse(
                    status_code=422,
                    content={"detail": "vehicle_registration upload requires a commercial_auto session"},
                )

        engine = intake.engine
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "LLM engine not available"},
            )

        try:
            fields = await ocr_document(content, kind, engine)  # type: ignore[arg-type]
        except OCRReadError as exc:
            return JSONResponse(
                status_code=422,
                content={"detail": {"error": "ocr_failed", "reason": str(exc)}},
            )
        except OCRConfigError as exc:
            return JSONResponse(
                status_code=503,
                content={"detail": {"error": "ocr_unavailable", "reason": str(exc)}},
            )

        # Log OCR text at DEBUG — PII-redacted
        _logger.debug(
            "upload-image: session=%s kind=%s ocr_fields=%s",
            session_id, kind,
            redact_pii_text(str(fields.model_dump(exclude_none=True))),
        )

        # Merge extracted fields into session submission
        sub = session.submission
        new_sub = _merge_ocr_into_submission(sub, fields, kind, merge_drivers, merge_vehicles)

        intake.store.update_submission(session_id, new_sub, tenant=tenant)

        _logger.info(
            "upload-image: session=%s kind=%s fields=%d merged",
            session_id, kind, len(fields.model_dump(exclude_none=True)),
        )

        extracted = fields.model_dump(exclude_none=True)
        return UploadImageResponse(
            status="ok",
            kind=kind,
            extracted=extracted,
            note="",
        )

    # 9.2 — /upload-filled-pdfs
    @app.post("/upload-filled-pdfs", response_model=UploadFilledPdfsResponse)
    @limiter.limit(answer_limit)
    async def upload_filled_pdfs(
        request: Request,
        session_id: str = Form(..., min_length=1, max_length=64),
        files: List[UploadFile] = File(...),
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Accept one or more filled PDFs; save locally + optionally push to Drive."""
        tenant = _auth_tenant(request)
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        uploaded: List[UploadedPdfInfo] = []
        failed: List[FailedPdfInfo] = []

        for uf in files:
            fname = uf.filename or "unknown.pdf"
            m = _FORM_FROM_FILENAME_RE.search(fname)
            if m is None:
                failed.append(FailedPdfInfo(
                    form_number=fname,
                    error="cannot extract form number from filename",
                ))
                continue
            form_number = m.group(1)

            content = await uf.read(_UPLOAD_MAX_BYTES + 1)
            if len(content) > _UPLOAD_MAX_BYTES:
                failed.append(FailedPdfInfo(form_number=form_number, error="file too large (max 10 MB)"))
                continue
            if not content.startswith(b"%PDF-"):
                failed.append(FailedPdfInfo(form_number=form_number, error="not a valid PDF (missing %PDF- header)"))
                continue

            content_hash = hash_bytes(content)
            await asyncio.to_thread(
                intake.filled_pdf_store.save,
                session_id, tenant, form_number, content, content_hash,
            )
            uploaded.append(UploadedPdfInfo(form_number=form_number))

        return UploadFilledPdfsResponse(uploaded=uploaded, failed=failed)

    # 9.3 — /upload-blank-pdfs (Phase 1.5 stub — no blank-PDF Drive uploader yet)
    @app.post("/upload-blank-pdfs", response_model=UploadBlankPdfsResponse)
    async def upload_blank_pdfs(
        req: UploadBlankPdfsRequest,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Stub: blank PDF seeding ships in Phase 1.5."""
        from accord_ai.forms.registry import UnknownFormError, forms_for_lob

        tenant = _auth_tenant(request)
        session = intake.store.get_session(req.session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {req.session_id}")

        try:
            available = forms_for_lob(req.lob)
        except UnknownFormError:
            return JSONResponse(
                status_code=422,
                content={"detail": f"unknown LOB: {req.lob!r}"},
            )

        return UploadBlankPdfsResponse(
            uploaded=[],
            failed=[],
            note="Blank PDF seeding ships in Phase 1.5",
        )

    # 9.4 — /debug/session/{session_id} (admin-only)
    @app.get("/debug/session/{session_id}", response_model=DebugSessionResponse)
    async def debug_session(
        session_id: str,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Full session state dump — admin key only. PII-redacted."""
        is_admin = getattr(request.state, "is_admin", False)
        if not is_admin:
            return JSONResponse(
                status_code=403,
                content={"detail": "admin key required for /debug endpoints"},
            )

        tenant = _auth_tenant(request)
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        messages = intake.store.get_messages(session_id, tenant=tenant)
        turns = [
            DebugTurn(
                turn_idx=i,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
            )
            for i, m in enumerate(messages)
        ]

        import json
        raw = json.dumps(
            {
                "session": {
                    "session_id": session.session_id,
                    "tenant": session.tenant,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                },
                "submission": session.submission.model_dump(mode="json"),
                "turns": [t.model_dump() for t in turns],
                "harness_version": "1.0-declarative",
                "last_refiner_stage": "none",
            },
            default=str,
        )
        redacted = redact_pii_text(raw) or raw
        return JSONResponse(content=json.loads(redacted))

    # 9.5 — /fields/{session_id}
    @app.get("/fields/{session_id}")
    async def get_fields(
        session_id: str,
        request: Request,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Return submission in FE-facing shape (full CustomerSubmission JSON).

        Includes `conflicts` — enrichment mismatches accumulated inline during
        the session, surfaced for the finalize review screen.
        """
        tenant = _auth_tenant(request)
        session = intake.store.get_session(session_id, tenant=tenant)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return JSONResponse(content=session.submission.model_dump(mode="json"))

    # 9.6 — /harness family

    @app.get("/harness", response_model=HarnessResponse)
    async def get_harness():
        """Current effective ruleset — declarative v4 harness."""
        from accord_ai.harness.critical_fields import get_critical_fields

        lobs = ["commercial_auto", "general_liability", "workers_comp"]
        critical_per_lob = {
            lob: [path for path, _ in get_critical_fields(lob)]
            for lob in lobs
        }
        return HarnessResponse(
            version="1.0-declarative",
            critical_fields_per_lob=critical_per_lob,
            active_rules=["negation"],
            refiner_harness_enabled=settings.harness_max_refines > 0,
        )

    @app.get("/harness/audit", response_model=HarnessAuditResponse)
    async def get_harness_audit(
        request: Request,
        session_id: Optional[str] = None,
        intake: IntakeApp = Depends(_get_intake),
    ):
        """Session-level harness stats. Per-turn refinement tracking ships in Phase D."""
        tenant = _auth_tenant(request)
        if session_id is not None:
            session = intake.store.get_session(session_id, tenant=tenant)
            if session is None:
                raise KeyError(f"session not found: {session_id}")
        return HarnessAuditResponse(
            session_id=session_id,
            refinement_count=0,
            judge_pass_rate=None,
            most_failed_paths=[],
            note="Per-session harness metrics ship in Phase D",
        )

    @app.get("/harness/history", response_model=HarnessHistoryResponse)
    async def get_harness_history():
        """Stub: living harness history ships in Phase D."""
        from datetime import datetime, timezone

        return HarnessHistoryResponse(
            versions=[
                {
                    "version": "1.0-declarative",
                    "activated_at": datetime.now(tz=timezone.utc).isoformat(),
                },
            ],
            note="Living harness history ships in Phase D",
        )

    @app.post("/harness/rollback")
    async def harness_rollback():
        """Not implementable without living harness history (Phase D)."""
        return JSONResponse(
            status_code=501,
            content={"detail": "Living harness rollback ships in Phase D"},
        )

    @app.get("/harness/provenance", response_model=HarnessProvenanceResponse)
    async def get_harness_provenance():
        """Stub: refinement provenance ships in Phase D."""
        return HarnessProvenanceResponse(
            entries=[],
            note="Refinement provenance ships in Phase D",
        )

    @app.get("/harness/review-queue", response_model=HarnessReviewQueueResponse)
    async def get_harness_review_queue():
        """Stub: review queue ships in Phase D."""
        return HarnessReviewQueueResponse(
            queue=[],
            note="Review queue ships in Phase D",
        )

    # CORS — added last so it becomes the outermost layer; preflight OPTIONS
    # is handled by CORSMiddleware before our auth middleware sees it.
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]

    # Wildcard + credentials is invalid per CORS spec (browsers refuse
    # credentialed requests to "*"). Downgrade credentials quietly when
    # operators leave the default — log so they know.
    if "*" in origins:
        allow_credentials = False
        _logger.warning(
            "CORS: allow_origins contains wildcard '*'; disabling "
            "allow_credentials for spec compliance"
        )
    else:
        allow_credentials = True

    cors_kwargs: dict = {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        # Make trace headers readable to CORS clients (browsers hide
        # non-safelisted response headers without this).
        "expose_headers": ["x-request-id", "x-tenant-slug"],
    }
    if settings.allowed_origin_regex:
        cors_kwargs["allow_origin_regex"] = settings.allowed_origin_regex
    app.add_middleware(CORSMiddleware, **cors_kwargs)

    return app
