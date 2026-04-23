from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionMode(str, Enum):
    """Schema-enforcement mode for the extraction LLM call (Step 25 experiment)."""
    XGRAMMAR = "xgrammar"      # A — current: vLLM guided_json via xgrammar
    JSON_OBJECT = "json_object" # B — OpenAI response_format json_object, no schema constraint
    FREE = "free"               # D — no format constraint; extractor parses first JSON block


class Settings(BaseSettings):
    """Env-driven config. Defaults production-safe; override via env vars.

    Note: log_level / log_format are case-sensitive by value. Use canonical
    upper-case for log_level ("DEBUG", "INFO", "WARNING", "ERROR").
    """

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    # --- API / storage ---
    api_port: int = Field(default=1505)
    db_path: str = Field(default="accord_ai.db")
    # Root directory for filled ACORD PDFs. Per-tenant / per-session subdirs
    # + manifest.json dedup — see accord_ai.forms.storage.FilledPdfStore.
    filled_pdf_dir: str = Field(default="filled_pdfs")

    # --- Logging ---
    log_dir: str = Field(default="logs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["text", "json"] = Field(default="text")
    pii_redaction: bool = Field(default=True)

    # --- LLM (Engine) ---
    # api_key=None means "local dev / vLLM ignores auth" — engine translates
    # None to "sk-unused" at client construction so the SDK accepts it.
    llm_api_key: Optional[SecretStr] = Field(default=None)
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_model: str = Field(default="insurance-agent")
    llm_timeout_s: float = Field(default=30.0)

    # --- LLM retry policy (consumed by RetryingEngine) ---
    # Equal-jitter exponential: min(base * 2**attempt, cap) + U(0, base).
    # Defaults → worst-case total wait ≈ 0.5 + 1 + 2 = 3.5s across 3 retries.
    llm_retries: int = Field(default=3)
    llm_retry_base_s: float = Field(default=0.5)
    llm_retry_cap_s: float = Field(default=8.0)

    # --- Harness (judge + refine) ---
    # max_refines=0 disables refinement entirely.
    # >=1 caps the judge→refine loop; it exits early on verdict.passed.
    harness_max_refines: int = Field(default=1, ge=0)

    # --- Boot warmup ---
    # Fire a one-shot dummy extraction + responder call on FastAPI startup
    # so vLLM compiles the guided_json grammar and warms the prefix cache
    # before any real user traffic lands. First-request latency without
    # warmup is 5-10x the steady state because xgrammar compilation +
    # CUDA-graph capture run on the hot path. Default False so tests
    # (FakeEngine, no LLM reachable) don't trip warmup; production sets
    # WARMUP_ON_BOOT=true in env.
    warmup_on_boot: bool = Field(default=False)

    # --- Rate limiting (slowapi, per-IP) ---
    # Caps unbounded LLM-cost exposure on public ngrok / tunnel deployments.
    # Every /answer hit is ~$0 on local vLLM but ~many tokens of GPU time;
    # an anonymous attacker with the URL can saturate the 3090 indefinitely
    # without these. Per-IP keying uses X-Forwarded-For when present
    # (behind ngrok / nginx) so the real client is limited, not the proxy.
    #
    # Defaults:
    #   /answer        — 60/minute  (one reasonable broker ≈ 2-3/min, room for
    #                                 slow-typing users; caps hostile hammer)
    #   /complete      — 30/minute  (heavy PDF + Drive work; tighter)
    #   /start-session — 60/minute  (cheap endpoint; still cap to prevent
    #                                 session-creation floods)
    # Set RATE_LIMIT_ENABLED=true to enable. Default off so tests don't
    # need per-IP setup and local dev iterates freely.
    rate_limit_enabled:               bool = Field(default=False)
    rate_limit_answer_per_minute:     int  = Field(default=60,  ge=1)
    rate_limit_complete_per_minute:   int  = Field(default=30,  ge=1)
    rate_limit_start_session_per_minute: int = Field(default=60, ge=1)

    # Refiner engine — per-field fallback to the main llm_* settings.
    # Override any/all to point the refiner at a different provider.
    # Privacy: a non-localhost base_url additionally requires
    # ACCORD_ALLOW_EXTERNAL_LLM=1 in the environment (see build_refiner_engine).
    harness_refiner_base_url: Optional[str] = Field(default=None)
    harness_refiner_model: Optional[str] = Field(default=None)
    harness_refiner_api_key: Optional[SecretStr] = Field(default=None)
    harness_refiner_timeout_s: Optional[float] = Field(default=None)

    # --- API auth + CORS ---
    # Two auth modes coexist (matches v3 wire convention + adds v4 isolation):
    #
    # 1. INTAKE_API_KEY = single shared "admin" key. Caller controls tenant
    #    via X-Tenant-Slug header (or body `tenant_slug` on /start-session).
    #    Backward-compatible with v3.
    #
    # 2. INTAKE_API_KEYS = JSON dict mapping {api_key: tenant_slug}. Each key
    #    is bound to a specific tenant; the binding wins over the header,
    #    and a mismatched X-Tenant-Slug header is rejected with 403.
    #    Tenant-leak-safe — a malicious holder of acme's key cannot read
    #    globex's data even by sending X-Tenant-Slug: globex.
    #
    # If both are configured, INTAKE_API_KEYS is checked first, then the
    # admin key. ACCORD_AUTH_DISABLED bypasses all of this (dev only).
    # ACCORD_CHAT_OPEN opens /start-session, /answer, /finalize for tunnel
    # demos; read endpoints (/sessions, /session/{id}) remain gated.
    intake_api_key: Optional[SecretStr] = Field(default=None)
    intake_api_keys: Dict[str, str] = Field(default_factory=dict)
    accord_auth_disabled: bool = Field(default=False)
    accord_chat_open: bool = Field(default=False)
    allowed_origins: str = Field(default="*")
    allowed_origin_regex: Optional[str] = Field(default=None)


    # --- Knowledge / RAG ---
    knowledge_db_path: str = Field(default="chroma_data")
    knowledge_collection_default: str = Field(default="default")
    knowledge_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    knowledge_embedding_dimension: int = Field(default=384)

    # --- Insurance Backend integration (Phase 10.C) ---
    # The backend lives at https://{tenant_slug}.{backend_host_suffix} and
    # issues service tokens that downstream Drive calls reuse.
    #
    # backend_enabled=False (default) disables the integration entirely —
    # local dev and offline tests continue to work.
    backend_enabled:        bool = Field(default=False)
    backend_host_suffix:    str  = Field(default="copilot.inevo.ai")
    backend_client_id:      str  = Field(default="intake-agent")
    backend_client_secret:  Optional[SecretStr] = Field(default=None)
    backend_timeout_s:      float = Field(default=15.0)
    backend_tls_verify:     bool = Field(default=True)
    # Service-token TTL cache window. v3 uses 10 min against a 15-min token
    # so a refresh lands before expiry. Setting to 0 disables caching.
    backend_token_ttl_s:    int  = Field(default=600, ge=0)

    # --- Drive (Google Drive API — direct calls for folder/file operations) ---
    # Drive access tokens are minted through the backend's
    # /api/v1/drive/service-token endpoint; the backend holds the domain-wide
    # delegation. Direct Drive calls use this as the base URL.
    drive_enabled:   bool  = Field(default=False)
    drive_api_base:  str   = Field(default="https://www.googleapis.com")
    drive_timeout_s: float = Field(default=15.0)

    # --- External validation keys ---
    # USPS Address Validation API v3 (OAuth2 client credentials)
    usps_consumer_key:    Optional[str] = Field(default=None)
    usps_consumer_secret: Optional[str] = Field(default=None)
    # Tax1099 TIN matching API
    tax1099_api_key:      Optional[str] = Field(default=None)
    # FMCSA SAFER API — free key from https://ai.fmcsa.dot.gov/SMS/Docs/DataQ/FMCSA_API.aspx
    fmcsa_web_key:        Optional[str] = Field(default=None)
    # SAM.gov entity information API — free key at https://sam.gov/content/entity-registration
    sam_gov_api_key:      Optional[str] = Field(default=None)
    # SEC EDGAR — no key required but SEC mandates a descriptive User-Agent
    sec_edgar_user_agent: str = Field(default="Accord v4 compliance-check@accord.example")
    # Path to NANPA area codes CSV (see scripts/refresh_area_codes.py)
    area_codes_csv_path:  str = Field(default="data/area_codes.csv")
    # Per-validator timeout for the finalize validation engine
    validation_timeout_s: float = Field(default=10.0)

    # --- DPO training export (Phase 2.3) ---
    training_data_dir: Path = Field(default=Path("training_data"))
    dpo_threshold: int = Field(default=50, ge=1)

    # --- Correction memory injection (Phase 2.4) ---
    # Set ENABLE_CORRECTION_MEMORY=false to disable (A/B test).
    enable_correction_memory: bool = Field(default=True)
    correction_memory_limit: int = Field(default=5, ge=1)
    correction_memory_max_age_days: int = Field(default=30, ge=1)

    # --- Session-transcript capture for SFT (Phase 2.7) ---
    # Set ENABLE_TRANSCRIPT_CAPTURE=false to disable.
    enable_transcript_capture: bool = Field(default=True)

    # --- Deterministic flow engine (Phase 3.2) ---
    # Set USE_FLOW_ENGINE=false to fall back to pure LLM question selection.
    use_flow_engine: bool = Field(default=True)

    # --- Flow-scoped extraction context (Phase 3.3) ---
    # Set EXTRACTION_CONTEXT=false to disable focus-field hints in extraction.
    extraction_context: bool = Field(default=True)

    # --- Step 25 harness compatibility experiment ---
    # These flags are EXPERIMENTAL — default to the current production behavior.
    # Set EXTRACTION_MODE=json_object or =free to test without xgrammar enforcement.
    # Set EXPERIMENT_HARNESS=light or =full to inject v3 harness text into extraction.
    extraction_mode: ExtractionMode = Field(default=ExtractionMode.XGRAMMAR)
    experiment_harness: Literal[
        "none", "light", "full", "core", "core_ca", "core_gl"
    ] = Field(default="none")

    # --- Harness placement (research 2026-04-22) ---
    # "before": harness + SYSTEM_V2 (Step 25 default — regressed on Qwen)
    # "after": SYSTEM_V2 + harness (v3's placement — canonical for Qwen3.5-9B)
    # Research identified this placement as the likely dominant factor in
    # harness regression during Step 25. Default kept at "before" until
    # empirically validated with an eval; flip via HARNESS_POSITION=after.
    harness_position: Literal["before", "after"] = Field(default="before")

    # --- NER post-extraction validation (Thread 3 port, 2026-04-23) ---
    # Enables v3's validate_extraction_with_ner in the postprocess chain.
    # Four fixes: ORG-as-contact removal, PERSON-as-contact injection
    # (gated to single-PERSON turns), ORG-as-business_name injection,
    # URL-as-website injection. Off by default until an eval confirms it
    # composes productively with the current stack. Flip via
    # NER_POSTPROCESS=true.
    ner_postprocess: bool = Field(default=False)

    # --- Reproducibility (Step 25.A variance diagnostic) ---
    # Pin vLLM's internal RNG. None = no seed (current behavior, non-deterministic
    # between runs). Set LLM_SEED=42 to get reproducible outputs at temperature=0.
    llm_seed: Optional[int] = Field(default=None)
