"""LLM layer — provider adapters + retry policy.

Privacy boundary (non-negotiable):
    Default LLM_BASE_URL is localhost (vLLM serving Qwen/Qwen3.5-9B).
    Raw customer data must NOT flow to external providers without
    explicit per-tenant opt-in. The external refiner path (Phase 5) is
    the sole exception and is gated by both a per-refiner config AND
    the ACCORD_ALLOW_EXTERNAL_LLM environment flag (see build_refiner_engine).
    If you're tempted to point LLM_BASE_URL at a public provider to test
    something — don't. Use a dedicated env + synthetic data, not production.
"""
import os
from urllib.parse import urlparse

from accord_ai.config import Settings
from accord_ai.llm.engine import Engine, EngineResponse, Message
from accord_ai.llm.fake_engine import FakeEngine
from accord_ai.llm.openai_engine import OpenAIEngine
from accord_ai.llm.retrying_engine import RetryingEngine

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


class PrivacyBoundaryError(ValueError):
    """Raised when the refiner's resolved base_url is non-localhost AND the
    ACCORD_ALLOW_EXTERNAL_LLM environment opt-in is absent.

    Ships customer data off-box only when the operator has explicitly
    acknowledged the boundary via env var.
    """


def _is_localhost(base_url: str) -> bool:
    """True if `base_url` points at a loopback / any-interface host."""
    return urlparse(base_url).hostname in _LOCALHOST_HOSTS


def build_engine(settings: Settings) -> Engine:
    """Production engine stack: OpenAI-compat adapter + retry wrapper."""
    return RetryingEngine(OpenAIEngine(settings), settings)


def build_refiner_engine(settings: Settings) -> Engine:
    """Refiner-specific engine stack. Per-field fallback to main llm_* settings.

    Raises PrivacyBoundaryError if the resolved base_url isn't localhost
    AND ACCORD_ALLOW_EXTERNAL_LLM isn't set in the environment.
    """
    # Per-field fallback. Explicit `is not None` check so a legitimate
    # falsy override (e.g., timeout=0.0 for "no timeout") isn't silently
    # replaced with the main settings' value.
    refiner_settings = settings.model_copy(update={
        "llm_base_url":  (settings.harness_refiner_base_url
                          if settings.harness_refiner_base_url is not None
                          else settings.llm_base_url),
        "llm_model":     (settings.harness_refiner_model
                          if settings.harness_refiner_model is not None
                          else settings.llm_model),
        "llm_api_key":   (settings.harness_refiner_api_key
                          if settings.harness_refiner_api_key is not None
                          else settings.llm_api_key),
        "llm_timeout_s": (settings.harness_refiner_timeout_s
                          if settings.harness_refiner_timeout_s is not None
                          else settings.llm_timeout_s),
    })

    if not _is_localhost(refiner_settings.llm_base_url):
        if not os.environ.get("ACCORD_ALLOW_EXTERNAL_LLM"):
            raise PrivacyBoundaryError(
                f"refiner base_url {refiner_settings.llm_base_url!r} is not "
                "localhost. Set ACCORD_ALLOW_EXTERNAL_LLM=1 to acknowledge "
                "that customer data may leave the local machine."
            )

    return RetryingEngine(OpenAIEngine(refiner_settings), refiner_settings)


__all__ = [
    "Engine",
    "EngineResponse",
    "Message",
    "FakeEngine",
    "OpenAIEngine",
    "PrivacyBoundaryError",
    "RetryingEngine",
    "build_engine",
    "build_refiner_engine",
]
