"""5.d — build_refiner_engine factory + privacy gate."""
import pytest

from accord_ai.config import Settings
from accord_ai.llm import (
    OpenAIEngine,
    PrivacyBoundaryError,
    RetryingEngine,
    build_refiner_engine,
)


# --- Localhost default: no gate needed ---

def test_builds_with_main_llm_defaults():
    engine = build_refiner_engine(Settings())
    assert isinstance(engine, RetryingEngine)
    assert isinstance(engine._inner, OpenAIEngine)


def test_localhost_127_passes_without_gate(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    engine = build_refiner_engine(Settings())
    assert isinstance(engine, RetryingEngine)


# --- Non-localhost WITHOUT gate: refuse ---

def test_external_base_url_without_gate_raises(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.delenv("ACCORD_ALLOW_EXTERNAL_LLM", raising=False)
    with pytest.raises(PrivacyBoundaryError, match="ACCORD_ALLOW_EXTERNAL_LLM"):
        build_refiner_engine(Settings())


def test_external_llm_base_url_without_gate_raises(monkeypatch):
    """Even if only the main llm_base_url is external, the refiner inherits it."""
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("ACCORD_ALLOW_EXTERNAL_LLM", raising=False)
    with pytest.raises(PrivacyBoundaryError):
        build_refiner_engine(Settings())


# --- Non-localhost WITH gate: allow ---

def test_external_base_url_with_gate_works(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("ACCORD_ALLOW_EXTERNAL_LLM", "1")
    engine = build_refiner_engine(Settings())
    assert isinstance(engine, RetryingEngine)


# --- Per-field fallback ---

def test_refiner_model_override_applied(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_MODEL", "claude-opus-4.7")
    engine = build_refiner_engine(Settings())
    assert engine._inner._settings.llm_model == "claude-opus-4.7"


def test_refiner_timeout_override_applied(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_TIMEOUT_S", "120.0")
    engine = build_refiner_engine(Settings())
    assert engine._inner._settings.llm_timeout_s == 120.0


def test_refiner_falls_back_to_main_settings_when_unset():
    """No refiner overrides — refiner uses main llm_* values."""
    settings = Settings()
    engine = build_refiner_engine(settings)
    assert engine._inner._settings.llm_base_url == settings.llm_base_url
    assert engine._inner._settings.llm_model == settings.llm_model


def test_refiner_timeout_zero_is_preserved(monkeypatch):
    """0.0 is a legal (if unusual) override — must not fall through to main via 'or'."""
    monkeypatch.setenv("HARNESS_REFINER_TIMEOUT_S", "0.0")
    engine = build_refiner_engine(Settings())
    assert engine._inner._settings.llm_timeout_s == 0.0


# --- Error type ---

def test_privacy_boundary_error_is_valueerror():
    assert issubclass(PrivacyBoundaryError, ValueError)
