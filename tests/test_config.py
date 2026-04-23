import pytest
from pydantic import ValidationError

from accord_ai.config import Settings


# --- API / storage defaults ---

def test_defaults():
    s = Settings()
    assert s.api_port == 1505
    assert s.db_path == "accord_ai.db"


def test_env_override(monkeypatch):
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("DB_PATH", "/tmp/x.db")
    s = Settings()
    assert s.api_port == 9999
    assert s.db_path == "/tmp/x.db"


def test_port_coerces_from_string(monkeypatch):
    monkeypatch.setenv("API_PORT", "8080")
    s = Settings()
    assert s.api_port == 8080
    assert isinstance(s.api_port, int)


def test_invalid_port_rejected(monkeypatch):
    monkeypatch.setenv("API_PORT", "not-a-number")
    with pytest.raises(Exception):
        Settings()


# --- Logging defaults ---

def test_log_defaults():
    s = Settings()
    assert s.log_dir == "logs"
    assert s.log_level == "INFO"
    assert s.log_format == "text"
    assert s.pii_redaction is True


def test_log_level_env_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.log_level == "DEBUG"


def test_log_dir_env_override(monkeypatch):
    monkeypatch.setenv("LOG_DIR", "/var/log/accord")
    s = Settings()
    assert s.log_dir == "/var/log/accord"


def test_pii_redaction_bool_from_env(monkeypatch):
    monkeypatch.setenv("PII_REDACTION", "false")
    s = Settings()
    assert s.pii_redaction is False


# --- Literal validation ---

def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "FOOBAR")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_log_format_rejected(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValidationError):
        Settings()


# --- LLM settings ---

def test_llm_defaults_target_local_vllm():
    s = Settings()
    assert s.llm_base_url == "http://localhost:8000/v1"
    assert s.llm_model == "insurance-agent"
    assert s.llm_api_key is None
    assert s.llm_timeout_s == 30.0


def test_llm_base_url_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://other:9000/v1")
    assert Settings().llm_base_url == "http://other:9000/v1"


def test_llm_model_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some/other-model")
    assert Settings().llm_model == "some/other-model"


def test_llm_api_key_is_secret(monkeypatch):
    """Secrets must not leak via repr / str."""
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-xyz")
    s = Settings()
    assert s.llm_api_key is not None
    assert "sk-secret-xyz" not in repr(s.llm_api_key)
    assert s.llm_api_key.get_secret_value() == "sk-secret-xyz"


# --- LLM retry settings ---

def test_llm_retry_defaults():
    s = Settings()
    assert s.llm_retries == 3
    assert s.llm_retry_base_s == 0.5
    assert s.llm_retry_cap_s == 8.0


def test_llm_retries_env_override(monkeypatch):
    monkeypatch.setenv("LLM_RETRIES", "5")
    assert Settings().llm_retries == 5


def test_llm_retry_base_and_cap_env(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_S", "0.1")
    monkeypatch.setenv("LLM_RETRY_CAP_S", "2.0")
    s = Settings()
    assert s.llm_retry_base_s == 0.1
    assert s.llm_retry_cap_s == 2.0


# --- Harness settings (5.d) ---

def test_harness_defaults():
    s = Settings()
    assert s.harness_max_refines == 1
    assert s.harness_refiner_base_url is None
    assert s.harness_refiner_model is None
    assert s.harness_refiner_api_key is None
    assert s.harness_refiner_timeout_s is None


def test_harness_max_refines_env_override(monkeypatch):
    monkeypatch.setenv("HARNESS_MAX_REFINES", "3")
    assert Settings().harness_max_refines == 3


def test_harness_max_refines_rejects_negative(monkeypatch):
    monkeypatch.setenv("HARNESS_MAX_REFINES", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_harness_refiner_base_url_env_override(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_BASE_URL", "https://api.anthropic.com/v1")
    assert Settings().harness_refiner_base_url == "https://api.anthropic.com/v1"


def test_harness_refiner_api_key_is_secret(monkeypatch):
    monkeypatch.setenv("HARNESS_REFINER_API_KEY", "sk-refiner-xyz")
    s = Settings()
    assert s.harness_refiner_api_key is not None
    assert "sk-refiner-xyz" not in repr(s.harness_refiner_api_key)
    assert s.harness_refiner_api_key.get_secret_value() == "sk-refiner-xyz"


# --- API auth + CORS settings (11.d) ---

def test_api_auth_defaults():
    s = Settings()
    assert s.intake_api_key is None
    assert s.accord_auth_disabled is False
    assert s.accord_chat_open is False
    assert s.allowed_origins == "*"
    assert s.allowed_origin_regex is None


def test_intake_api_key_is_secret(monkeypatch):
    monkeypatch.setenv("INTAKE_API_KEY", "sk-intake-xyz")
    s = Settings()
    assert s.intake_api_key is not None
    assert "sk-intake-xyz" not in repr(s.intake_api_key)
    assert s.intake_api_key.get_secret_value() == "sk-intake-xyz"


def test_accord_auth_disabled_from_env(monkeypatch):
    monkeypatch.setenv("ACCORD_AUTH_DISABLED", "true")
    assert Settings().accord_auth_disabled is True


def test_accord_chat_open_from_env(monkeypatch):
    monkeypatch.setenv("ACCORD_CHAT_OPEN", "true")
    assert Settings().accord_chat_open is True


# --- INTAKE_API_KEYS (key→tenant binding map, P10.0.a) ---

def test_intake_api_keys_defaults_to_empty_dict():
    s = Settings()
    assert s.intake_api_keys == {}


def test_intake_api_keys_parses_json_from_env(monkeypatch):
    monkeypatch.setenv(
        "INTAKE_API_KEYS",
        '{"sk-acme-xyz": "acme", "sk-globex-abc": "globex"}',
    )
    s = Settings()
    assert s.intake_api_keys == {
        "sk-acme-xyz": "acme",
        "sk-globex-abc": "globex",
    }


def test_intake_api_keys_malformed_json_raises_at_construction(monkeypatch):
    monkeypatch.setenv("INTAKE_API_KEYS", "not-json")
    with pytest.raises(Exception):
        Settings()


def test_intake_api_keys_empty_json_object_is_valid(monkeypatch):
    monkeypatch.setenv("INTAKE_API_KEYS", "{}")
    s = Settings()
    assert s.intake_api_keys == {}
