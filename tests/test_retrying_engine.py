"""7.d — RetryingEngine unit tests. Sleep + jitter are monkeypatched for speed + determinism."""
import asyncio
from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from accord_ai.config import Settings
from accord_ai.llm.engine import Engine, EngineResponse
from accord_ai.llm.retrying_engine import RetryingEngine, _is_retryable


# --- Helpers ---

def _response(text="ok"):
    return EngineResponse(text=text, model="m", tokens_in=1, tokens_out=1, latency_ms=0.0)


class _FlakyEngine:
    """Fails the first N generate() calls, then returns a canned response."""

    def __init__(self, fail_times=0, error=None, response=None):
        self.calls = 0
        self.fail_times = fail_times
        self.error = error
        self.response = response or _response()

    async def generate(self, messages, *, temperature=0.0, max_tokens=4096, json_schema=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.response


def _resp(status_code):
    r = MagicMock()
    r.status_code = status_code
    return r


def _timeout():       return APITimeoutError(request=MagicMock())
def _connection():    return APIConnectionError(request=MagicMock())
def _rate_limit():    return RateLimitError("rate limited", response=_resp(429), body=None)
def _internal():      return InternalServerError("server error", response=_resp(500), body=None)
def _auth():          return AuthenticationError("unauthorized", response=_resp(401), body=None)
def _bad_request():   return BadRequestError("bad request", response=_resp(400), body=None)


# --- Fixtures ---

@pytest.fixture
def no_sleep(monkeypatch):
    """Capture asyncio.sleep durations without actually sleeping."""
    sleeps = []

    async def fake(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake)
    return sleeps


@pytest.fixture
def no_jitter(monkeypatch):
    """Pin jitter to 0 for deterministic backoff assertions."""
    monkeypatch.setattr(
        "accord_ai.llm.retrying_engine.random.uniform", lambda a, b: 0.0
    )


@pytest.fixture
def settings():
    return Settings()


# --- _is_retryable ---

@pytest.mark.parametrize("factory, expected", [
    (_timeout,      True),
    (_connection,   True),
    (_rate_limit,   True),
    (_internal,     True),
    (_auth,         False),
    (_bad_request,  False),
    (lambda: ValueError("generic"), False),
    (lambda: RuntimeError("boom"),  False),
])
def test_is_retryable(factory, expected):
    assert _is_retryable(factory()) is expected


def test_is_retryable_catches_any_5xx_api_status_error():
    class _Generic5xx(APIStatusError):
        pass
    err = _Generic5xx("svc unavailable", response=_resp(503), body=None)
    assert _is_retryable(err) is True


def test_is_retryable_rejects_4xx_api_status_error():
    class _Generic4xx(APIStatusError):
        pass
    err = _Generic4xx("conflict", response=_resp(409), body=None)
    assert _is_retryable(err) is False


# --- Happy path ---

@pytest.mark.asyncio
async def test_success_first_try_no_retries(settings, no_sleep):
    inner = _FlakyEngine(fail_times=0)
    r = await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert r.text == "ok"
    assert inner.calls == 1
    assert no_sleep == []


# --- Retry behavior ---

@pytest.mark.asyncio
async def test_retries_on_timeout_then_succeeds(settings, no_sleep, no_jitter):
    inner = _FlakyEngine(fail_times=2, error=_timeout())
    r = await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert r.text == "ok"
    assert inner.calls == 3       # 2 failures + 1 success
    assert len(no_sleep) == 2


@pytest.mark.asyncio
async def test_retries_on_rate_limit(settings, no_sleep, no_jitter):
    inner = _FlakyEngine(fail_times=1, error=_rate_limit())
    await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert inner.calls == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_and_reraises(settings, no_sleep, no_jitter):
    """With llm_retries=3, we see 4 total attempts; the last error propagates."""
    inner = _FlakyEngine(fail_times=100, error=_timeout())
    with pytest.raises(APITimeoutError):
        await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert inner.calls == settings.llm_retries + 1
    assert len(no_sleep) == settings.llm_retries


# --- Non-retryable ---

@pytest.mark.asyncio
async def test_auth_error_does_not_retry(settings, no_sleep):
    inner = _FlakyEngine(fail_times=1, error=_auth())
    with pytest.raises(AuthenticationError):
        await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert inner.calls == 1
    assert no_sleep == []


@pytest.mark.asyncio
async def test_bad_request_does_not_retry(settings, no_sleep):
    inner = _FlakyEngine(fail_times=1, error=_bad_request())
    with pytest.raises(BadRequestError):
        await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert inner.calls == 1


# --- Backoff math ---

@pytest.mark.asyncio
async def test_backoff_is_exponential(settings, no_sleep, no_jitter):
    """Attempts 0,1,2 → sleeps 0.5, 1.0, 2.0 (no jitter, no cap hit)."""
    inner = _FlakyEngine(fail_times=100, error=_timeout())
    with pytest.raises(APITimeoutError):
        await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    assert no_sleep == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_backoff_respects_cap(no_sleep, no_jitter, monkeypatch):
    """cap=1.0 forces attempts 1+ to clamp at 1.0."""
    monkeypatch.setenv("LLM_RETRY_CAP_S", "1.0")
    settings = Settings()
    inner = _FlakyEngine(fail_times=100, error=_timeout())
    with pytest.raises(APITimeoutError):
        await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    # attempt 0: min(0.5, 1.0)=0.5 ; attempt 1: min(1.0, 1.0)=1.0 ; attempt 2: min(2.0, 1.0)=1.0
    assert no_sleep == [0.5, 1.0, 1.0]


# --- Jitter bounded ---

@pytest.mark.asyncio
async def test_jitter_keeps_sleep_in_expected_band(settings, no_sleep):
    """No jitter-pin fixture — verify sleep is within [base*2^n, base*2^n + base]."""
    inner = _FlakyEngine(fail_times=1, error=_timeout())
    await RetryingEngine(inner, settings).generate([{"role": "user", "content": "q"}])
    [sleep] = no_sleep
    assert 0.5 <= sleep <= 1.0   # attempt 0: 0.5 + U(0, 0.5)


# --- Protocol conformance ---

@pytest.mark.asyncio
async def test_conforms_to_engine_protocol(settings):
    wrapped: Engine = RetryingEngine(_FlakyEngine(), settings)
    r = await wrapped.generate([{"role": "user", "content": "q"}])
    assert isinstance(r, EngineResponse)
