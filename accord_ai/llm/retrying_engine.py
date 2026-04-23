"""RetryingEngine — wraps any Engine with exponential-backoff retry.

Retries on transient openai.* exceptions:
  - APITimeoutError
  - APIConnectionError
  - RateLimitError
  - InternalServerError
  - Any APIStatusError with status_code >= 500 (catches future 5xx types)

All other exceptions (AuthenticationError, BadRequestError, etc.) propagate
unchanged — they're permanent errors; retrying just wastes time and quota.

Backoff (equal-jitter exponential):
    sleep = min(base * 2**attempt, cap) + U(0, base)

Implements the Engine protocol so it stacks transparently.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple, Type

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from accord_ai.config import Settings
from accord_ai.llm.engine import Engine, EngineResponse, Message
from accord_ai.logging_config import get_logger

_logger = get_logger("retrying_engine")

_RETRYABLE_TYPES: Tuple[Type[Exception], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


def _is_retryable(exc: Exception) -> bool:
    """True if `exc` should trigger a retry."""
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


class RetryingEngine:
    """Engine wrapper. Same protocol as the inner engine; adds retry."""

    def __init__(self, inner: Engine, settings: Settings) -> None:
        self._inner = inner
        self._max_retries = settings.llm_retries
        self._base_s = settings.llm_retry_base_s
        self._cap_s = settings.llm_retry_cap_s

    async def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> EngineResponse:
        # NOTE: default mirrors OpenAIEngine — we were previously hardcoded at
        # 4096 here, which clobbered the inner engine's 1024 default and blew
        # past `max_model_len=8192` when the extractor schema prompt ran
        # (schema ~4100 in tokens + 4096 out = 8196, rejected by vLLM).
        for attempt in range(self._max_retries + 1):
            try:
                return await self._inner.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_schema=json_schema,
                )
            except Exception as e:
                if not _is_retryable(e) or attempt >= self._max_retries:
                    raise
                sleep_s = self._backoff(attempt)
                _logger.warning(
                    "engine retry attempt=%d/%d sleep=%.2fs error=%s",
                    attempt + 1, self._max_retries, sleep_s, type(e).__name__,
                )
                await asyncio.sleep(sleep_s)
        # Unreachable — loop body always returns or raises
        raise RuntimeError("unreachable retry-loop fallthrough")

    def _backoff(self, attempt: int) -> float:
        return min(self._base_s * (2 ** attempt), self._cap_s) + random.uniform(0, self._base_s)
