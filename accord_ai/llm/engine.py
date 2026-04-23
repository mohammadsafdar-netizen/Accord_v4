"""LLM Engine protocol — abstraction over provider adapters.

Any concrete engine (FakeEngine for tests, OpenAIEngine for production)
implements the same async interface. Callers (extraction runner, harness
judge/refiner, responder) depend only on this protocol — they don't know
which provider is behind it.

EngineResponse carries observability fields (model, tokens, latency) so
metering and cache-hit analysis work without provider-specific code paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict


MessageRole = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """Chat message shape — matches OpenAI wire format exactly."""
    role: MessageRole
    content: str


@dataclass(frozen=True)
class EngineResponse:
    """Result of a single non-streaming generation."""
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class Engine(Protocol):
    """Async non-streaming LLM interface. Non-streaming by design —
    extraction needs the whole response, and most callers do too.

    ``json_schema`` (optional) is a pydantic-style JSON schema dict the
    engine passes to the provider for structured-output enforcement
    (vLLM: ``extra_body={"guided_json": schema}`` via xgrammar). When
    set, the model is token-constrained to emit JSON conforming to that
    schema — no malformed output, no unescaped quotes, no cut-off
    structures. Callers that need free-form text (the responder) omit
    the kwarg. Providers that don't support structured output simply
    ignore it.
    """

    async def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> EngineResponse:
        ...
