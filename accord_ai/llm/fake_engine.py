"""In-memory FakeEngine — FIFO-queued responses for tests.

Usage:
    engine = FakeEngine(["plain text", {"business_name": "Acme"}])
    r1 = await engine.generate([{"role": "user", "content": "q1"}])
    r2 = await engine.generate([{"role": "user", "content": "q2"}])
    # r1.text == "plain text"
    # r2.text == '{"business_name": "Acme"}'

Canonicalization: dict responses are json.dumps'd at enqueue time so the
engine contract stays honest (EngineResponse.text is always str, same as a
real LLM). Test authors get the ergonomics; extraction code gets the same
parse path as production.

Call history is recorded on `engine.calls` for assertions.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Union

from accord_ai.llm.engine import EngineResponse, Message


@dataclass(frozen=True)
class FakeCall:
    """One recorded ``generate()`` call — surface enough for assertions
    about what the production code shipped to the engine, including the
    guided-JSON schema and other kwargs."""
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: int = 4096
    json_schema: Optional[Dict[str, Any]] = None


class FakeEngine:
    """Deterministic Engine for tests. Pops canned responses in FIFO order."""

    def __init__(
        self,
        responses: Optional[List[Union[str, dict]]] = None,
        *,
        model: str = "fake",
    ) -> None:
        self._queue: Deque[str] = deque()
        self._model = model
        # Call log for assertions — list of FakeCall records. `calls`
        # stays backward-compatible: indexing ``engine.calls[i]`` returns
        # the messages list as before (FakeCall's first positional).
        self.history: List[FakeCall] = []
        if responses:
            self.extend(responses)

    # Back-compat: tests reach for `engine.calls[i]` expecting the raw
    # messages list. Route it via a property that maps to history so both
    # `engine.calls` (list of message lists) and `engine.history` (list
    # of FakeCall) are available without copy-paste across tests.
    @property
    def calls(self) -> List[List[Message]]:
        return [c.messages for c in self.history]

    def extend(self, responses: List[Union[str, dict]]) -> None:
        """Append more canned responses to the queue."""
        for r in responses:
            self._queue.append(json.dumps(r) if isinstance(r, dict) else r)

    @property
    def last_messages(self) -> List[Message]:
        """Messages list from the most recent generate() call.

        Raises RuntimeError if generate() has never been called — refusing
        to return `None` so assertions don't silently pass.
        """
        if not self.history:
            raise RuntimeError("no generate() calls yet")
        return self.history[-1].messages

    @property
    def last_call(self) -> FakeCall:
        """Full kwargs of the most recent call — includes json_schema."""
        if not self.history:
            raise RuntimeError("no generate() calls yet")
        return self.history[-1]

    async def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> EngineResponse:
        if not self._queue:
            raise RuntimeError(
                "FakeEngine queue exhausted — add responses before generate()"
            )
        self.history.append(FakeCall(
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        ))
        text = self._queue.popleft()
        return EngineResponse(
            text=text,
            model=self._model,
            # Token approximation (~4 chars/token) — good enough for tests
            tokens_in=sum(len(m["content"]) // 4 for m in messages),
            tokens_out=len(text) // 4,
            latency_ms=0.0,
        )
