"""OpenAIEngine — OpenAI-compat HTTP adapter.

Default target: local vLLM at localhost:8000 serving Qwen/Qwen3.5-9B.
Same class talks to any OpenAI-compatible endpoint (Groq, Cerebras,
OpenRouter, Anthropic's compat layer) via llm_base_url.

Privacy: config defaults point at localhost. Raw customer data must NEVER
be routed to an external endpoint without explicit approval.

Error policy: pass-through. openai.* exceptions propagate unchanged so the
runner (Phase 7.d) can inspect RateLimitError / APITimeoutError / etc. for
retry decisions.

Testing: inject a stub client via the `client` kwarg — no network in tests.

Qwen3 hybrid-reasoning note: the model's default chat template emits a
"Thinking Process:" scaffolding before the user-facing answer. That text
leaks into extraction/judge/responder output if not suppressed. We pass
`chat_template_kwargs={"enable_thinking": False}` via `extra_body` on
every call, which the Qwen3 jinja template honors. As a safety net we
also strip any `<think>...</think>` fragment that survives — different
Qwen variants / providers emit tag-wrapped reasoning instead of inline.
"""
from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from accord_ai.config import ExtractionMode, Settings
from accord_ai.llm.engine import EngineResponse, Message
from accord_ai.logging_config import get_logger

_logger = get_logger("openai_engine")

# Strip <think>…</think> blocks (case-insensitive, multiline) that some
# reasoning models emit despite enable_thinking=False — belt-and-suspenders
# with the chat_template_kwargs override. Handles both closed tags and the
# "unterminated reasoning, then answer" failure mode where the model never
# emits </think>.
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?(?:</think>|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _sanitize_schema_for_xgrammar(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a pydantic JSON schema into a shape xgrammar can enforce.

    Pydantic emits three idioms that vLLM/xgrammar's grammar compiler
    partially ignores, letting the LLM produce schema-valid-but-pydantic-
    invalid output:

    1. ``Optional[T]`` → ``{"anyOf": [T-schema, {"type": "null"}]}``. The
       grammar is permissive with ``anyOf`` and the LLM sometimes emits
       ``null`` mid-object where pydantic expects a typed value. Flatten
       to just ``T-schema`` — the LLM is supposed to OMIT unknown fields
       per the prompt, so offering ``null`` as a valid alternative is
       counterproductive.

    2. ``Literal[...]`` wrapped in ``Optional`` → an ``anyOf`` around an
       ``enum`` plus ``{"type": "null"}``. Post-flatten the enum lands at
       the top of the field's schema node and xgrammar pins tokens to
       the enum set — no more ``entity_type: "sole_proprietorship"``.

    3. Discriminated unions are emitted as
       ``anyOf: [{"discriminator": ..., "oneOf": [...]}, {"type": "null"}]``.
       Flattening drops the null branch; the ``oneOf`` with discriminator
       stays, which xgrammar handles correctly.

    Observed failures on the 3-scenario live run (2026-04-19):
      * ``entity_type: "sole_proprietorship"``  (Literal escaped)
      * ``lob_details: "general_liability"``    (bare string for union)
    Both should stop after sanitization.

    Non-mutating — returns a deep-copied, rewritten schema.
    """
    return _strip_null_branches(copy.deepcopy(schema))


def _strip_null_branches(node: Any) -> Any:
    """Recursive walker. Flattens ``anyOf: [T, null]`` to ``T`` throughout."""
    if isinstance(node, dict):
        # Flatten Optional[T] pattern before recursing into children so
        # the replacement node (T's schema, now hoisted up one level)
        # also gets walked.
        if "anyOf" in node and isinstance(node["anyOf"], list):
            variants = node["anyOf"]
            non_null = [
                v for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            # Exactly one non-null variant + at least one null variant →
            # hoist the non-null shape up, preserving wrapper metadata.
            if len(non_null) == 1 and len(non_null) < len(variants):
                only = non_null[0]
                if not isinstance(only, dict):
                    only = {"type": only} if isinstance(only, str) else {}
                # Preserve wrapper keys that don't collide with the
                # hoisted schema (title/description/default live on the
                # Optional-wrapped schema in pydantic output).
                merged = {k: v for k, v in only.items()}
                for meta in ("title", "description", "default"):
                    if meta in node and meta not in merged:
                        merged[meta] = node[meta]
                return _strip_null_branches(merged)
            # Multi-variant (true union) or all-null → keep anyOf, walk children.
            node["anyOf"] = [_strip_null_branches(v) for v in variants]
        return {k: _strip_null_branches(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_null_branches(v) for v in node]
    return node


def _strip_thinking(text: str) -> str:
    """Remove reasoning-scaffolding artifacts from model output.

    Two signatures seen in the wild:
      * `<think>…</think>` tag form (most Qwen-family tools)
      * Plain `Thinking Process:` / `Draft N:` / `Final Polish:` prefix
        that survives even with enable_thinking=False when the model
        has been further fine-tuned to narrate.

    The plain-prefix stripper scans for a "Final" or "Answer:" marker
    and truncates everything before it. That's aggressive but safe —
    if the marker isn't present, we return the original text verbatim.
    """
    # Tag-form: global replace.
    text = _THINK_BLOCK_RE.sub("", text).strip()

    # Prefix-form: only fire if the output looks like a reasoning dump.
    if text.lower().startswith(("thinking process", "thought process", "reasoning:")):
        m = re.search(
            r"(?:\n\s*(?:Final (?:Answer|Response|Polish|Message)|Answer)\s*:?\s*\n)"
            r"(?P<answer>.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m and m.group("answer").strip():
            return m.group("answer").strip().strip('"')
    return text


class OpenAIEngine:
    """Async OpenAI-compatible engine. Non-streaming."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=(
                settings.llm_api_key.get_secret_value()
                if settings.llm_api_key is not None
                else "sk-unused"   # vLLM ignores value; SDK requires a string
            ),
            timeout=settings.llm_timeout_s,
            # RetryingEngine owns the retry budget. Disable SDK-side retries
            # so we don't end up with 2 × 4 = 8 real calls per "generate".
            max_retries=0,
        )

    async def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> EngineResponse:
        # Compose extra_body. Always include the thinking-disable toggle.
        # Schema enforcement branches on extraction_mode (Step 25 experiment):
        #   XGRAMMAR   — current behavior: guided_json via xgrammar (when schema provided)
        #   JSON_OBJECT — response_format json_object; no xgrammar constraint
        #   FREE        — no format constraint; parser strips fences + first-{}-block fallback
        extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._settings.llm_seed is not None:
            extra_body["seed"] = self._settings.llm_seed
        mode = self._settings.extraction_mode
        create_kwargs: Dict[str, Any] = {}

        if mode == ExtractionMode.XGRAMMAR:
            if json_schema is not None:
                # vLLM 0.18+: xgrammar-backed structured output. Grammar is
                # compiled once per unique schema + cached. See comment above
                # _sanitize_schema_for_xgrammar for the shelved sanitize attempt.
                extra_body["guided_json"] = json_schema
        elif mode == ExtractionMode.JSON_OBJECT:
            # OpenAI response_format json_object — model outputs valid JSON
            # without schema constraints. Pydantic validates after.
            create_kwargs["response_format"] = {"type": "json_object"}
        # FREE mode: no format constraint; pass-through

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **create_kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        text = _strip_thinking(raw_text)
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        _logger.info(
            "engine call: model=%s tokens=%d->%d latency=%.1fms "
            "mode=%s guided=%s stripped=%s",
            response.model, tokens_in, tokens_out, latency_ms,
            mode.value,
            json_schema is not None and mode == ExtractionMode.XGRAMMAR,
            len(raw_text) != len(text),
        )
        return EngineResponse(
            text=text,
            model=response.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
