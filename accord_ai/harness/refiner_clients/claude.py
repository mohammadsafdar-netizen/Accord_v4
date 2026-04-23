"""Claude refiner client.

Uses the Anthropic Messages API directly via httpx (no SDK dependency).
Model: claude-haiku-4-5-20251001
Timeout: 15 s, zero retries — cascade handles fallthrough on any failure.

PII flow identical to GeminiRefiner:
  1. redact_dict before sending
  2. parse patch from response text
  3. _reapply_patch_to_original to restore untouched PII fields
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.harness.refiner_cascade import _reapply_patch_to_original, redact_dict
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("refiner_clients.claude")

_TIMEOUT_S: float = 15.0
_DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
_ANTHROPIC_VERSION: str = "2023-06-01"
_MESSAGES_URL: str = "https://api.anthropic.com/v1/messages"
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?")

_SYSTEM_PROMPT = (
    "You are an insurance-intake extraction refiner. "
    "The submission below has PII redacted. "
    "Return ONLY a JSON object containing corrections for the failed field paths listed. "
    "Do NOT invent values — only fix what is clearly wrong based on the user message. "
    "Output ONLY the JSON object — no preamble, no markdown fences, no commentary."
)


def _build_prompt(
    original_user_message: str,
    redacted_submission: dict,
    verdict: JudgeVerdict,
) -> str:
    paths_text = "\n".join(f"- {p}" for p in verdict.failed_paths) or "(none)"
    reasons_text = "\n".join(f"- {r}" for r in verdict.reasons) or "(none)"
    return (
        f"User message:\n{original_user_message}\n\n"
        f"Current extraction (PII redacted):\n"
        f"{json.dumps(redacted_submission, indent=2, default=str)}\n\n"
        f"Failed paths:\n{paths_text}\n\n"
        f"Reasons:\n{reasons_text}\n\n"
        f"Return a JSON patch object with only the corrected fields."
    )


def _parse_json_patch(text: str) -> Optional[dict]:
    text = _FENCE_RE.sub("", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    return data


class ClaudeRefiner:
    provider: str = "claude"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def refine(
        self,
        *,
        original_user_message: str,
        current_submission: CustomerSubmission,
        verdict: JudgeVerdict,
    ) -> Optional[CustomerSubmission]:
        redacted = redact_dict(current_submission.model_dump())
        prompt = _build_prompt(original_user_message, redacted, verdict)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": 2048,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(_MESSAGES_URL, json=payload, headers=headers)
                response.raise_for_status()
        except Exception as exc:
            _logger.warning("claude: HTTP error (%s) — falling through: %s", type(exc).__name__, exc)
            return None

        try:
            text = response.json()["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            _logger.warning("claude: unexpected response shape: %s", exc)
            return None

        patch = _parse_json_patch(text)
        if not patch:
            _logger.debug("claude: empty or unparseable patch — falling through")
            return None

        try:
            return _reapply_patch_to_original(current_submission, patch)
        except Exception as exc:
            _logger.warning("claude: patch reapply failed: %s", exc)
            return None
