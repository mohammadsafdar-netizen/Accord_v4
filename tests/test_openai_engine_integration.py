"""7.c integration — real round-trip against local vLLM.

Gated by ACCORD_LLM_INTEGRATION=1 so the default test run stays hermetic.

Run:
    ACCORD_LLM_INTEGRATION=1 python -m pytest tests/test_openai_engine_integration.py -v

Prereqs: vLLM serving Qwen/Qwen3.5-9B at LLM_BASE_URL
(default http://localhost:8000/v1).
"""
import os

import pytest

from accord_ai.config import Settings
from accord_ai.llm.engine import EngineResponse
from accord_ai.llm.openai_engine import OpenAIEngine


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ACCORD_LLM_INTEGRATION"),
        reason="set ACCORD_LLM_INTEGRATION=1 to run",
    ),
]


@pytest.mark.asyncio
async def test_real_vllm_roundtrip():
    engine = OpenAIEngine(Settings())
    r = await engine.generate(
        [{"role": "user", "content": "Reply with exactly the word OK."}],
        max_tokens=20,
    )
    assert isinstance(r, EngineResponse)
    assert isinstance(r.text, str)
    assert len(r.text) > 0
    assert r.tokens_in > 0
    assert r.tokens_out > 0
    assert r.latency_ms > 0
