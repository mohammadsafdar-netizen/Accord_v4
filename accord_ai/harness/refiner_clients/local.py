"""Local (vLLM / Qwen) refiner client.

Wraps the existing accord_ai.harness.refiner.Refiner so it fits the
RefinerClient protocol.  LocalRefiner never sends data externally — it
calls the local vLLM instance that the rest of the stack uses.

RefinerOutputError is caught and converted to None (cascade falls
through to the next client, though there is none past this point by
default — so the cascade returns the original submission).

Other engine exceptions (RateLimitError, APITimeoutError, etc.) also
convert to None rather than propagating, keeping the cascade clean.
"""
from __future__ import annotations

from typing import Optional

from accord_ai.harness.judge import JudgeVerdict
from accord_ai.harness.refiner import Refiner, RefinerOutputError
from accord_ai.logging_config import get_logger
from accord_ai.schema import CustomerSubmission

_logger = get_logger("refiner_clients.local")


class LocalRefiner:
    provider: str = "local"

    def __init__(self, refiner: Refiner) -> None:
        self._refiner = refiner

    async def refine(
        self,
        *,
        original_user_message: str,
        current_submission: CustomerSubmission,
        verdict: JudgeVerdict,
    ) -> Optional[CustomerSubmission]:
        try:
            return await self._refiner.refine(
                original_user_message=original_user_message,
                current_submission=current_submission,
                verdict=verdict,
            )
        except RefinerOutputError as exc:
            _logger.warning("local: RefinerOutputError — returning None: %s", exc)
            return None
        except Exception as exc:
            _logger.warning("local: engine error — returning None: %s", exc)
            return None
