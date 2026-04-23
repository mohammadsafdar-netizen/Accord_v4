"""Validation engine — runs all validators in parallel with per-validator timeouts."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import TYPE_CHECKING, List, Optional

from .types import ValidationResult, Validator

if TYPE_CHECKING:
    from accord_ai.config import Settings

_logger = logging.getLogger(__name__)


class ValidationEngine:
    def __init__(
        self,
        validators: List[Validator],
        timeout_s: float = 10.0,
    ) -> None:
        self._validators = validators
        self._timeout_s = timeout_s

    async def run_all(self, submission: object) -> List[ValidationResult]:
        if not self._validators:
            return []

        async def _run_one(v: Validator) -> ValidationResult:
            try:
                return await asyncio.wait_for(
                    v.run(submission), timeout=self._timeout_s
                )
            except asyncio.TimeoutError:
                from datetime import datetime, timezone

                _logger.warning("validator=%s timed out after %.1fs", v.name, self._timeout_s)
                return ValidationResult(
                    validator=v.name,
                    ran_at=datetime.now(tz=timezone.utc),
                    duration_ms=self._timeout_s * 1000,
                    success=False,
                    error=f"timed out after {self._timeout_s}s",
                )
            except Exception as exc:
                from datetime import datetime, timezone

                _logger.warning("validator=%s raised %s: %s", v.name, type(exc).__name__, exc)
                return ValidationResult(
                    validator=v.name,
                    ran_at=datetime.now(tz=timezone.utc),
                    duration_ms=0.0,
                    success=False,
                    error=str(exc),
                )

        return list(await asyncio.gather(*(_run_one(v) for v in self._validators)))


def build_engine(settings: "Settings") -> ValidationEngine:
    """Build the ValidationEngine from settings.

    ENABLE_EXTERNAL_VALIDATION=false (default) → empty validator list.
    When enabled, validators requiring API keys are only added if the keys
    are present in settings (graceful degradation).

    Startup log: validation_engine_ready validators=[...] disabled=[...]
    """
    enabled = os.environ.get("ENABLE_EXTERNAL_VALIDATION", "false").lower() not in (
        "0", "false", "no",
    )
    if not enabled:
        return ValidationEngine(validators=[], timeout_s=settings.validation_timeout_s)

    from .census_naics import NaicsValidator
    from .cross_field import CrossFieldValidator
    from .dns_mx import DnsMxValidator
    from .nhtsa_recalls import NhtsaRecallsValidator
    from .nhtsa_safety import NhtsaSafetyValidator
    from .nhtsa_vpic import NhtsaVpicValidator
    from .ofac import OFACValidator
    from .phone_area import PhoneAreaValidator
    from .sec_edgar import SecEdgarValidator
    from .zippopotam import ZippopotamValidator

    # Always-active validators (no key required)
    validators: List[Validator] = [
        OFACValidator(),
        NhtsaVpicValidator(),
        ZippopotamValidator(),
        NhtsaRecallsValidator(),
        NhtsaSafetyValidator(),
        NaicsValidator(),
        PhoneAreaValidator(
            csv_path=pathlib.Path(settings.area_codes_csv_path)
        ),
        DnsMxValidator(),
        SecEdgarValidator(settings.sec_edgar_user_agent),
        CrossFieldValidator(),
    ]
    disabled: List[str] = []

    # USPS — requires both consumer key and secret
    usps_key = getattr(settings, "usps_consumer_key", None)
    usps_secret = getattr(settings, "usps_consumer_secret", None)
    if usps_key and usps_secret:
        from .usps import UspsValidator
        validators.append(UspsValidator(consumer_key=usps_key, consumer_secret=usps_secret))
    else:
        disabled.append("usps (no USPS_CONSUMER_KEY/USPS_CONSUMER_SECRET)")

    # Tax1099 — requires API key
    tax_key = getattr(settings, "tax1099_api_key", None)
    if tax_key:
        from .tax1099 import Tax1099Validator
        validators.append(Tax1099Validator(api_key=tax_key))
    else:
        disabled.append("tax1099 (no TAX1099_API_KEY)")

    # FMCSA SAFER — requires web key
    fmcsa_key = getattr(settings, "fmcsa_web_key", None)
    if fmcsa_key:
        from .fmcsa import FmcsaValidator
        validators.append(FmcsaValidator(web_key=fmcsa_key))
    else:
        disabled.append("fmcsa_safer (no FMCSA_WEB_KEY)")

    # SAM.gov — requires API key
    sam_key = getattr(settings, "sam_gov_api_key", None)
    if sam_key:
        from .sam_gov import SamGovValidator
        validators.append(SamGovValidator(api_key=sam_key))
    else:
        disabled.append("sam_gov (no SAM_GOV_API_KEY)")

    active_names = [v.name for v in validators]
    _logger.info(
        "validation_engine_ready validators=%s disabled=%s",
        active_names,
        disabled,
    )

    return ValidationEngine(validators=validators, timeout_s=settings.validation_timeout_s)
