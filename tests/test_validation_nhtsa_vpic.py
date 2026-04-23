"""Tests for NhtsaVpicValidator (Phase 1.6.B)."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from accord_ai.schema import (
    CommercialAutoDetails,
    CustomerSubmission,
    FieldConflict,
    Vehicle,
)
from accord_ai.validation.nhtsa_vpic import NhtsaVpicValidator, _build_vehicle_patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_with_vehicle(**veh_kwargs) -> CustomerSubmission:
    return CustomerSubmission(
        lob_details=CommercialAutoDetails(
            vehicles=[Vehicle(**veh_kwargs)]
        )
    )


_DECODED_F150 = {
    "Model Year": "2024",
    "Make": "FORD",
    "Model": "F-150",
    "Body Class": "Crew Cab Pickup",
    "Gross Vehicle Weight Rating From": "Class 3: 10001-14000 lb",
    "Error Code": "0",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vpic_fills_missing_engine_from_vin():
    """prefill() fills body_type when user hasn't provided it."""
    existing = {"year": 2024, "make": "Ford", "model": "F-150", "vin": "1FTFW1E83RFB12345"}
    idx_patch, conflicts = _build_vehicle_patch(_DECODED_F150, existing, 0)
    assert idx_patch[0].get("body_type") == "Crew Cab Pickup"
    assert conflicts == []


def test_vpic_year_mismatch_creates_conflict():
    """When user said 2023 but VIN decodes 2024, emit a Conflict (no silent overwrite)."""
    existing = {"year": 2023, "make": "Ford", "model": "F-150", "vin": "1FTFW1E83RFB12345"}
    _, conflicts = _build_vehicle_patch(_DECODED_F150, existing, 0)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.field_path == "lob_details.vehicles[0].year"
    assert c.user_value == 2023
    assert c.enriched_value == 2024
    assert c.source == "nhtsa_vpic"


@pytest.mark.asyncio
async def test_vpic_invalid_vin_returns_no_patch():
    """When NHTSA returns None (bad VIN), prefill returns None."""
    with patch(
        "accord_ai.validation.nhtsa_vpic._decode_vin",
        AsyncMock(return_value=None),
    ):
        sub = _sub_with_vehicle(vin="INVALIDVIN1234567")
        just = {"lob_details": {"vehicles": [{"vin": "INVALIDVIN1234567"}]}}
        patch_result = await NhtsaVpicValidator().prefill(sub, just)
    assert patch_result is None


@pytest.mark.asyncio
async def test_vpic_multiple_vehicles_enriched():
    """prefill() enriches all newly-extracted VINs in parallel."""
    with patch(
        "accord_ai.validation.nhtsa_vpic._decode_vin",
        AsyncMock(return_value=_DECODED_F150),
    ):
        sub = CustomerSubmission(
            lob_details=CommercialAutoDetails(vehicles=[
                Vehicle(vin="VIN1", year=2024),
                Vehicle(vin="VIN2"),
            ])
        )
        just = {"lob_details": {"vehicles": [{"vin": "VIN1"}, {"vin": "VIN2"}]}}
        patch_result = await NhtsaVpicValidator().prefill(sub, just)
    assert patch_result is not None
    vehicle_patches = patch_result.patch["lob_details"]["vehicles"]
    # Both vehicles received patches
    assert len(vehicle_patches) >= 1


@pytest.mark.asyncio
async def test_vpic_http_timeout_silent_failure():
    """HTTP timeout → _decode_vin returns None → prefill returns None gracefully."""
    import asyncio

    async def _timeout(*_a, **_k):
        raise asyncio.TimeoutError()

    with patch("accord_ai.validation.nhtsa_vpic._decode_vin", side_effect=_timeout):
        sub = _sub_with_vehicle(vin="1FTFW1E83RFB12345")
        just = {"lob_details": {"vehicles": [{"vin": "1FTFW1E83RFB12345"}]}}
        patch_result = await NhtsaVpicValidator().prefill(sub, just)
    assert patch_result is None
