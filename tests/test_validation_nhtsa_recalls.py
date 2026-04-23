"""Tests for NhtsaRecallsValidator + NhtsaSafetyValidator (Phase 1.6.B)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from accord_ai.schema import CommercialAutoDetails, CustomerSubmission, Vehicle
from accord_ai.validation.nhtsa_recalls import NhtsaRecallsValidator
from accord_ai.validation.nhtsa_safety import NhtsaSafetyValidator


_RECALL_RESPONSE = [
    {
        "Subject": "Engine may overheat",
        "NHTSACampaignNumber": "24V-001",
        "Component": "ENGINE",
    }
]

_SAFETY_RESPONSE = [
    {"OverallRating": "5", "VehicleId": "12345"},
]


def _sub_with_vehicle(year=2024, make="Ford", model="F-150") -> CustomerSubmission:
    return CustomerSubmission(
        lob_details=CommercialAutoDetails(
            vehicles=[Vehicle(year=year, make=make, model=model, vin="VIN123")]
        )
    )


@pytest.mark.asyncio
async def test_recalls_finds_open_recall():
    with patch(
        "accord_ai.validation.nhtsa_recalls._fetch_recalls",
        AsyncMock(return_value=_RECALL_RESPONSE),
    ):
        result = await NhtsaRecallsValidator().run(_sub_with_vehicle())

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "Engine may overheat" in f.message
    assert f.details["campaign_number"] == "24V-001"


def test_recalls_inline_eligible_false():
    assert NhtsaRecallsValidator.inline_eligible is False
    assert NhtsaSafetyValidator.inline_eligible is False


@pytest.mark.asyncio
async def test_safety_ratings_found():
    with patch(
        "accord_ai.validation.nhtsa_safety._fetch_safety",
        AsyncMock(return_value=_SAFETY_RESPONSE),
    ):
        result = await NhtsaSafetyValidator().run(_sub_with_vehicle())

    assert result.success is True
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "info"
    assert "5 stars" in f.message
