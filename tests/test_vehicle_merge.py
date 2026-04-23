"""Tests for accord_ai.core.vehicle_merge — 3-tier vehicle merge + driver merge.

8 vehicle cases + 3 driver cases = 11 total.
"""
from __future__ import annotations

from datetime import date

import pytest

from accord_ai.core.vehicle_merge import merge_drivers, merge_vehicles
from accord_ai.schema import Driver, Vehicle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v(**kwargs) -> Vehicle:
    return Vehicle(**kwargs)


def _d(**kwargs) -> Driver:
    return Driver(**kwargs)


# ---------------------------------------------------------------------------
# Vehicle merge tests
# ---------------------------------------------------------------------------

class TestMergeVehiclesTier1:
    def test_same_vin_different_year_incoming_wins(self):
        """Tier 1: broker corrects the year — VIN is authoritative."""
        existing = [_v(vin="1HGBH41JXMN109186", year=2022, make="Honda", model="Civic")]
        incoming = [_v(vin="1HGBH41JXMN109186", year=2023, make="Honda", model="Civic")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 1
        assert result[0].year == 2023   # incoming year wins

    def test_same_vin_no_duplicate(self):
        """Tier 1: identical vehicles are not duplicated."""
        v = _v(vin="2T1BURHE0JC054321", year=2020, make="Toyota", model="Corolla")
        result = merge_vehicles([v], [v])
        assert len(result) == 1


class TestMergeVehiclesTier2:
    def test_matching_tuple_no_vin_no_duplicate(self):
        """Tier 2: both lack VIN + same (year, make, model) → merge, not append."""
        existing = [_v(year=2021, make="Honda", model="Civic")]
        incoming = [_v(year=2021, make="Honda", model="Civic", body_type="sedan")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 1
        assert result[0].body_type == "sedan"   # incoming field merged in

    def test_different_tuples_both_no_vin_both_kept(self):
        """Tier 2 does not match when tuples differ — both vehicles kept."""
        existing = [_v(year=2021, make="Honda", model="Civic")]
        incoming = [_v(year=2020, make="Toyota", model="Corolla")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 2


class TestMergeVehiclesTier3:
    def test_incoming_has_vin_existing_no_vin_ymm_match_populates_vin(self):
        """Tier 3: incoming adds a VIN to a previously VIN-less vehicle."""
        existing = [_v(year=2021, make="Honda", model="Civic")]
        incoming = [_v(vin="1HGBH41JXMN109186", year=2021, make="Honda", model="Civic")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 1
        assert result[0].vin == "1HGBH41JXMN109186"


class TestMergeVehiclesEdgeCases:
    def test_no_match_different_vin_different_tuple_appended(self):
        """Unrelated vehicles are appended — not merged."""
        existing = [_v(vin="1HGBH41JXMN109186", year=2021, make="Honda", model="Civic")]
        incoming = [_v(vin="2T1BURHE0JC054321", year=2020, make="Toyota", model="Corolla")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 2

    def test_empty_existing_n_incoming_returns_n(self):
        """Starting from empty, all incoming vehicles are added."""
        incoming = [
            _v(vin="1HGBH41JXMN109186", year=2021, make="Honda", model="Civic"),
            _v(vin="2T1BURHE0JC054321", year=2020, make="Toyota", model="Corolla"),
        ]
        result = merge_vehicles([], incoming)
        assert len(result) == 2

    def test_incoming_empty_existing_unchanged(self):
        """No incoming vehicles → existing list returned unchanged."""
        existing = [_v(vin="1HGBH41JXMN109186", year=2021, make="Honda", model="Civic")]
        result = merge_vehicles(existing, [])
        assert result == existing

    def test_different_vins_same_ymm_not_merged(self):
        """Two vehicles with different VINs but same y+m+m are different — not merged."""
        existing = [_v(vin="1HGBH41JXMN109186", year=2021, make="Honda", model="Civic")]
        incoming = [_v(vin="2HGBH41JXMN109999", year=2021, make="Honda", model="Civic")]
        result = merge_vehicles(existing, incoming)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Driver merge tests
# ---------------------------------------------------------------------------

class TestMergeDrivers:
    def test_same_license_number_merges(self):
        """License-primary: same license → update, not append."""
        existing = [_d(first_name="John", last_name="Doe", license_number="D1234567")]
        incoming = [_d(first_name="John", last_name="Doe",
                       license_number="D1234567", years_experience=5)]
        result = merge_drivers(existing, incoming)
        assert len(result) == 1
        assert result[0].years_experience == 5

    def test_different_license_numbers_both_kept(self):
        """Different license numbers → two distinct drivers."""
        existing = [_d(first_name="John", last_name="Doe", license_number="D1234567")]
        incoming = [_d(first_name="Jane", last_name="Smith", license_number="S7654321")]
        result = merge_drivers(existing, incoming)
        assert len(result) == 2

    def test_missing_license_number_incoming_appended(self):
        """When incoming has no license number, it falls back to name+DOB.
        If that also doesn't match, the driver is appended."""
        existing = [_d(first_name="John", last_name="Doe", license_number="D1234567")]
        incoming = [_d(first_name="Alice", last_name="Brown")]  # no license, different name
        result = merge_drivers(existing, incoming)
        assert len(result) == 2
