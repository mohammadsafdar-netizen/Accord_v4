"""Tests for accord_ai.extraction.fleet_ingest.

Unit tests (9): header detection, column classification, value coercion,
  total-row rejection, MVR normalization.
Integration tests (5): CSV/XLSX parsing end-to-end, LOB auto-set.
"""
from __future__ import annotations

import io
from datetime import date
from decimal import Decimal

import pytest

from accord_ai.extraction.fleet_ingest import (
    FleetIngestResult,
    _clean_value,
    _classify_one_column,
    _find_header,
    merge_fleet_into_submission,
    parse_fleet_sheet,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFindHeader:
    def test_detects_keyword_row(self):
        rows = [
            ("2022", "Toyota", "Camry", "1HGBH41JXMN109186"),     # data
            ("VIN", "Make", "Model", "Year"),                       # header
            ("1HGBH41JXMN109186", "Honda", "Civic", 2021),
        ]
        assert _find_header(rows) == 1

    def test_skips_leading_numeric_rows(self):
        rows = [
            (1, 2, 3, 4),                        # all numeric — score penalised
            ("VIN Number", "Make", "Year", "Model"),
        ]
        assert _find_header(rows) == 1

    def test_returns_minus_one_when_no_keywords_match(self):
        rows = [
            ("Alpha", "Beta", "Gamma"),
            ("foo", "bar", "baz"),
        ]
        assert _find_header(rows) == -1


class TestClassifyColumns:
    def test_keyword_match_vin(self):
        assert _classify_one_column("VIN Number", []) == "vin"

    def test_keyword_match_date_of_birth(self):
        assert _classify_one_column("Date of Birth", []) == "date_of_birth"

    def test_keyword_match_gvw_from_v3_alias(self):
        assert _classify_one_column("GVWR", []) == "gvw"

    def test_keyword_match_stated_amount(self):
        assert _classify_one_column("Stated Value", []) == "stated_amount"

    def test_value_shape_fallback_vin(self):
        samples = ["1HGBH41JXMN109186", "2T1BURHE0JC054321", "3VWFE21C04M000001"]
        # 17-char uppercase alphanumeric → vin
        assert _classify_one_column("", samples) == "vin"


class TestCleanValue:
    def test_date_of_birth_string_returns_date_object(self):
        result = _clean_value("date_of_birth", "03/15/1990")
        assert result == date(1990, 3, 15)

    def test_stated_amount_strips_currency_symbols(self):
        result = _clean_value("stated_amount", "$45,000.00")
        assert result == Decimal("45000.00")

    def test_mvr_status_dui_normalises_to_major_violations(self):
        assert _clean_value("mvr_status", "DUI") == "major_violations"

    def test_mvr_status_clean_normalises(self):
        assert _clean_value("mvr_status", "Clean") == "clean"

    def test_mvr_status_suspended(self):
        assert _clean_value("mvr_status", "suspended") == "suspended"

    def test_use_type_commercial(self):
        assert _clean_value("use_type", "Commercial") == "commercial"

    def test_radius_of_travel_strips_units(self):
        assert _clean_value("radius_of_travel", "250 miles") == 250


class TestTotalRowRejection:
    def _make_csv(self, rows: list[str]) -> bytes:
        return "\n".join(rows).encode()

    def test_total_row_skipped(self):
        content = self._make_csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
            "Total,,,",
        ])
        result = parse_fleet_sheet(content, "fleet.csv")
        assert len(result.vehicles) == 1, "Total row must not produce a vehicle"

    def test_grand_total_row_skipped(self):
        content = self._make_csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
            "Grand Total,,,",
        ])
        result = parse_fleet_sheet(content, "fleet.csv")
        assert len(result.vehicles) == 1

    def test_normal_data_row_not_skipped(self):
        content = self._make_csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
            "2T1BURHE0JC054321,Toyota,Corolla,2020",
        ])
        result = parse_fleet_sheet(content, "fleet.csv")
        assert len(result.vehicles) == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def _csv(rows: list[str]) -> bytes:
    return "\n".join(rows).encode()


class TestParseCsvVehicles:
    def test_basic_vehicle_roster(self):
        content = _csv([
            "VIN,Make,Model,Year,Stated Amount,GVW,Radius of Travel",
            "1HGBH41JXMN109186,Honda,Civic,2021,25000,6000,150",
            "2T1BURHE0JC054321,Toyota,Corolla,2020,22000,5500,200",
        ])
        result = parse_fleet_sheet(content, "vehicles.csv")
        assert len(result.vehicles) == 2
        v = result.vehicles[0]
        assert v.vin == "1HGBH41JXMN109186"
        assert v.make == "Honda"
        assert v.year == 2021
        assert v.stated_amount == Decimal("25000")
        assert v.gvw == 6000
        assert v.radius_of_travel == 150
        assert result.header_row_idx == 0
        assert result.warnings == []

    def test_garaging_fields_become_garage_address(self):
        content = _csv([
            "VIN,Make,Model,Year,Garaging Zip,Garaging State,Garaging City",
            "1HGBH41JXMN109186,Honda,Civic,2021,90210,CA,Beverly Hills",
        ])
        result = parse_fleet_sheet(content, "vehicles.csv")
        assert len(result.vehicles) == 1
        addr = result.vehicles[0].garage_address
        assert addr is not None
        assert addr.zip_code == "90210"
        assert addr.state == "CA"
        assert addr.city == "Beverly Hills"


class TestParseCsvDrivers:
    def test_basic_driver_roster(self):
        content = _csv([
            "First Name,Last Name,Date of Birth,License Number,License State,MVR Status",
            "John,Doe,03/15/1985,D1234567,CA,Clean",
            "Jane,Smith,07/22/1990,S7654321,TX,Minor",
        ])
        result = parse_fleet_sheet(content, "drivers.csv")
        assert len(result.drivers) == 2
        d = result.drivers[0]
        assert d.first_name == "John"
        assert d.last_name == "Doe"
        assert d.date_of_birth == date(1985, 3, 15)
        assert d.license_number == "D1234567"
        assert d.license_state == "CA"
        assert d.mvr_status == "clean"
        assert result.warnings == []

    def test_full_name_column_splits_to_first_last(self):
        content = _csv([
            "Driver Name,License Number,MVR",
            "Alice Johnson,A123456,Clean",
        ])
        result = parse_fleet_sheet(content, "drivers.csv")
        assert len(result.drivers) == 1
        d = result.drivers[0]
        assert d.first_name == "Alice"
        assert d.last_name == "Johnson"


class TestParseCsvNoHeader:
    def test_no_keywords_warns_and_returns_empty(self):
        content = _csv([
            "Alpha,Beta,Gamma",
            "foo,bar,baz",
        ])
        result = parse_fleet_sheet(content, "mystery.csv")
        assert len(result.vehicles) == 0
        assert len(result.drivers) == 0
        assert any("no clear header" in w for w in result.warnings)


class TestParseXlsx:
    def test_vehicle_worksheet(self):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Fleet"
        ws.append(["VIN", "Make", "Model", "Year", "Stated Amount"])
        ws.append(["1HGBH41JXMN109186", "Honda", "Civic", 2021, 25000])
        ws.append(["2T1BURHE0JC054321", "Toyota", "Corolla", 2020, 22000])

        buf = io.BytesIO()
        wb.save(buf)
        content = buf.getvalue()

        result = parse_fleet_sheet(content, "fleet.xlsx")
        assert isinstance(result, FleetIngestResult)
        assert len(result.vehicles) == 2
        assert result.vehicles[0].vin == "1HGBH41JXMN109186"
        assert result.vehicles[1].make == "Toyota"
        assert result.warnings == []


class TestMergeFleetIntoSubmission:
    def test_auto_sets_commercial_auto_lob(self):
        content = _csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
        ])
        fleet = parse_fleet_sheet(content, "vehicles.csv")
        sub = {}
        merge_fleet_into_submission(sub, fleet)
        assert sub["lob_details"]["lob"] == "commercial_auto"
        assert len(sub["lob_details"]["vehicles"]) == 1

    def test_raises_on_wrong_lob(self):
        content = _csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
        ])
        fleet = parse_fleet_sheet(content, "vehicles.csv")
        sub = {"lob_details": {"lob": "general_liability"}}
        with pytest.raises(ValueError, match="commercial_auto"):
            merge_fleet_into_submission(sub, fleet)

    def test_dedupe_by_vin(self):
        content = _csv([
            "VIN,Make,Model,Year",
            "1HGBH41JXMN109186,Honda,Civic,2021",
        ])
        fleet = parse_fleet_sheet(content, "vehicles.csv")
        sub = {
            "lob_details": {
                "lob": "commercial_auto",
                "vehicles": [{"vin": "1HGBH41JXMN109186", "make": "Honda", "model": "Civic", "year": 2021}],
            }
        }
        counts = merge_fleet_into_submission(sub, fleet)
        assert counts["vehicles_added"] == 0
        assert counts["vehicles_updated"] == 1
        assert len(sub["lob_details"]["vehicles"]) == 1
