"""Commercial Auto LOB plugin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from accord_ai.lobs.common import COMMON_CRITICAL
from accord_ai.lobs.registry import register

# CA-specific fields beyond the common baseline.
# Verbatim port from accord_ai_v3/lobs/commercial_auto/__init__.py.
_CA_SPECIFIC: List[Tuple[str, str]] = [
    ("contacts[0].full_name",                  "a contact name is required"),
    ("contacts[0].phone",                      "contact phone is required"),
    ("contacts[0].email",                      "contact email is required"),
    ("full_time_employees",                    "full_time_employees is required"),
    ("nature_of_business",                     "nature_of_business is required"),
    ("lob_details.fleet_use_type",             "fleet use_type is required (service/commercial/retail/pleasure)"),
    ("lob_details.fleet_for_hire",             "fleet_for_hire is required (true/false)"),
    ("lob_details.states_of_operation",        "states_of_operation is required (list of state codes)"),
    ("lob_details.radius_of_operations",       "radius_of_operations is required"),
    ("lob_details.vehicle_count",              "vehicle_count is required"),
    ("lob_details.hazmat",                     "hazmat (true/false) is required"),
    ("lob_details.trailer_interchange",        "trailer_interchange (true/false) is required"),
    ("lob_details.driver_training",            "driver_training (true/false) is required"),
]


@dataclass(frozen=True)
class CommercialAutoPlugin:
    lob_key: str = "commercial_auto"

    @property
    def critical_fields(self) -> List[Tuple[str, str]]:
        return [*COMMON_CRITICAL, *_CA_SPECIFIC]


register(CommercialAutoPlugin())
