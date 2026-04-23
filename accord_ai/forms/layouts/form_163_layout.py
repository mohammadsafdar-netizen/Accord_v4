"""Authoritative widget layout for ACORD 163 — Commercial Auto Driver Schedule.

ACORD 163 uses generic numbered widget names (Text1[0], Text2[0], …) instead
of the semantic names used by 125/127/129/137. The column ordering inside a
driver row is NOT systematic — e.g. in driver row 1, MI is at Text19 while
First Name is at Text16 and Address is at Text17. So a pure row_base+col
formula is wrong for multiple columns across every row.

This module provides a ground-truth map derived from the physical widget
coordinates in `form_templates/acord_163_blank.pdf` (widgets y-clustered
into driver rows, then sorted left-to-right by x). It supports up to 24
driver rows — the full capacity of the form.

Exposed API:
  HEADER_MAP — logical header key → Text##[0] widget name
  DRIVER_MAP — {row_idx: {column_key: Text##[0]}} for rows 1-24
  DRIVER_COLUMNS — ordered list of column keys (for UI / docs)
  map_structured(payload) -> dict[str, str]
      Given a structured payload (e.g. {"_header": {...}, "drivers": [...]})
      return a flat {widget_name: value} dict ready for pdf_filler.fill_all.

Fallback: callers may still send raw Text##[0] keys directly; map_structured
passes them through unchanged.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Header widgets — business / producer / policy info above the driver table.
# Derived from the header region (y < 160) of the blank template.
# ---------------------------------------------------------------------------

HEADER_MAP: dict[str, str] = {
    # This map mirrors the dual project's production `_HEADER_MAP` in
    # dual/pipeline/llm_field_mapper.py — the battle-tested mapping used
    # in every v1 production 163 fill.  Do not diverge without verifying
    # against a real filled PDF from the dual output/ folder.
    "policy_effective_date":   "Text1[0]",
    "producer_city":           "Text2[0]",
    "producer_zip":            "Text3[0]",
    "producer_state":          "Text4[0]",
    "producer_phone":          "Text5[0]",
    "producer_fax":            "Text6[0]",
    "producer_code":           "Text7[0]",    # agency producer code (NOT fax main)
    "producer_name":           "Text8[0]",
    "producer_address":        "Text9[0]",    # street line one
    "agency_code":             "Text10[0]",
    "sub_code":                "Text11[0]",
    "customer_id":             "Text12[0]",
    "business_name":           "Text13[0]",
    # Text14 is the business's FULL address (line1, city, state, zip joined)
    # — form 163 has no separate business city/state/zip widgets, so the
    # dual project concatenates them into this single field. Callers may
    # pass a pre-joined string as "business_full_address", OR supply the
    # address components individually via the _header keys below and let
    # map_structured combine them.
    "business_full_address":   "Text14[0]",
}


# Component keys a caller may pass under `_header` to have Text14 auto-built.
# If `business_full_address` is explicitly set, it wins (escape hatch).
_BUSINESS_ADDRESS_COMPONENT_KEYS = (
    "business_address",          # street line 1
    "business_city",
    "business_state",
    "business_zip",
)


# ---------------------------------------------------------------------------
# Driver row → widget map.
#
# Row keys are 1-indexed (1 = first driver on the form).
# Column keys match the conventional driver-entity shape.
# Columns that correspond to SSN / flags / driver_num are intentionally
# omitted — fill_pdf's clear-pass will leave them blank.
# ---------------------------------------------------------------------------

DRIVER_COLUMNS = (
    "first_name", "middle_initial", "last_name",
    "addr_line_one", "city", "state", "zip",
    "sex", "dob", "years_exp", "licensed_year",
    "license_num", "license_state", "hire_date",
    "vehicle_assigned", "pct_use",
)


DRIVER_MAP: dict[int, dict[str, str]] = {
    1: {
        "first_name": "Text16[0]", "middle_initial": "Text17[0]",
        "last_name": "Text18[0]", "addr_line_one": "Text19[0]",
        "city": "Text20[0]", "state": "Text21[0]", "zip": "Text22[0]",
        "sex": "Text23[0]", "dob": "Text24[0]", "years_exp": "Text25[0]",
        "licensed_year": "Text26[0]", "license_num": "Text27[0]",
        "license_state": "Text29[0]", "hire_date": "Text30[0]",
        "vehicle_assigned": "Text33[0]", "pct_use": "Text34[0]",
    },
    2: {
        "first_name": "Text36[0]", "middle_initial": "Text37[0]",
        "last_name": "Text38[0]", "addr_line_one": "Text39[0]",
        "city": "Text40[0]", "state": "Text41[0]", "zip": "Text42[0]",
        "sex": "Text43[0]", "dob": "Text44[0]", "years_exp": "Text45[0]",
        "licensed_year": "Text46[0]", "license_num": "Text47[0]",
        "license_state": "Text49[0]", "hire_date": "Text50[0]",
        "vehicle_assigned": "Text53[0]", "pct_use": "Text54[0]",
    },
    3: {
        "first_name": "Text56[0]", "middle_initial": "Text57[0]",
        "last_name": "Text58[0]", "addr_line_one": "Text59[0]",
        "city": "Text60[0]", "state": "Text61[0]", "zip": "Text62[0]",
        "sex": "Text63[0]", "dob": "Text64[0]", "years_exp": "Text65[0]",
        "licensed_year": "Text66[0]", "license_num": "Text66a[0]",
        "license_state": "Text68[0]", "hire_date": "Text69[0]",
        "vehicle_assigned": "Text72[0]", "pct_use": "Text73[0]",
    },
    4: {
        "first_name": "Text75[0]", "middle_initial": "Text76[0]",
        "last_name": "Text77[0]", "addr_line_one": "Text78[0]",
        "city": "Text79[0]", "state": "Text80[0]", "zip": "Text81[0]",
        "sex": "Text82[0]", "dob": "Text83[0]", "years_exp": "Text84[0]",
        "licensed_year": "Text85[0]", "license_num": "Text86[0]",
        "license_state": "Text88[0]", "hire_date": "Text89[0]",
        "vehicle_assigned": "Text92[0]", "pct_use": "Text93[0]",
    },
    5: {
        "first_name": "Text95[0]", "middle_initial": "Text96[0]",
        "last_name": "Text97[0]", "addr_line_one": "Text98[0]",
        "city": "Text99[0]", "state": "Text100[0]", "zip": "Text101[0]",
        "sex": "Text102[0]", "dob": "Text103[0]", "years_exp": "Text104[0]",
        "licensed_year": "Text105[0]", "license_num": "Text106[0]",
        "license_state": "Text108[0]", "hire_date": "Text109[0]",
        "vehicle_assigned": "Text112[0]", "pct_use": "Text113[0]",
    },
    6: {
        "first_name": "Text115[0]", "middle_initial": "Text116[0]",
        "last_name": "Text117[0]", "addr_line_one": "Text118[0]",
        "city": "Text119[0]", "state": "Text120[0]", "zip": "Text121[0]",
        "sex": "Text122[0]", "dob": "Text123[0]", "years_exp": "Text124[0]",
        "licensed_year": "Text125[0]", "license_num": "Text127[0]",
        "license_state": "Text129[0]", "hire_date": "Text130[0]",
        "vehicle_assigned": "Text133[0]", "pct_use": "Text134[0]",
    },
    7: {
        "first_name": "Text136[0]", "middle_initial": "Text137[0]",
        "last_name": "Text138[0]", "addr_line_one": "Text139[0]",
        "city": "Text140[0]", "state": "Text141[0]", "zip": "Text142[0]",
        "sex": "Text143[0]", "dob": "Text144[0]", "years_exp": "Text145[0]",
        "licensed_year": "Text146[0]", "license_num": "Text147[0]",
        "license_state": "Text149[0]", "hire_date": "Text150[0]",
        "vehicle_assigned": "Text153[0]", "pct_use": "Text154[0]",
    },
    8: {
        "first_name": "Text156[0]", "middle_initial": "Text157[0]",
        "last_name": "Text158[0]", "addr_line_one": "Text159[0]",
        "city": "Text160[0]", "state": "Text161[0]", "zip": "Text162[0]",
        "sex": "Text163[0]", "dob": "Text164[0]", "years_exp": "Text165[0]",
        "licensed_year": "Text166[0]", "license_num": "Text167[0]",
        "license_state": "Text169[0]", "hire_date": "Text170[0]",
        "vehicle_assigned": "Text173[0]", "pct_use": "Text174[0]",
    },
    9: {
        "first_name": "Text176[0]", "middle_initial": "Text177[0]",
        "last_name": "Text178[0]", "addr_line_one": "Text179[0]",
        "city": "Text180[0]", "state": "Text181[0]", "zip": "Text182[0]",
        "sex": "Text183[0]", "dob": "Text184[0]", "years_exp": "Text185[0]",
        "licensed_year": "Text186[0]", "license_num": "Text187[0]",
        "license_state": "Text189[0]", "hire_date": "Text190[0]",
        "vehicle_assigned": "Text193[0]", "pct_use": "Text194[0]",
    },
    10: {
        "first_name": "Text196[0]", "middle_initial": "Text197[0]",
        "last_name": "Text198[0]", "addr_line_one": "Text199[0]",
        "city": "Text200[0]", "state": "Text201[0]", "zip": "Text202[0]",
        "sex": "Text203[0]", "dob": "Text204[0]", "years_exp": "Text205[0]",
        "licensed_year": "Text206[0]", "license_num": "Text207[0]",
        "license_state": "Text209[0]", "hire_date": "Text210[0]",
        "vehicle_assigned": "Text213[0]", "pct_use": "Text214[0]",
    },
    11: {
        "first_name": "Text216[0]", "middle_initial": "Text217[0]",
        "last_name": "Text218[0]", "addr_line_one": "Text219[0]",
        "city": "Text220[0]", "state": "Text221[0]", "zip": "Text222[0]",
        "sex": "Text223[0]", "dob": "Text224[0]", "years_exp": "Text225[0]",
        "licensed_year": "Text226[0]", "license_num": "Text227[0]",
        "license_state": "Text229[0]", "hire_date": "Text230[0]",
        "vehicle_assigned": "Text233[0]", "pct_use": "Text234[0]",
    },
    12: {
        "first_name": "Text236[0]", "middle_initial": "Text237[0]",
        "last_name": "Text238[0]", "addr_line_one": "Text239[0]",
        "city": "Text240[0]", "state": "Text241[0]", "zip": "Text242[0]",
        "sex": "Text243[0]", "dob": "Text244[0]", "years_exp": "Text245[0]",
        "licensed_year": "Text246[0]", "license_num": "Text247[0]",
        "license_state": "Text249[0]", "hire_date": "Text250[0]",
        "vehicle_assigned": "Text253[0]", "pct_use": "Text254[0]",
    },
    13: {
        "first_name": "Text256[0]", "middle_initial": "Text257[0]",
        "last_name": "Text258[0]", "addr_line_one": "Text259[0]",
        "city": "Text260[0]", "state": "Text261[0]", "zip": "Text262[0]",
        "sex": "Text263[0]", "dob": "Text264[0]", "years_exp": "Text265[0]",
        "licensed_year": "Text266[0]", "license_num": "Text267[0]",
        "license_state": "Text269[0]", "hire_date": "Text270[0]",
        "vehicle_assigned": "Text273[0]", "pct_use": "Text274[0]",
    },
    14: {
        "first_name": "Text276[0]", "middle_initial": "Text277[0]",
        "last_name": "Text278[0]", "addr_line_one": "Text279[0]",
        "city": "Text280[0]", "state": "Text281[0]", "zip": "Text282[0]",
        "sex": "Text283[0]", "dob": "Text284[0]", "years_exp": "Text285[0]",
        "licensed_year": "Text286[0]", "license_num": "Text287[0]",
        "license_state": "Text289[0]", "hire_date": "Text290[0]",
        "vehicle_assigned": "Text293[0]", "pct_use": "Text294[0]",
    },
    15: {
        "first_name": "Text296[0]", "middle_initial": "Text297[0]",
        "last_name": "Text298[0]", "addr_line_one": "Text299[0]",
        "city": "Text300[0]", "state": "Text301[0]", "zip": "Text302[0]",
        "sex": "Text303[0]", "dob": "Text304[0]", "years_exp": "Text305[0]",
        "licensed_year": "Text306[0]", "license_num": "Text307[0]",
        "license_state": "Text309[0]", "hire_date": "Text310[0]",
        "vehicle_assigned": "Text313[0]", "pct_use": "Text314[0]",
    },
    16: {
        "first_name": "Text316[0]", "middle_initial": "Text317[0]",
        "last_name": "Text318[0]", "addr_line_one": "Text319[0]",
        "city": "Text320[0]", "state": "Text321[0]", "zip": "Text322[0]",
        "sex": "Text323[0]", "dob": "Text324[0]", "years_exp": "Text325[0]",
        "licensed_year": "Text326[0]", "license_num": "Text327[0]",
        "license_state": "Text329[0]", "hire_date": "Text330[0]",
        "vehicle_assigned": "Text333[0]", "pct_use": "Text334[0]",
    },
    17: {
        "first_name": "Text336[0]", "middle_initial": "Text337[0]",
        "last_name": "Text338[0]", "addr_line_one": "Text339[0]",
        "city": "Text340[0]", "state": "Text341[0]", "zip": "Text342[0]",
        "sex": "Text343[0]", "dob": "Text344[0]", "years_exp": "Text345[0]",
        "licensed_year": "Text346[0]", "license_num": "Text347[0]",
        "license_state": "Text349[0]", "hire_date": "Text350[0]",
        "vehicle_assigned": "Text353[0]", "pct_use": "Text354[0]",
    },
    18: {
        "first_name": "Text356[0]", "middle_initial": "Text357[0]",
        "last_name": "Text358[0]", "addr_line_one": "Text359[0]",
        "city": "Text360[0]", "state": "Text361[0]", "zip": "Text362[0]",
        "sex": "Text363[0]", "dob": "Text364[0]", "years_exp": "Text365[0]",
        "licensed_year": "Text366[0]", "license_num": "Text367[0]",
        "license_state": "Text369[0]", "hire_date": "Text370[0]",
        "vehicle_assigned": "Text373[0]", "pct_use": "Text374[0]",
    },
    19: {
        "first_name": "Text376[0]", "middle_initial": "Text377[0]",
        "last_name": "Text378[0]", "addr_line_one": "Text379[0]",
        "city": "Text380[0]", "state": "Text381[0]", "zip": "Text382[0]",
        "sex": "Text383[0]", "dob": "Text384[0]", "years_exp": "Text385[0]",
        "licensed_year": "Text386[0]", "license_num": "Text387[0]",
        "license_state": "Text389[0]", "hire_date": "Text390[0]",
        "vehicle_assigned": "Text393[0]", "pct_use": "Text394[0]",
    },
    20: {
        "first_name": "Text396[0]", "middle_initial": "Text397[0]",
        "last_name": "Text398[0]", "addr_line_one": "Text399[0]",
        "city": "Text400[0]", "state": "Text401[0]", "zip": "Text402[0]",
        "sex": "Text403[0]", "dob": "Text404[0]", "years_exp": "Text405[0]",
        "licensed_year": "Text406[0]", "license_num": "Text407[0]",
        "license_state": "Text409[0]", "hire_date": "Text410[0]",
        "vehicle_assigned": "Text413[0]", "pct_use": "Text414[0]",
    },
    21: {
        "first_name": "Text416[0]", "middle_initial": "Text417[0]",
        "last_name": "Text418[0]", "addr_line_one": "Text419[0]",
        "city": "Text420[0]", "state": "Text421[0]", "zip": "Text422[0]",
        "sex": "Text423[0]", "dob": "Text424[0]", "years_exp": "Text425[0]",
        "licensed_year": "Text426[0]", "license_num": "Text427[0]",
        "license_state": "Text429[0]", "hire_date": "Text430[0]",
        "vehicle_assigned": "Text433[0]", "pct_use": "Text434[0]",
    },
    22: {
        "first_name": "Text436[0]", "middle_initial": "Text437[0]",
        "last_name": "Text438[0]", "addr_line_one": "Text439[0]",
        "city": "Text440[0]", "state": "Text441[0]", "zip": "Text442[0]",
        "sex": "Text443[0]", "dob": "Text444[0]", "years_exp": "Text445[0]",
        "licensed_year": "Text446[0]", "license_num": "Text447[0]",
        "license_state": "Text449[0]", "hire_date": "Text450[0]",
        "vehicle_assigned": "Text453[0]", "pct_use": "Text454[0]",
    },
    23: {
        "first_name": "Text456[0]", "middle_initial": "Text457[0]",
        "last_name": "Text458[0]", "addr_line_one": "Text459[0]",
        "city": "Text460[0]", "state": "Text461[0]", "zip": "Text462[0]",
        "sex": "Text463[0]", "dob": "Text464[0]", "years_exp": "Text465[0]",
        "licensed_year": "Text466[0]", "license_num": "Text467[0]",
        "license_state": "Text469[0]", "hire_date": "Text470[0]",
        "vehicle_assigned": "Text473[0]", "pct_use": "Text474[0]",
    },
    24: {
        "first_name": "Text476[0]", "middle_initial": "Text477[0]",
        "last_name": "Text478[0]", "addr_line_one": "Text479[0]",
        "city": "Text480[0]", "state": "Text481[0]", "zip": "Text482[0]",
        "sex": "Text483[0]", "dob": "Text484[0]", "years_exp": "Text485[0]",
        "licensed_year": "Text486[0]", "license_num": "Text487[0]",
        "license_state": "Text489[0]", "hire_date": "Text490[0]",
        "vehicle_assigned": "Text493[0]", "pct_use": "Text494[0]",
    },
}


MAX_DRIVERS = max(DRIVER_MAP.keys())


# ---------------------------------------------------------------------------
# Structured input → flat widget dict.
# ---------------------------------------------------------------------------

def _is_structured(payload: dict[str, Any]) -> bool:
    """True if payload uses the logical-field shape (has _header or drivers),
    False if it's a flat map of Text##[0] → value (raw passthrough)."""
    if not isinstance(payload, dict):
        return False
    if "_header" in payload or "drivers" in payload:
        return True
    # Any non-Text-numbered key means structured input
    return any(not k.startswith("Text") for k in payload.keys())


def map_structured(payload: dict[str, Any]) -> dict[str, str]:
    """Translate a structured payload into a flat widget-value dict.

    Structured payload shape:
        {
          "_header": {"business_name": "...", "policy_effective_date": "...", ...},
          "drivers": [
              {"first_name": "...", "dob": "...", ...},
              ...
          ]
        }

    Raw Text##[0] keys in the payload pass through unchanged (escape hatch).
    Unknown keys are silently dropped (so callers can include comments in
    their logical form without errors).
    """
    result: dict[str, str] = {}
    if not isinstance(payload, dict):
        return result

    # 1) Raw widget keys: passthrough.
    for k, v in payload.items():
        if isinstance(k, str) and k.startswith("Text") and "[0]" in k:
            if v is not None and str(v).strip():
                result[k] = str(v)

    # 2) Header block.
    header = payload.get("_header", {})
    if isinstance(header, dict):
        for logical_key, value in header.items():
            widget = HEADER_MAP.get(logical_key)
            if widget and value is not None and str(value).strip():
                result[widget] = str(value)

        # Build Text14 (business full address) from components if the caller
        # didn't pass business_full_address explicitly. Mirrors dual's logic:
        #     "line1, city, state zip"
        if HEADER_MAP["business_full_address"] not in result:
            parts = []
            line1 = (header.get("business_address") or "").strip()
            if line1:
                parts.append(line1)
            city = (header.get("business_city") or "").strip()
            state = (header.get("business_state") or "").strip()
            zipc = (header.get("business_zip") or "").strip()
            tail = []
            if city:
                tail.append(city)
            if state:
                tail.append(state)
            tail_str = ", ".join(tail[:2])  # "city, state"
            if zipc:
                tail_str = f"{tail_str} {zipc}".strip()
            if tail_str:
                parts.append(tail_str)
            if parts:
                result[HEADER_MAP["business_full_address"]] = ", ".join(parts)

    # 3) Drivers array — each dict maps to a row.
    drivers = payload.get("drivers", [])
    if isinstance(drivers, list):
        for idx, driver in enumerate(drivers[:MAX_DRIVERS]):
            if not isinstance(driver, dict):
                continue
            row_idx = idx + 1  # drivers are 1-indexed
            row_map = DRIVER_MAP.get(row_idx)
            if not row_map:
                continue
            for logical_key, value in driver.items():
                widget = row_map.get(logical_key)
                if widget and value is not None and str(value).strip():
                    result[widget] = str(value)

    return result
