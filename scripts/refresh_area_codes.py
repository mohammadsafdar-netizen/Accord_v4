#!/usr/bin/env python3
"""Refresh data/area_codes.csv from NANPA public data.

Usage:
    uv run scripts/refresh_area_codes.py

Downloads the NANPA NPA (area code) list and writes to data/area_codes.csv.
Run when a new area code is activated (NANPA maintains the authoritative list).

Source: https://www.nanpa.com/area-codes/view-area-code-maps
NANPA publishes a downloadable spreadsheet; we convert it to CSV.
"""

from __future__ import annotations

import csv
import io
import pathlib
import sys
import urllib.request


_NANPA_URL = "https://www.nanpa.com/nanp1/npa_report.csv"
_OUT_PATH = pathlib.Path(__file__).parent.parent / "data" / "area_codes.csv"


def _download() -> list[dict]:
    print(f"Downloading {_NANPA_URL} ...", flush=True)
    try:
        req = urllib.request.Request(
            _NANPA_URL,
            headers={"User-Agent": "Accord v4 area-code refresh script"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8-sig", errors="replace")
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = []
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        # NANPA CSV columns vary by year; try multiple spellings
        code = (
            row.get("NPA") or row.get("Area Code") or row.get("area_code") or ""
        ).strip()
        state = (
            row.get("State/Province/Territory") or row.get("State") or row.get("state") or ""
        ).strip()
        region = (
            row.get("Location") or row.get("Region") or row.get("region") or state
        ).strip()
        if code and code.isdigit() and len(code) == 3:
            rows.append({
                "area_code": code,
                "state": state[:2].upper() if len(state) >= 2 else state,
                "region": region,
            })
    return rows


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = _download()
    if not rows:
        print("ERROR: no rows parsed from NANPA download", file=sys.stderr)
        sys.exit(1)

    with _OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["area_code", "state", "region"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} area codes to {_OUT_PATH}")


if __name__ == "__main__":
    main()
