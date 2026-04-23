#!/usr/bin/env python3
"""Download the current NAICS 2022 code list from the Census Bureau.

Usage:
    uv run scripts/refresh_naics.py

Downloads to data/naics_2022.csv (overwriting). Run every 5 years or when
the Census Bureau publishes a new NAICS revision.

Census source:
  https://www.census.gov/naics/?58967?yearbck=2022
  ZIP file contains "2022_NAICS_Structure_Summary_Table.xlsx" and
  "2022.naics.xls" — we convert the latter to CSV.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
import zipfile
import io
import pathlib


_CENSUS_ZIP_URL = (
    "https://www.census.gov/naics/2022NAICS/2022_NAICS_Structure.zip"
)
_OUT_PATH = pathlib.Path(__file__).parent.parent / "data" / "naics_2022.csv"


def _download_and_extract() -> list[tuple[str, str]]:
    print(f"Downloading {_CENSUS_ZIP_URL} ...", flush=True)
    with urllib.request.urlopen(_CENSUS_ZIP_URL, timeout=60) as resp:
        raw = resp.read()

    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Census ships a .csv or .xlsx — try the CSV variant first
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in ZIP. Files: {zf.namelist()}")
        with zf.open(csv_names[0]) as fh:
            reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
            header = next(reader, None)
            if header is None:
                raise RuntimeError("Empty CSV in ZIP")
            # Detect code/title column indices
            header_lower = [h.lower() for h in header]
            code_col = next(
                (i for i, h in enumerate(header_lower) if "code" in h), 0
            )
            title_col = next(
                (i for i, h in enumerate(header_lower) if "title" in h), 1
            )
            for row in reader:
                if len(row) <= max(code_col, title_col):
                    continue
                code = row[code_col].strip()
                title = row[title_col].strip()
                if code and title:
                    rows.append((code, title))
    return rows


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = _download_and_extract()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    with _OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "title"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} NAICS codes to {_OUT_PATH}")


if __name__ == "__main__":
    main()
