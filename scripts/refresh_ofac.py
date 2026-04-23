#!/usr/bin/env python3
"""Download the latest OFAC SDN CSV to data/ofac/sdn.csv.

Usage:
    uv run scripts/refresh_ofac.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accord_ai.validation.ofac import download_sdn, _CACHE_FILE

if __name__ == "__main__":
    print(f"Downloading OFAC SDN list → {_CACHE_FILE}")
    ok = download_sdn(timeout=60.0)
    if ok:
        print(f"Done. {_CACHE_FILE.stat().st_size:,} bytes")
    else:
        print("Download failed.", file=sys.stderr)
        sys.exit(1)
