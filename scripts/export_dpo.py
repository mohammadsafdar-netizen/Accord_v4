"""Export pending corrections for a broker as a DPO training JSONL.

Usage:
  uv run python scripts/export_dpo.py --tenant acme_trucking
  uv run python scripts/export_dpo.py --tenant acme_trucking --force
  uv run python scripts/export_dpo.py --all    # every tenant above threshold
"""
from __future__ import annotations

import argparse
import sys

from accord_ai.config import Settings
from accord_ai.feedback.dpo import DPOManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DPO training pairs per broker.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tenant", type=str, help="Tenant slug to export.")
    group.add_argument("--all", action="store_true", help="Export all tenants above threshold.")
    parser.add_argument("--force", action="store_true", help="Export even if below threshold.")
    args = parser.parse_args()

    settings = Settings()
    mgr = DPOManager(
        db_path=settings.db_path,
        output_dir=settings.training_data_dir,
        threshold=settings.dpo_threshold,
    )

    tenants = [args.tenant] if args.tenant else _discover_tenants(mgr)
    if not tenants:
        print("No tenants with pending corrections found.")
        sys.exit(0)

    for tenant in tenants:
        pending = mgr.count_pending(tenant)
        if not args.force and not mgr.eligible_for_training(tenant):
            print(
                f"SKIP {tenant}: {pending} pending < {mgr._threshold} threshold"
                " (use --force to override)"
            )
            continue
        result = mgr.export(tenant)
        if result.count == 0:
            print(f"SKIP {tenant}: no exportable pairs (all may be graduated)")
        else:
            print(f"EXPORT {tenant}: {result.count} pairs → {result.path}")


def _discover_tenants(mgr: DPOManager) -> list[str]:
    return mgr.list_tenants_with_pending()


if __name__ == "__main__":
    main()
