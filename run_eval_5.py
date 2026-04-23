"""Focused 5-scenario eval — fast signal across the most representative classes.

Pick set covers:
  * standard-solo-plumber       — small business / single vehicle (broker baseline)
  * standard-delivery-fleet     — small fleet (was already best — regression check)
  * bulk-all-business-info      — long single-turn dump (extra='ignore' fix target)
  * multi-five-vehicle-fleet    — medium fleet (numeric-disambig + correction stress)
  * negation-no-hired-auto      — direct test of harness negation rules
"""
import asyncio
import json
import pathlib
import time

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import load_scenarios, run_all


WANTED = {
    "standard-solo-plumber",
    "standard-delivery-fleet",
    "bulk-all-business-info",
    "multi-five-vehicle-fleet",
    "negation-no-hired-auto",
}


async def main() -> None:
    all_scs = load_scenarios(pathlib.Path("eval/scenarios"))
    scs = [s for s in all_scs if s.id in WANTED]
    print(f"Running {len(scs)}/{len(WANTED)} target scenarios:", flush=True)
    for s in scs:
        print(f"  - {s.id}", flush=True)

    s = Settings(
        db_path="/tmp/eval_5.db",
        filled_pdf_dir="/tmp/eval_5_pdfs",
        accord_auth_disabled=True,
        harness_max_refines=1,
        llm_timeout_s=120.0,
        llm_retries=1,
    )
    app = build_intake_app(s)

    t0 = time.perf_counter()
    report = await run_all(app, scs, concurrency=1)
    elapsed = time.perf_counter() - t0

    out = report.to_dict()
    out["elapsed_seconds"] = elapsed
    pathlib.Path("/tmp/eval_5_report.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\n=== Done in {elapsed:.1f}s ===", flush=True)
    print(f"precision={report.aggregate_precision:.4f}", flush=True)
    print(f"recall={report.aggregate_recall:.4f}", flush=True)
    print(f"f1={report.aggregate_f1:.4f}", flush=True)
    print(
        f"turns total={report.total_turns} fp={report.first_pass_passed} "
        f"rescue={report.refiner_rescued} fail={report.still_failing}",
        flush=True,
    )
    print("\n=== Per scenario ===", flush=True)
    for r in report.scenarios:
        d = r.to_dict()
        print(
            f"  {r.scenario_id:35s}  P={d['precision']:.3f}  "
            f"R={d['recall']:.3f}  F1={d['f1']:.3f}  "
            f"v3={r.score.matched_v3_paths}/{d['total_expected']}  "
            f"turns(fp={d['turns']['first_pass_passed']}/"
            f"rescue={d['turns']['refiner_rescued']}/"
            f"fail={d['turns']['still_failing']})",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
