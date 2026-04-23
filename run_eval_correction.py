"""Phase A postmortem 1A — run only the correction-family scenarios.

Correlates with the extraction_route + extraction_output log lines in
logs/app.log to diagnose why Step 4 (correction-detection) regressed
the correction family on the 55-eval.
"""
import asyncio
import json
import pathlib
import time

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import load_scenarios, run_all
from accord_ai.logging_config import configure_logging


async def main() -> None:
    # Eval runners normally skip configure_logging (only the FastAPI
    # factory calls it). For the postmortem we need the extraction_route
    # / extraction_output traces in logs/app.log — call it here.
    configure_logging()
    all_scs = load_scenarios(pathlib.Path("eval/scenarios"))
    scs = [s for s in all_scs if s.id.startswith("correction-")]
    print(f"Running {len(scs)} correction scenarios:", flush=True)
    for s in scs:
        print(f"  - {s.id} ({len(s.turns)} turns)", flush=True)

    s = Settings(
        db_path="/tmp/eval_correction.db",
        filled_pdf_dir="/tmp/eval_correction_pdfs",
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
    pathlib.Path("/tmp/eval_correction_report.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\n=== Done in {elapsed:.1f}s ===", flush=True)
    print(f"f1={report.aggregate_f1:.4f}", flush=True)
    print("\n=== Per scenario ===", flush=True)
    for r in report.scenarios:
        d = r.to_dict()
        print(
            f"  {r.scenario_id:35s}  F1={d['f1']:.3f}  "
            f"P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"matched={r.score.matched_v3_paths}/{d['total_expected']}  "
            f"turns(fp={d['turns']['first_pass_passed']}/"
            f"rescue={d['turns']['refiner_rescued']}/"
            f"fail={d['turns']['still_failing']})",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
