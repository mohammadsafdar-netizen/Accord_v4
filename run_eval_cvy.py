"""Step 3A — run ONLY correction-vehicle-year with diagnostic logging.

Emits extraction_route + extraction_output to logs/app.log so we can
identify the turn where SYSTEM_V2 vs SYSTEM_CORRECTION_V1 divergence
caused the regression (F1 1.0 → 0.625).
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
    configure_logging()
    all_scs = load_scenarios(pathlib.Path("eval/scenarios"))
    scs = [s for s in all_scs if s.id == "correction-vehicle-year"]
    print(f"Running {len(scs)} scenario(s):", flush=True)
    for s in scs:
        print(f"  - {s.id} ({len(s.turns)} turns)", flush=True)

    s = Settings(
        db_path="/tmp/eval_cvy.db",
        filled_pdf_dir="/tmp/eval_cvy_pdfs",
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
    pathlib.Path("/tmp/eval_cvy_report.json").write_text(
        json.dumps(out, indent=2)
    )
    print(f"\n=== Done in {elapsed:.1f}s ===", flush=True)
    print(f"f1={report.aggregate_f1:.4f}", flush=True)
    for r in report.scenarios:
        d = r.to_dict()
        print(
            f"  {r.scenario_id:35s}  F1={d['f1']:.3f}  "
            f"P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"matched={r.score.matched_v3_paths}/{d['total_expected']}",
            flush=True,
        )
        # Dump per-field comparison for failure diagnosis
        print("\n  Per-field comparisons:", flush=True)
        for c in r.score.comparisons:
            mark = "OK" if c.matched else "MISS"
            print(
                f"    [{mark}] v3={c.v3_path:35s} v4={c.v4_path:40s}  "
                f"exp={c.expected_value!r:30s}  act={c.actual_value!r}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
