"""Full 67-scenario eval run — writes JSON report to eval_results/<tag>.json.

Run from accord_v4/ so ``accord_ai`` resolves on sys.path.

Environment overrides:
  ACCORD_DISABLE_REFINEMENT=1   sets harness_max_refines=0 (reproducibility)
  LLM_SEED=<int>                pins vLLM's internal RNG (determinism at temp=0)
  EXTRACTION_CONTEXT=false      disables flow-scoped extraction context hints
  EXTRACTION_MODE=<mode>        xgrammar | json_object | free (Step 25)
  EXPERIMENT_HARNESS=<level>    none | light | full (Step 25)
"""
import argparse
import asyncio
import json
import os
import pathlib
import time

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import load_scenarios, run_all


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tag",
        default="run",
        help="Label for this run — report written to eval_results/<tag>.json",
    )
    return p.parse_args()


async def main(tag: str) -> None:
    scs = load_scenarios(pathlib.Path("eval/scenarios"))
    print(f"Loaded {len(scs)} scenarios", flush=True)

    disable_refinement = os.environ.get("ACCORD_DISABLE_REFINEMENT", "0") == "1"
    seed_str = os.environ.get("LLM_SEED")

    s = Settings(
        db_path=f"/tmp/eval_67_{tag}.db",
        filled_pdf_dir=f"/tmp/eval_67_{tag}_pdfs",
        accord_auth_disabled=True,
        harness_max_refines=0 if disable_refinement else 1,
        llm_timeout_s=120.0,
        llm_retries=1,
        llm_seed=int(seed_str) if seed_str else None,
    )

    print(
        f"Config: refinement={'OFF' if disable_refinement else 'ON'} "
        f"seed={s.llm_seed} mode={s.extraction_mode.value} "
        f"harness={s.experiment_harness} context={s.extraction_context}",
        flush=True,
    )

    app = build_intake_app(s)

    t0 = time.perf_counter()
    report = await run_all(app, scs, concurrency=1)
    elapsed = time.perf_counter() - t0

    out = report.to_dict()
    out["elapsed_seconds"] = elapsed
    out["tag"] = tag
    out["config"] = {
        "refinement": not disable_refinement,
        "seed": s.llm_seed,
        "extraction_mode": s.extraction_mode.value,
        "experiment_harness": s.experiment_harness,
        "extraction_context": s.extraction_context,
    }

    results_dir = pathlib.Path("eval_results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Legacy path for backward compat
    pathlib.Path("/tmp/eval_67_report.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Done in {elapsed:.1f}s ===", flush=True)
    print(f"tag={tag}", flush=True)
    print(f"scenarios={len(report.scenarios)}", flush=True)
    print(f"precision={report.aggregate_precision:.4f}", flush=True)
    print(f"recall={report.aggregate_recall:.4f}", flush=True)
    print(f"f1={report.aggregate_f1:.4f}", flush=True)
    print(
        f"turns total={report.total_turns} fp={report.first_pass_passed} "
        f"rescue={report.refiner_rescued} fail={report.still_failing}",
        flush=True,
    )


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.tag))
