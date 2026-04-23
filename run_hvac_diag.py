"""Diagnostic: HVAC contractor per-field comparison."""
import asyncio
import pathlib
import time

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import load_scenarios, run_all
from accord_ai.logging_config import configure_logging


async def main():
    configure_logging()
    all_scs = load_scenarios(pathlib.Path("eval/scenarios"))
    scs = [s for s in all_scs if s.id == "standard-hvac-contractor"]

    s = Settings(
        db_path="/tmp/eval_hvac.db", filled_pdf_dir="/tmp/eval_hvac_pdfs",
        accord_auth_disabled=True, harness_max_refines=1,
        llm_timeout_s=120.0, llm_retries=1,
    )
    app = build_intake_app(s)
    report = await run_all(app, scs, concurrency=1)

    for r in report.scenarios:
        d = r.to_dict()
        print(f"F1={d['f1']:.3f}  P={d['precision']:.3f}  R={d['recall']:.3f}")
        print(f"matched={r.score.matched_v3_paths}/{d['total_expected']}  turns={d['turns']}")
        print("\nPer-field comparisons:")
        for c in r.score.comparisons:
            mark = "OK" if c.matched else "MISS"
            print(
                f"  [{mark}] v3={c.v3_path:40s} "
                f"exp={c.expected_value!r:25s} act={c.actual_value!r}  reason={c.reason}"
            )


if __name__ == "__main__":
    asyncio.run(main())
