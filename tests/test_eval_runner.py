"""Scenario-runner tests (P10.S.11b).

FakeEngine-driven. Each test queues deterministic extractor / refiner /
responder responses and asserts on the rolled-up ScenarioResult + the
aggregate RunReport. No network, no vLLM, safe for CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from accord_ai.app import build_intake_app
from accord_ai.config import Settings
from accord_ai.eval import (
    RunReport,
    Scenario,
    ScenarioResult,
    load_scenarios,
    run_all,
    run_scenario,
)
from accord_ai.llm.fake_engine import FakeEngine

from tests._fixtures import valid_ca_dict


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def _write_yaml(p: Path, payload: dict) -> None:
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_scenarios_from_single_file(tmp_path):
    f = tmp_path / "one.yaml"
    _write_yaml(f, {"scenarios": [
        {
            "id": "a1",
            "name": "Alpha",
            "turns": ["hi"],
            "expected": {"business.business_name": "Acme"},
            "tags": ["standard"],
        },
    ]})
    scs = load_scenarios(f)
    assert len(scs) == 1
    assert scs[0].id == "a1"
    assert scs[0].turns == ("hi",)


def test_load_scenarios_from_directory(tmp_path):
    _write_yaml(tmp_path / "a.yaml", {"scenarios": [
        {"id": "a1", "turns": ["hi"],
         "expected": {"business.business_name": "Acme"}},
    ]})
    _write_yaml(tmp_path / "b.yaml", {"scenarios": [
        {"id": "b1", "turns": ["hello"],
         "expected": {"business.business_name": "Other"}},
        {"id": "b2", "turns": ["hey"],
         "expected": {"business.business_name": "Third"}},
    ]})
    scs = load_scenarios(tmp_path)
    ids = sorted(s.id for s in scs)
    assert ids == ["a1", "b1", "b2"]


def test_load_scenarios_skips_malformed_entries(tmp_path):
    _write_yaml(tmp_path / "bad.yaml", {"scenarios": [
        {"id": "ok", "turns": ["hi"], "expected": {"business.business_name": "A"}},
        {"id": "no-turns", "expected": {"x": "y"}},        # missing turns
        {"id": "no-expected", "turns": ["hi"]},             # missing expected
    ]})
    scs = load_scenarios(tmp_path / "bad.yaml")
    assert [s.id for s in scs] == ["ok"]


def test_load_scenarios_accepts_single_scenario_at_root(tmp_path):
    """v3 dual/-style YAML has turns + expected at the root with no
    ``scenarios:`` block. The loader recognizes that shape too."""
    f = tmp_path / "solo.yaml"
    _write_yaml(f, {
        "turns": ["a message"],
        "expected": {"business.business_name": "Acme"},
    })
    scs = load_scenarios(f)
    assert len(scs) == 1
    assert scs[0].id == "solo"


# ---------------------------------------------------------------------------
# Single-scenario run
# ---------------------------------------------------------------------------

def _app(tmp_path, engine, *, refiner=None):
    s = Settings(
        db_path=str(tmp_path / "runner.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=0,         # deterministic — no refiner fire-by-default
    )
    return build_intake_app(s, engine=engine, refiner_engine=refiner or FakeEngine())


@pytest.mark.asyncio
async def test_run_scenario_scores_final_submission(tmp_path):
    """One turn → one extraction → final submission scored against expected."""
    engine = FakeEngine([
        valid_ca_dict(),         # turn 1 extract
        "greeting-response-1",   # turn 1 respond
    ])
    app = _app(tmp_path, engine)
    scn = Scenario(
        id="s1",
        name="Solo",
        turns=("we are Acme Trucking",),
        expected={"business.business_name": "Acme Trucking"},
    )
    result = await run_scenario(app, scn)
    assert isinstance(result, ScenarioResult)
    assert result.scenario_id == "s1"
    assert result.score.matched == 1
    assert result.score.precision == 1.0
    assert result.turn_stats.total == 1
    # harness_max_refines=0 disables the refiner → every turn is
    # either first-pass-passed or still-failing.
    assert result.turn_stats.refiner_rescued == 0


@pytest.mark.asyncio
async def test_run_scenario_multi_turn_accumulates_submission(tmp_path):
    """Two turns write different fields; the final submission has both."""
    engine = FakeEngine([
        {"business_name": "Acme"},   "resp-1",
        {"ein": "12-3456789"},        "resp-2",
    ])
    app = _app(tmp_path, engine)
    scn = Scenario(
        id="s2",
        name="MultiTurn",
        turns=("we are Acme", "ein is 12-3456789"),
        expected={
            "business.business_name": "Acme",
            "business.tax_id":        "12-3456789",
        },
    )
    result = await run_scenario(app, scn)
    assert result.score.matched == 2
    assert result.score.precision == 1.0
    assert result.turn_stats.total == 2


@pytest.mark.asyncio
async def test_run_scenario_counts_first_pass_vs_rescue(tmp_path):
    """Refiner rescue split is observable: max_refines=1, first extract is
    partial, refiner fills in the rest → turn counts as rescued."""
    main = FakeEngine([
        {"business_name": "Acme"},   "resp-1",     # extractor + responder turn 1
    ])
    # Refiner rescues: returns a fully-valid submission.
    refiner = FakeEngine([valid_ca_dict()])
    s = Settings(
        db_path=str(tmp_path / "rescue.db"),
        filled_pdf_dir=str(tmp_path / "filled"),
        accord_auth_disabled=True,
        harness_max_refines=1,
    )
    app = build_intake_app(s, engine=main, refiner_engine=refiner)
    scn = Scenario(
        id="rescue",
        name="Rescue",
        turns=("we are Acme",),
        expected={"business.business_name": "Acme Trucking"},
    )
    result = await run_scenario(app, scn)
    assert result.turn_stats.refiner_rescued == 1
    assert result.turn_stats.first_pass_passed == 0
    assert result.turn_stats.still_failing == 0


@pytest.mark.asyncio
async def test_run_scenario_absent_violation_reported(tmp_path):
    """absent: [business.tax_id] should flag as violated when extraction
    sets EIN against the scenario's assertion that it should stay empty."""
    engine = FakeEngine([
        valid_ca_dict(),   # sets ein="12-3456789" — violates absent assertion
        "resp",
    ])
    app = _app(tmp_path, engine)
    scn = Scenario(
        id="absent",
        name="Absent",
        turns=("we are Acme",),
        expected={"business.business_name": "Acme Trucking"},
        absent=("business.tax_id",),
    )
    result = await run_scenario(app, scn)
    assert "business.tax_id" in result.absent_violations


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_all_aggregates_precision_and_turn_stats(tmp_path):
    # Shared engine queues responses for both scenarios sequentially.
    # concurrency=1 so scenarios run serially and the queue order is
    # deterministic.
    engine = FakeEngine([
        # s1: one turn
        valid_ca_dict(),   "resp-1",
        # s2: one turn
        valid_ca_dict(),   "resp-2",
    ])
    app = _app(tmp_path, engine)
    scenarios = [
        Scenario(
            id="s1", name="One",
            turns=("a",),
            expected={"business.business_name": "Acme Trucking"},
        ),
        Scenario(
            id="s2", name="Two",
            turns=("b",),
            expected={"business.business_name": "Acme Trucking"},
        ),
    ]
    report = await run_all(app, scenarios, concurrency=1)
    assert isinstance(report, RunReport)
    assert len(report.scenarios) == 2
    assert report.aggregate_precision == 1.0
    assert report.aggregate_recall == 1.0
    assert report.total_turns == 2
    assert report.first_pass_passed == 2
    assert report.refiner_rescued == 0


def test_runreport_to_dict_shape(tmp_path):
    """Report dict is serializable + includes the derived rates."""
    from accord_ai.eval.runner import TurnStats
    from accord_ai.eval.types import ScoreResult
    zero_score = ScoreResult(
        scenario_id="x",
        total_expected=0, translated=0, matched=0,
        precision=0.0, recall=0.0, f1=0.0,
        comparisons=(),
        untranslatable_paths=(),
    )
    empty = RunReport(
        scenarios=(ScenarioResult(
            scenario_id="x", name="X", score=zero_score,
            turn_stats=TurnStats(total=0, first_pass_passed=0,
                                 refiner_rescued=0, still_failing=0),
        ),),
        aggregate_precision=0.0, aggregate_recall=0.0, aggregate_f1=0.0,
        total_turns=0, first_pass_passed=0, refiner_rescued=0,
        still_failing=0,
    )
    d = empty.to_dict()
    assert "aggregate" in d
    assert "turn_stats" in d
    assert "scenarios" in d
    assert d["turn_stats"]["first_pass_rate"] == 0.0
