"""Accord AI evaluation harness (P10.S.11).

Public surface:
    translate(v3_path, v3_value) -> List[(v4_path, v4_value)]
    score_submission(scenario_id, submission, expected) -> ScoreResult
    load_scenarios(path) -> List[Scenario]
    run_scenario(app, scenario) -> ScenarioResult
    run_all(app, scenarios) -> RunReport

10.S.11a is the path translator + L3 scorer; 10.S.11b is the scenario
runner that drives the full ConversationController per scenario and
rolls per-scenario L3 results into an aggregate report with per-stage
(first-pass vs refiner-rescue) turn counts.
"""
from accord_ai.eval.path_map import translate
from accord_ai.eval.runner import (
    RunReport,
    Scenario,
    ScenarioResult,
    TurnStats,
    load_scenarios,
    run_all,
    run_scenario,
)
from accord_ai.eval.scorer import score_submission
from accord_ai.eval.types import FieldComparison, ScoreResult

__all__ = [
    "FieldComparison",
    "RunReport",
    "Scenario",
    "ScenarioResult",
    "ScoreResult",
    "TurnStats",
    "load_scenarios",
    "run_all",
    "run_scenario",
    "score_submission",
    "translate",
]
