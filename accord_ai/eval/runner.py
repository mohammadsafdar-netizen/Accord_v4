"""Scenario runner (P10.S.11b).

Loads v3-shaped scenario YAMLs, drives each scenario through v4's
``ConversationController`` turn-by-turn, and scores the final submission
against the scenario's ``expected`` block via
:func:`accord_ai.eval.score_submission` (10.S.11a).

The runner is engine-agnostic: hand in an :class:`IntakeApp` wired with
any :class:`Engine` implementation — :class:`FakeEngine` (no network,
CI-friendly) for deterministic replay, or a real
:class:`OpenAIEngine` pointing at local vLLM for honest accuracy
numbers against a warm guided-json grammar.

Report shape (:class:`RunReport`):
  * aggregate L3 precision / recall / F1 across all scenarios
  * per-stage counts: turns where the first-pass extraction satisfied
    the judge vs. turns where the refiner had to rescue
  * per-scenario detail — each :class:`ScoreResult` plus refined/total
    turn counts — so regressions can be isolated to a specific scenario

Scenario YAML shape (v3):
  scenarios:
    - id:    str
      name:  str
      tags:  [str, ...]
      turns: [str, ...]             # user messages, in order
      expected: {v3_path: value}    # scoring target
      absent:   [v3_path, ...]      # optional — paths that MUST be empty
      # negation_checks / expected_lobs / expected_forms ignored for now

Absent-path scoring: fails when the resolved v4 value at that path is
non-empty. Not every v3 scenario supplies an absent block; scorer
handles both.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from accord_ai.app import IntakeApp
from accord_ai.eval.scorer import score_submission, _resolve_v4_path
from accord_ai.eval.path_map import translate
from accord_ai.eval.types import ScoreResult
from accord_ai.logging_config import get_logger

_logger = get_logger("eval.runner")


# ---------------------------------------------------------------------------
# Scenario + report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One v3 scenario — turns to feed, expected fields to check."""
    id: str
    name: str
    turns: tuple[str, ...]
    expected: Dict[str, Any]
    absent: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnStats:
    """Per-turn observability counts.

    Not all turns produce extraction (some are just acknowledgments or
    "continue" — the extractor returns empty and judge runs on unchanged
    state). ``first_pass_passed`` is therefore "did the turn END with a
    passing verdict without the refiner firing"; ``refiner_rescued`` is
    "turn ended passing BUT the refiner had to run".
    """
    total: int
    first_pass_passed: int
    refiner_rescued: int
    still_failing: int


@dataclass(frozen=True)
class ScenarioResult:
    """Per-scenario report."""
    scenario_id: str
    name: str
    score: ScoreResult
    turn_stats: TurnStats
    absent_violations: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id":       self.scenario_id,
            "name":              self.name,
            "precision":         self.score.precision,
            "recall":            self.score.recall,
            "f1":                self.score.f1,
            "matched":           self.score.matched,
            "translated":        self.score.translated,
            "total_expected":    self.score.total_expected,
            "untranslatable":    list(self.score.untranslatable_paths),
            "absent_violations": list(self.absent_violations),
            "turns": {
                "total":              self.turn_stats.total,
                "first_pass_passed":  self.turn_stats.first_pass_passed,
                "refiner_rescued":    self.turn_stats.refiner_rescued,
                "still_failing":      self.turn_stats.still_failing,
            },
        }


@dataclass(frozen=True)
class RunReport:
    """Aggregate report over every scenario that ran.

    The aggregate precision/recall/F1 are micro-averaged: one scorer
    summary computed across the union of every scenario's field matches
    and translations. Macro-average (per-scenario mean) is derivable
    from ``scenarios[i].score`` if needed.
    """
    scenarios: tuple[ScenarioResult, ...]
    aggregate_precision: float
    aggregate_recall:    float
    aggregate_f1:        float
    total_turns:         int
    first_pass_passed:   int
    refiner_rescued:     int
    still_failing:       int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate": {
                "precision": self.aggregate_precision,
                "recall":    self.aggregate_recall,
                "f1":        self.aggregate_f1,
                "scenarios_run": len(self.scenarios),
            },
            "turn_stats": {
                "total":              self.total_turns,
                "first_pass_passed":  self.first_pass_passed,
                "refiner_rescued":    self.refiner_rescued,
                "still_failing":      self.still_failing,
                "first_pass_rate":    (
                    self.first_pass_passed / self.total_turns
                    if self.total_turns else 0.0
                ),
                "rescue_rate": (
                    self.refiner_rescued / (self.total_turns - self.first_pass_passed)
                    if (self.total_turns - self.first_pass_passed) else 0.0
                ),
                "final_pass_rate": (
                    (self.first_pass_passed + self.refiner_rescued)
                    / self.total_turns
                    if self.total_turns else 0.0
                ),
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------


def load_scenarios(path: Path) -> List[Scenario]:
    """Load every scenario from a YAML file or a directory of YAMLs.

    YAMLs with ``{scenarios: [...]}`` at root produce one Scenario per
    entry. Missing optional fields fall back to empty tuples / dicts.
    Scenarios without ``turns`` or ``expected`` are skipped with a
    warning — they can't be scored.
    """
    if path.is_dir():
        files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
    else:
        files = [path]

    out: List[Scenario] = []
    for f in files:
        data = yaml.safe_load(f.read_text()) or {}
        raw_scenarios = data.get("scenarios") if isinstance(data, dict) else None
        if not isinstance(raw_scenarios, list):
            # Some v3 YAMLs are single-scenario-at-root (like dual's
            # 01_solo_plumber.yaml). Accept that shape too.
            if isinstance(data, dict) and "turns" in data and "expected" in data:
                raw_scenarios = [{"id": f.stem, "name": f.stem, **data}]
            else:
                _logger.warning("skipping %s — no scenarios block", f)
                continue

        for raw in raw_scenarios:
            sid = str(raw.get("id") or raw.get("name") or f.stem)
            turns = raw.get("turns") or []
            expected = raw.get("expected") or {}
            if not turns or not expected:
                _logger.warning(
                    "skipping scenario %s — missing turns or expected", sid,
                )
                continue
            out.append(Scenario(
                id=sid,
                name=str(raw.get("name") or sid),
                turns=tuple(str(t) for t in turns),
                expected=dict(expected),
                absent=tuple(raw.get("absent") or ()),
                tags=tuple(raw.get("tags") or ()),
            ))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _check_absent(submission_dict: Any, absent_paths: Sequence[str]) -> List[str]:
    """Return the subset of absent_paths that are NON-empty in the submission.

    An absent-path violation is: the scenario promised this field would
    not be set, but extraction set it anyway.
    """
    violations: List[str] = []
    for v3_path in absent_paths:
        pairs = translate(v3_path, None)
        for v4_path, _ in pairs:
            if v4_path.startswith("@count:"):
                real = v4_path[len("@count:"):]
                resolved = _resolve_v4_path(submission_dict, real)
                if isinstance(resolved, list) and len(resolved) > 0:
                    violations.append(v3_path)
                    break
            else:
                resolved = _resolve_v4_path(submission_dict, v4_path)
                if resolved not in (None, "", [], {}):
                    violations.append(v3_path)
                    break
    return violations


async def run_scenario(
    app: IntakeApp,
    scenario: Scenario,
    *,
    tenant: Optional[str] = None,
) -> ScenarioResult:
    """Drive one scenario through the full controller stack and score it.

    Each turn is an independent ``process_turn`` call. Between turns the
    session's submission accumulates via ``apply_submission_diff`` exactly
    as in production. After the final turn we load the persisted
    submission and hand it to the L3 scorer along with the scenario's
    expected block.

    Tracking: per turn we check whether the refiner ran and whether the
    final verdict was passing. The four-way bucket (passed-no-refine,
    rescued, refined-still-failing, no-refine-still-failing) covers
    every possible outcome and aggregates cleanly across scenarios.
    """
    sid = app.store.create_session(tenant=tenant)

    first_pass = 0
    rescued = 0
    still_failing = 0

    for turn_idx, user_msg in enumerate(scenario.turns):
        result = await app.controller.process_turn(
            session_id=sid, user_message=user_msg, tenant=tenant,
        )
        passed = result.verdict.passed
        refined = result.refined

        if passed and not refined:
            first_pass += 1
        elif passed and refined:
            rescued += 1
        else:
            still_failing += 1

        _logger.debug(
            "scenario=%s turn=%d passed=%s refined=%s",
            scenario.id, turn_idx, passed, refined,
        )

    session = app.store.get_session(sid, tenant=tenant)
    submission = session.submission

    score = score_submission(
        scenario.id, submission, scenario.expected,
    )

    submission_dict = submission.model_dump(mode="python")
    absent_viol = _check_absent(submission_dict, scenario.absent)

    turn_stats = TurnStats(
        total=len(scenario.turns),
        first_pass_passed=first_pass,
        refiner_rescued=rescued,
        still_failing=still_failing,
    )
    return ScenarioResult(
        scenario_id=scenario.id,
        name=scenario.name,
        score=score,
        turn_stats=turn_stats,
        absent_violations=tuple(absent_viol),
    )


async def run_all(
    app: IntakeApp,
    scenarios: Sequence[Scenario],
    *,
    tenant: Optional[str] = None,
    concurrency: int = 1,
) -> RunReport:
    """Run every scenario and aggregate.

    ``concurrency`` bounds in-flight scenarios via a semaphore. Sessions
    are tenant-isolated so cross-scenario state bleed is impossible, but
    the LLM backend is shared — set concurrency=1 for real vLLM runs on
    a single GPU (the server already serializes) and higher for
    FakeEngine CI runs.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[ScenarioResult]] = [None] * len(scenarios)

    async def _run_at(i: int, scn: Scenario) -> None:
        async with sem:
            results[i] = await run_scenario(app, scn, tenant=tenant)

    await asyncio.gather(*(_run_at(i, s) for i, s in enumerate(scenarios)))
    concrete: List[ScenarioResult] = [r for r in results if r is not None]

    # Aggregate L3:
    #   precision = sum(v4-pair matches) / sum(v4-pair translations)
    #   recall    = sum(matched v3 paths) / sum(v3 expected paths)
    # Using matched_v3_paths for the recall numerator caps per-scenario
    # recall at 1.0 even when a v3 path like drivers[0].full_name expands
    # to 2 v4 pairs (earlier scorer bug produced recall > 1.0).
    total_matched = sum(r.score.matched for r in concrete)
    total_translated = sum(r.score.translated for r in concrete)
    total_expected = sum(r.score.total_expected for r in concrete)
    total_matched_v3 = sum(r.score.matched_v3_paths for r in concrete)
    precision = total_matched / total_translated if total_translated else 0.0
    recall = total_matched_v3 / total_expected if total_expected else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) else 0.0
    )

    total_turns = sum(r.turn_stats.total for r in concrete)
    return RunReport(
        scenarios=tuple(concrete),
        aggregate_precision=precision,
        aggregate_recall=recall,
        aggregate_f1=f1,
        total_turns=total_turns,
        first_pass_passed=sum(r.turn_stats.first_pass_passed for r in concrete),
        refiner_rescued=sum(r.turn_stats.refiner_rescued for r in concrete),
        still_failing=sum(r.turn_stats.still_failing for r in concrete),
    )
