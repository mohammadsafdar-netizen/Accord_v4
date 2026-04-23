"""L3 field-level scorer for v4 submissions against v3-shape expectations.

Usage:
    from accord_ai.eval import score_submission
    result = score_submission(scenario_id, submission, expected_dict)
    print(result.precision, result.recall, result.f1)
    print(result.to_dict())
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List

from accord_ai.eval.path_map import translate
from accord_ai.eval.types import FieldComparison, ScoreResult
from accord_ai.schema import CustomerSubmission


_PATH_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


def _resolve_v4_path(submission_dict: Any, path: str) -> Any:
    """Walk a v4 dotted path with [N] index support. None on any miss."""
    cur = submission_dict
    for raw in path.split("."):
        if cur is None:
            return None
        m = _PATH_SEGMENT.match(raw)
        if not m:
            return None
        attr, idx = m.group(1), m.group(2)
        if isinstance(cur, dict):
            cur = cur.get(attr)
        else:
            cur = getattr(cur, attr, None)
        if idx is not None:
            if cur is None:
                return None
            try:
                cur = cur[int(idx)]
            except (IndexError, TypeError, KeyError):
                return None
    return cur


def _normalize_for_compare(v: Any) -> Any:
    """Reduce both expected and actual to a canonical comparison form.

    Rules:
      None / "" / [] / {} → None (all "empty" — no signal)
      int/float/Decimal   → str (v3 everything-is-string baseline)
      bool                → "true" / "false"
      date/datetime       → ISO date string
      str                 → lowered, stripped; empty → None
      other               → repr(v)
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, Decimal)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip().lower()
        return s or None
    if isinstance(v, (list, dict)) and not v:
        return None
    return repr(v)


def _normalize_v3_expected(v: Any) -> Any:
    """v3 expectations are all strings — lower, strip. Also accept dates
    written as MM/DD/YYYY and convert to ISO so they match a v4 date."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # MM/DD/YYYY → YYYY-MM-DD for date-like strings.
        mdy = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
        if mdy:
            month, day, year = mdy.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return s.lower()
    return _normalize_for_compare(v)


_COUNT_PREFIX = "@count:"


def _score_one(
    sub_dict: Any, v4_path: str, v4_expected: Any,
) -> tuple[bool, str, Any]:
    """Score a single (v4_path, v4_expected) pair.

    Handles:
      * `@count:<path>` — resolves path, compares len(resolved) to expected int
      * `_ANY`  sentinel — matches if actual is present and non-empty
      * `_LIST` sentinel — matches if actual is a non-empty list
      * Plain path — normalized equality, with explicit "missing" vs "mismatch"
        reasons. Empty-expected with a non-None actual is a mismatch (not a
        silent pass, as the prior implementation did).
    """
    # --- @count:path — list-length expectation -----------------------------
    if v4_path.startswith(_COUNT_PREFIX):
        real_path = v4_path[len(_COUNT_PREFIX):]
        resolved = _resolve_v4_path(sub_dict, real_path)
        actual_len = len(resolved) if isinstance(resolved, list) else 0
        try:
            expected_int = int(str(v4_expected).strip())
        except (TypeError, ValueError):
            # Motivating case: `prior_insurance: "_LIST"` in v3 YAMLs —
            # translator emits @count: route, but the expected value is the
            # _LIST sentinel. Fall through to sentinel scoring against the
            # resolved list so "_LIST"/"_ANY" works on count-route keys too.
            if isinstance(v4_expected, str) and v4_expected.strip() in ("_LIST", "_ANY"):
                return _score_sentinel(resolved, v4_expected.strip())
            return (False, "mismatch", resolved)
        if actual_len == expected_int:
            return (True, "ok", actual_len)
        return (False, "mismatch", actual_len)

    # --- Sentinel values on plain paths ------------------------------------
    if isinstance(v4_expected, str) and v4_expected.strip() in ("_ANY", "_LIST"):
        actual = _resolve_v4_path(sub_dict, v4_path)
        return _score_sentinel(actual, v4_expected.strip())

    # --- Normal value comparison -------------------------------------------
    actual = _resolve_v4_path(sub_dict, v4_path)
    exp_norm = _normalize_v3_expected(v4_expected)
    act_norm = _normalize_for_compare(actual)

    if exp_norm is None and act_norm is None:
        return (True, "ok", actual)
    if exp_norm == act_norm:
        return (True, "ok", actual)
    if act_norm is None:
        return (False, "missing", actual)
    if exp_norm is None:
        # Expected empty/None but actual has a value — a real divergence,
        # not a silent pass (was the pre-fix H1 bug).
        return (False, "unexpected_value", actual)
    return (False, "mismatch", actual)


def _score_sentinel(actual: Any, sentinel: str) -> tuple[bool, str, Any]:
    """_ANY = any non-empty value; _LIST = non-empty list."""
    if sentinel == "_LIST":
        if isinstance(actual, list) and len(actual) > 0:
            return (True, "ok", actual)
        return (False, "missing", actual)
    # _ANY
    if actual is None:
        return (False, "missing", actual)
    if isinstance(actual, (str, list, dict)) and len(actual) == 0:
        return (False, "missing", actual)
    return (True, "ok", actual)


def score_submission(
    scenario_id: str,
    submission: CustomerSubmission,
    expected: Dict[str, Any],
) -> ScoreResult:
    """Score a v4 submission against a v3-shape expected dict.

    Returns precision/recall/F1 + per-field comparisons.
    Untranslatable v3 paths are tracked separately so schema gaps are
    visible rather than silently penalizing recall.
    """
    # python mode preserves date/Decimal; JSON mode would stringify them.
    sub_dict = submission.model_dump(mode="python")

    comparisons: List[FieldComparison] = []
    untranslatable: List[str] = []
    translated_count = 0
    matched_count = 0
    # Track recall at the v3-path level: a v3 path counts as "matched"
    # iff ALL its v4 pair expansions matched. Prevents recall>1.0 when
    # a v3 path like `drivers[0].full_name` expands to 2 v4 pairs and
    # both hit (pre-fix the numerator counted pair-matches but the
    # denominator counted v3 paths, so 2/1 = 2.0 recall was possible).
    matched_v3_paths = 0

    for v3_path, v3_value in expected.items():
        pairs = translate(v3_path, v3_value)
        if not pairs:
            untranslatable.append(v3_path)
            comparisons.append(FieldComparison(
                v3_path=v3_path, v4_path=None,
                expected_value=v3_value, actual_value=None,
                matched=False, reason="untranslatable",
            ))
            continue

        all_pairs_matched = True
        for v4_path, v4_expected in pairs:
            translated_count += 1
            matched, reason, actual = _score_one(sub_dict, v4_path, v4_expected)

            if matched:
                matched_count += 1
            else:
                all_pairs_matched = False
            comparisons.append(FieldComparison(
                v3_path=v3_path, v4_path=v4_path,
                expected_value=v4_expected, actual_value=actual,
                matched=matched, reason=reason,
            ))
        if all_pairs_matched:
            matched_v3_paths += 1

    total_expected = len(expected)
    # Precision: over v4 pair translations (what the schema actually
    # asked the extractor to produce). Unchanged.
    precision = (
        (matched_count / translated_count) if translated_count else 0.0
    )
    # Recall: over v3 scenario paths (what the scenario author listed).
    # Uses matched_v3_paths so a multi-pair v3 path counts as one
    # observation in the denominator and one match in the numerator.
    recall = (
        (matched_v3_paths / total_expected) if total_expected else 0.0
    )
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) else 0.0
    )

    return ScoreResult(
        scenario_id=scenario_id,
        total_expected=total_expected,
        translated=translated_count,
        matched=matched_count,
        matched_v3_paths=matched_v3_paths,
        precision=precision,
        recall=recall,
        f1=f1,
        comparisons=tuple(comparisons),
        untranslatable_paths=tuple(untranslatable),
    )
