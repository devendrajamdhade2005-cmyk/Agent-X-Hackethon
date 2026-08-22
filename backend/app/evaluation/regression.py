"""Regression comparison between consecutive evaluation suites.

Negative changes are reported as prominently as positive ones (section 44). Direction
is taken from the metric catalogue, so a *drop* in hallucination is correctly shown
as an improvement while a drop in groundedness is a regression.
"""

from __future__ import annotations

from typing import Any

from . import metrics as M
from . import store

# A change smaller than this is noise, not a regression.
NOISE_FLOOR = 0.01
LATENCY_NOISE_FLOOR_MS = 250.0


def compare_with_previous(suite_id: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    """Compare this suite's aggregate against the previous stored suite."""
    previous = store.previous_full_suite(exclude_suite_id=suite_id)
    if previous is None:
        return {
            "compared": False,
            "reason": "no previous evaluation suite is stored, so there is no baseline to compare against",
            "changes": [],
            "regressions": [],
            "improvements": [],
        }

    prev_aggregate = previous.get("aggregate") or {}
    changes: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    names = [n for n in aggregate if n in M.CATALOGUE or n == "robustness"]
    for name in names:
        current = _value(aggregate.get(name))
        before = _value(prev_aggregate.get(name))
        if current is None or before is None:
            changes.append({
                "metric": name,
                "previous": before,
                "current": current,
                "delta": None,
                "direction": "unmeasurable",
                "note": "metric unavailable in one of the two suites",
            })
            continue

        higher_better = (
            M.CATALOGUE[name].higher_is_better if name in M.CATALOGUE else True
        )
        unit = M.CATALOGUE[name].unit if name in M.CATALOGUE else "ratio"
        delta = current - before
        floor = LATENCY_NOISE_FLOOR_MS if unit == "ms" else NOISE_FLOOR

        if abs(delta) < floor:
            direction = "unchanged"
        else:
            gain = delta if higher_better else -delta
            direction = "improved" if gain > 0 else "regressed"

        entry = {
            "metric": name,
            "label": M.CATALOGUE[name].label if name in M.CATALOGUE else name.title(),
            "previous": round(before, 4),
            "current": round(current, 4),
            "delta": round(delta, 4),
            "unit": unit,
            "higher_is_better": higher_better,
            "direction": direction,
        }
        changes.append(entry)
        if direction == "regressed":
            regressions.append(entry)
        elif direction == "improved":
            improvements.append(entry)

    overall_before = _value(prev_aggregate.get("overall_score")) or prev_aggregate.get("overall_score")
    overall_now = aggregate.get("overall_score")
    overall_delta = None
    if isinstance(overall_before, (int, float)) and isinstance(overall_now, (int, float)):
        overall_delta = round(float(overall_now) - float(overall_before), 4)

    return {
        "compared": True,
        "previous_suite_id": previous.get("suite_id"),
        "previous_completed_at": previous.get("completed_at"),
        "overall_previous": overall_before if isinstance(overall_before, (int, float)) else None,
        "overall_current": overall_now,
        "overall_delta": overall_delta,
        "changes": changes,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }


def _value(entry: Any) -> float | None:
    """Metric entries are dicts; the overall score is a bare number."""
    if isinstance(entry, dict):
        if not entry.get("available"):
            return None
        v = entry.get("value")
        return float(v) if isinstance(v, (int, float)) else None
    return float(entry) if isinstance(entry, (int, float)) else None
