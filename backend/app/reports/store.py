"""Generated-report cache.

A report is a pure projection of a finished run, so it is built once and reused.
Regenerating the same run returns the cached document unless the caller explicitly
asks for a fresh one (the "Generate again" action).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

_REPORTS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_BY_RUN: dict[str, str] = {}
_MAX_REPORTS = 25


def save(report: dict[str, Any]) -> dict[str, Any]:
    report_id = report["report_id"]
    _REPORTS[report_id] = report
    _BY_RUN[report["run_id"]] = report_id

    while len(_REPORTS) > _MAX_REPORTS:
        old_id, old = _REPORTS.popitem(last=False)
        if _BY_RUN.get(old.get("run_id", "")) == old_id:
            _BY_RUN.pop(old["run_id"], None)
    return report


def get(report_id: str) -> dict[str, Any] | None:
    return _REPORTS.get(report_id)


def get_by_run(run_id: str) -> dict[str, Any] | None:
    report_id = _BY_RUN.get(run_id)
    return _REPORTS.get(report_id) if report_id else None


def drop_run(run_id: str) -> None:
    report_id = _BY_RUN.pop(run_id, None)
    if report_id:
        _REPORTS.pop(report_id, None)


def listing() -> list[dict[str, Any]]:
    return [
        {
            "report_id": r["report_id"],
            "run_id": r["run_id"],
            "tracking_goal": r["tracking_goal"],
            "generated_at": r["generated_at"],
            "total_insights": (r.get("summary_stats") or {}).get("total_insights", 0),
        }
        for r in reversed(_REPORTS.values())
    ]
