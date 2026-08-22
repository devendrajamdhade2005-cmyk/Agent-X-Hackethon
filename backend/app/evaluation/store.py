"""Evaluation persistence + regression history.

Follows the project's existing pattern exactly (module-global `OrderedDict`, plain
dicts, capped size) rather than introducing a database. Suite results are also
mirrored to a JSON file under `DATA_DIR` so history survives a process restart where
the filesystem is durable — and degrades silently to memory-only where it is not
(Render's free tier has no persistent disk). Persistence failure never raises.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from ..config import DATA_DIR

_MAX_SUITES = 25
_MAX_HUMAN = 500

_SUITES: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_RUNS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_HUMAN: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()

_HISTORY_FILE = DATA_DIR / "evaluation_history.json"

# Non-empty when disk persistence is unavailable; surfaced honestly in the API.
degraded: str = ""
_loaded = False


# ─────────────────────────────────────────────────────────────
# disk mirror
# ─────────────────────────────────────────────────────────────
def _persist() -> None:
    global degraded
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = [_slim(s) for s in _SUITES.values()]
        _HISTORY_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        degraded = ""
    except Exception as exc:  # noqa: BLE001 — never break an evaluation on storage
        degraded = f"history not persisted to disk ({type(exc).__name__})"


def _slim(suite: dict[str, Any]) -> dict[str, Any]:
    """History entry: aggregates and counts, without the full per-run payloads."""
    return {
        "suite_id": suite.get("suite_id"),
        "started_at": suite.get("started_at"),
        "completed_at": suite.get("completed_at"),
        "status": suite.get("status"),
        "mode": suite.get("mode"),
        "aggregate": suite.get("aggregate") or {},
        "counts": suite.get("counts") or {},
        "scenario_matrix": suite.get("scenario_matrix") or {},
        "baseline_comparison": suite.get("baseline_comparison") or {},
        # Per-case repeated-run detail is small and worth keeping: without it the
        # reliability/consistency breakdown (which case, how many runs) is lost on
        # restart even though the aggregate figure survives.
        "reliability": suite.get("reliability") or {},
        "consistency": suite.get("consistency") or {},
        "regression": suite.get("regression") or {},
        "thresholds": suite.get("thresholds") or {},
        "provenance": suite.get("provenance") or {},
    }


def load_history() -> list[dict[str, Any]]:
    """Read the persisted history once per process. Never raises."""
    global _loaded, degraded
    if _loaded:
        return [_slim(s) for s in reversed(_SUITES.values())]
    _loaded = True
    try:
        if _HISTORY_FILE.is_file():
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for entry in data:
                    sid = entry.get("suite_id")
                    if sid and sid not in _SUITES:
                        _SUITES[sid] = entry
    except Exception as exc:  # noqa: BLE001
        degraded = f"history could not be read ({type(exc).__name__})"
    return [_slim(s) for s in reversed(_SUITES.values())]


# ─────────────────────────────────────────────────────────────
# suites
# ─────────────────────────────────────────────────────────────
def save_suite(suite: dict[str, Any]) -> dict[str, Any]:
    load_history()
    suite_id = suite.get("suite_id")
    if not suite_id:
        return suite
    _SUITES[suite_id] = suite
    for run in suite.get("runs") or []:
        rid = run.get("evaluation_run_id")
        if rid:
            _RUNS[rid] = run
    while len(_SUITES) > _MAX_SUITES:
        _SUITES.popitem(last=False)
    _persist()
    return suite


def get_suite(suite_id: str) -> dict[str, Any] | None:
    load_history()
    return _SUITES.get(suite_id)


def latest_suite() -> dict[str, Any] | None:
    """Most recent suite with measured results.

    Prefers a suite that still has its full per-run payloads (one that ran in this
    process), but falls back to the newest suite carrying an aggregate. That matters
    after a restart: the persisted history is deliberately slim, and the dashboard
    should still show the last measured metrics rather than claiming no data exists.
    Endpoints that need per-run detail handle its absence on their own.
    """
    load_history()
    for suite in reversed(_SUITES.values()):
        if suite.get("runs"):
            return suite
    for suite in reversed(_SUITES.values()):
        if suite.get("aggregate"):
            return suite
    return None


def previous_full_suite(exclude_suite_id: str = "") -> dict[str, Any] | None:
    """The prior complete suite, used for regression comparison."""
    load_history()
    for suite in reversed(_SUITES.values()):
        if suite.get("suite_id") == exclude_suite_id:
            continue
        if suite.get("aggregate"):
            return suite
    return None


def history() -> list[dict[str, Any]]:
    """Newest-first slim history for the dashboard."""
    load_history()
    return [_slim(s) for s in reversed(_SUITES.values())]


def suite_count() -> int:
    load_history()
    return len(_SUITES)


# ─────────────────────────────────────────────────────────────
# individual evaluation runs
# ─────────────────────────────────────────────────────────────
def get_run(evaluation_run_id: str) -> dict[str, Any] | None:
    return _RUNS.get(evaluation_run_id)


def list_runs() -> list[dict[str, Any]]:
    return list(reversed(_RUNS.values()))


# ─────────────────────────────────────────────────────────────
# human reviews
# ─────────────────────────────────────────────────────────────
def add_human_review(review: dict[str, Any]) -> dict[str, Any]:
    rid = review.get("evaluation_run_id") or ""
    if not rid:
        return review
    bucket = _HUMAN.setdefault(rid, [])
    # One review per reviewer per run: a resubmission replaces the earlier score.
    reviewer = review.get("reviewer_id")
    for idx, existing in enumerate(bucket):
        if existing.get("reviewer_id") == reviewer:
            bucket[idx] = review
            break
    else:
        bucket.append(review)
    while len(_HUMAN) > _MAX_HUMAN:
        _HUMAN.popitem(last=False)
    return review


def human_reviews(evaluation_run_id: str) -> list[dict[str, Any]]:
    return list(_HUMAN.get(evaluation_run_id) or [])


def all_human_reviews() -> dict[str, list[dict[str, Any]]]:
    return {k: list(v) for k, v in _HUMAN.items()}


def human_review_count() -> int:
    return sum(len(v) for v in _HUMAN.values())


# ─────────────────────────────────────────────────────────────
# test / demo support
# ─────────────────────────────────────────────────────────────
def reset() -> None:
    """Clear in-memory state. Used by tests, which share a process."""
    global _loaded, degraded
    _SUITES.clear()
    _RUNS.clear()
    _HUMAN.clear()
    _loaded = True          # do not re-read disk into a test's clean state
    degraded = ""


def status() -> dict[str, Any]:
    load_history()
    return {
        "suites_stored": len(_SUITES),
        "runs_stored": len(_RUNS),
        "human_reviews": human_review_count(),
        "history_file": str(_HISTORY_FILE),
        "degraded": degraded,
    }
