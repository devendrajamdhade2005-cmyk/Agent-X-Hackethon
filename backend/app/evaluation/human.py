"""Human evaluation workflow.

Reviewers score a completed evaluation run on seven 1–5 dimensions plus a
PASS/PARTIAL/FAIL decision and an optional comment. Multiple reviewers per run are
supported, and the aggregate reports variance rather than presenting one person's
subjective score as objective truth (section 32).

Where both automated and human scores exist for the same dimension, the comparison
surfaces disagreement explicitly instead of smoothing it away (section 56).
"""

from __future__ import annotations

from typing import Any

from . import metrics as M
from . import store
from .schemas import HUMAN_SCORE_FIELDS, HumanEvaluation

# Human 1–5 scores map onto the 0–1 automated scale as (score - 1) / 4.
def to_unit(score: int | float) -> float:
    return round(max(0.0, min(1.0, (float(score) - 1.0) / 4.0)), 4)


# Which automated metric each human dimension is comparable to.
HUMAN_TO_METRIC = {
    "accuracy_score": M.ACCURACY,
    "completion_score": M.TASK_COMPLETION,
    "evidence_score": M.EVIDENCE_QUALITY,
    "groundedness_score": M.GROUNDEDNESS,
    "uncertainty_score": M.UNCERTAINTY_HANDLING,
}

# Gap beyond which automated and human judgement are flagged as disagreeing.
DISAGREEMENT_THRESHOLD = 0.15


def submit(review: HumanEvaluation) -> dict[str, Any]:
    """Persist one reviewer's scores for an evaluation run."""
    stored = store.add_human_review(review.to_dict())
    return {
        "review": stored,
        "aggregate": aggregate(review.evaluation_run_id),
    }


def aggregate(evaluation_run_id: str) -> dict[str, Any]:
    """Reviewer count, per-dimension average and variance."""
    reviews = store.human_reviews(evaluation_run_id)
    if not reviews:
        return {
            "evaluation_run_id": evaluation_run_id,
            "reviewer_count": 0,
            "available": False,
            "reason": "no human review submitted for this run yet",
        }

    per_field: dict[str, Any] = {}
    for field in HUMAN_SCORE_FIELDS:
        values = [float(r.get(field, 0)) for r in reviews if isinstance(r.get(field), (int, float))]
        if not values:
            continue
        mean = M.mean_of(values) or 0.0
        variance = (
            round(sum((v - mean) ** 2 for v in values) / len(values), 4)
            if len(values) > 1 else 0.0
        )
        per_field[field] = {
            "average": round(mean, 3),
            "variance": variance,
            "min": min(values),
            "max": max(values),
            "unit_scale": to_unit(mean),
        }

    decisions: dict[str, int] = {}
    for r in reviews:
        d = str(r.get("decision") or "PARTIAL")
        decisions[d] = decisions.get(d, 0) + 1

    overall = per_field.get("overall_score", {}).get("average")
    return {
        "evaluation_run_id": evaluation_run_id,
        "available": True,
        "reviewer_count": len(reviews),
        "reviewers": [r.get("reviewer_id") for r in reviews],
        "per_dimension": per_field,
        "decisions": decisions,
        "average_overall": overall,
        "score_variance": per_field.get("overall_score", {}).get("variance"),
        "comments": [
            {"reviewer_id": r.get("reviewer_id"), "comment": r.get("comment", "")}
            for r in reviews if r.get("comment")
        ],
    }


def compare_with_automated(
    evaluation_run_id: str, automated_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Automated vs human, per comparable dimension, with disagreement flagged."""
    human = aggregate(evaluation_run_id)
    if not human.get("available"):
        return {
            "available": False,
            "reason": human.get("reason", "no human review available"),
            "automated_only": True,
        }

    rows: list[dict[str, Any]] = []
    disagreements: list[str] = []
    for field, metric_name in HUMAN_TO_METRIC.items():
        h = (human.get("per_dimension") or {}).get(field)
        a = (automated_metrics or {}).get(metric_name) or {}
        h_unit = h.get("unit_scale") if h else None
        a_val = a.get("value") if a.get("available") else None
        row: dict[str, Any] = {
            "dimension": field,
            "metric": metric_name,
            "human_1_5": h.get("average") if h else None,
            "human_normalised": h_unit,
            "automated": a_val,
            "gap": None,
            "disagreement": False,
        }
        if isinstance(h_unit, (int, float)) and isinstance(a_val, (int, float)):
            gap = round(abs(float(a_val) - float(h_unit)), 4)
            row["gap"] = gap
            row["disagreement"] = gap > DISAGREEMENT_THRESHOLD
            if row["disagreement"]:
                disagreements.append(
                    f"{metric_name}: automated {float(a_val):.0%} vs human {float(h_unit):.0%}"
                )
        elif a_val is None:
            row["note"] = str(a.get("unavailable_reason") or "automated metric unavailable")
        rows.append(row)

    # Reported score keeps both visible rather than blending them into one number.
    return {
        "available": True,
        "reviewer_count": human.get("reviewer_count"),
        "rows": rows,
        "disagreements": disagreements,
        "disagreement_detected": bool(disagreements),
        "human_overall_1_5": human.get("average_overall"),
        "human_overall_normalised": (
            to_unit(human["average_overall"]) if isinstance(human.get("average_overall"), (int, float)) else None
        ),
        "note": (
            "Automated and human scores are reported side by side. Disagreement is "
            "surfaced deliberately — it is more informative than a blended figure."
        ),
    }


def pending_and_completed(suite: dict[str, Any] | None) -> dict[str, Any]:
    """Split a suite's runs into those awaiting review and those already reviewed."""
    if not suite:
        return {"pending": [], "completed": [], "note": "no evaluation suite available yet"}

    pending: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for run in suite.get("runs") or []:
        rid = run.get("evaluation_run_id")
        reviews = store.human_reviews(rid) if rid else []
        entry = {
            "evaluation_run_id": rid,
            "case_id": run.get("case_id"),
            "case_name": run.get("case_name"),
            "scenario_type": run.get("scenario_type"),
            "system": run.get("system"),
            "outcome": run.get("outcome"),
            "reviewer_count": len(reviews),
        }
        (completed if reviews else pending).append(entry)
    return {"pending": pending, "completed": completed}
