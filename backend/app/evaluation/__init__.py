"""Evaluation, benchmarking and quality measurement (Task 6).

This layer measures how good the agent's *performance* was. It is deliberately
separate from the Task 5 self-evaluator, which answers a different question:

  * Task 5 evaluator  → "what should the agent do next?"   (online, steers routing)
  * Task 6 evaluation → "how good was that performance?"   (offline, scores quality)

Every number produced here comes from a real InsightPulse execution. Nothing is
hardcoded, and where a metric genuinely cannot be measured from the available data
it reports `available=False` with a reason rather than inventing a score.
"""

from __future__ import annotations

from .schemas import (
    SCENARIO_TYPES,
    CaseOutcome,
    EvaluationCase,
    ScenarioType,
    Thresholds,
)

__all__ = [
    "SCENARIO_TYPES",
    "CaseOutcome",
    "EvaluationCase",
    "ScenarioType",
    "Thresholds",
]
