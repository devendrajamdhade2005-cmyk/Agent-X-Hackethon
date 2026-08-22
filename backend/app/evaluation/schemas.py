"""Evaluation data contracts.

Plain dataclasses with `to_dict()`, matching the project's existing style (no ORM,
no new database). Everything a judge or a test needs to reproduce a result is
recorded on the objects themselves: the case, the scenario, the thresholds that
decided PASS/PARTIAL/FAIL, and the provenance of the run that was measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# ── scenario taxonomy (section 6) ────────────────────────────
ScenarioType = Literal[
    "NORMAL",
    "AMBIGUOUS",
    "ADVERSARIAL",
    "CONTRADICTORY",
    "INCOMPLETE",
    "TOOL_FAILURE",
    "UNSUPPORTED_CONCLUSION",
]

SCENARIO_TYPES: tuple[str, ...] = (
    "NORMAL",
    "AMBIGUOUS",
    "ADVERSARIAL",
    "CONTRADICTORY",
    "INCOMPLETE",
    "TOOL_FAILURE",
    "UNSUPPORTED_CONCLUSION",
)

# Outcome of one evaluated case.
CaseOutcome = Literal["PASS", "PARTIAL", "FAIL", "ERROR"]

# Claim classification (sections 14/15).
CLAIM_SUPPORTED = "SUPPORTED"
CLAIM_PARTIAL = "PARTIALLY_SUPPORTED"
CLAIM_UNSUPPORTED = "UNSUPPORTED"
CLAIM_INFERRED = "INFERRED"
CLAIM_UNCERTAIN = "UNCERTAIN"

# Claim kinds. Only FACTUAL claims are scored for groundedness/hallucination:
# recommendations and explicit hypotheses are not factual assertions (section 15).
KIND_FACTUAL = "FACTUAL"
KIND_INTERPRETIVE = "INTERPRETIVE"
KIND_RECOMMENDATION = "RECOMMENDATION"
KIND_HYPOTHESIS = "HYPOTHESIS"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────
# Thresholds / quality gates (sections 47-49)
# ─────────────────────────────────────────────────────────────
@dataclass
class Thresholds:
    """Internal evaluation thresholds — configurable, not hackathon rules.

    `critical_*` values are hard gates: breaching one fails the case outright, so a
    single excellent metric can never mask a critical failure (section 48).
    """

    min_groundedness: float = 0.70
    max_hallucination_rate: float = 0.20
    min_task_completion: float = 0.70
    min_recovery_rate: float = 0.80
    min_evidence_quality: float = 0.50
    max_latency_ms: int = 60_000

    # Hard gates.
    critical_hallucination_rate: float = 0.35
    critical_groundedness: float = 0.50

    # PASS needs every soft gate met; PARTIAL tolerates soft misses but no
    # critical breach.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Thresholds":
        base = cls()
        for key, value in (data or {}).items():
            if hasattr(base, key) and isinstance(value, (int, float)):
                setattr(base, key, value)
        return base


# ─────────────────────────────────────────────────────────────
# Evaluation case (section 6)
# ─────────────────────────────────────────────────────────────
@dataclass
class EvaluationCase:
    case_id: str
    name: str
    scenario_type: ScenarioType
    user_goal: str
    description: str = ""

    keywords: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # What a correct run looks like, in prose (shown in the UI and the report).
    expected_behavior: str = ""

    # Ground truth. Deliberately modest: only things that are actually checkable
    # from the agent's own output and the deterministic simulation fixtures.
    expected_entities: list[str] = field(default_factory=list)
    expected_subtasks: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    # Behavioural expectations for the harder scenarios.
    allowed_uncertainty: bool = False      # uncertainty is acceptable here
    expects_uncertainty: bool = False      # uncertainty is *required* here
    expected_refusal: bool = False         # must not assert the requested conclusion
    expected_recovery: bool = False        # must recover from injected failure
    expects_conflict_detection: bool = False

    # Deterministic fault injection, passed straight to the Task 5 adversarial mode.
    failure_injections: dict[str, Any] = field(default_factory=dict)

    difficulty: Literal["easy", "medium", "hard"] = "medium"
    repeat_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Claim / evidence records
# ─────────────────────────────────────────────────────────────
@dataclass
class ClaimRecord:
    """One assertion pulled out of the agent's final output, with its verdict."""

    claim_id: str
    text: str
    kind: str                    # KIND_*
    source_field: str            # which output field it came from
    finding_id: str = ""         # evidence link, when the output carries one
    verdict: str = CLAIM_UNSUPPORTED
    evidence_overlap: float = 0.0
    evidence_provider: str = ""
    evidence_credibility: str = ""
    hedged: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Metric result
# ─────────────────────────────────────────────────────────────
@dataclass
class MetricResult:
    """A single measured metric, self-describing so the UI/report need no lookup."""

    name: str
    value: float | None
    unit: str
    definition: str
    method: str
    scope: str                   # run | case | suite
    available: bool = True
    unavailable_reason: str = ""
    higher_is_better: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": (round(self.value, 4) if isinstance(self.value, float) else self.value),
            "unit": self.unit,
            "definition": self.definition,
            "method": self.method,
            "scope": self.scope,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "higher_is_better": self.higher_is_better,
            "details": self.details,
            "notes": self.notes,
        }

    @classmethod
    def unavailable(
        cls, name: str, reason: str, *, unit: str = "ratio", definition: str = "",
        method: str = "", scope: str = "run", higher_is_better: bool = True,
    ) -> "MetricResult":
        """Honest 'not measurable' result — never a fabricated zero."""
        return cls(
            name=name, value=None, unit=unit, definition=definition, method=method,
            scope=scope, available=False, unavailable_reason=reason,
            higher_is_better=higher_is_better,
        )


# ─────────────────────────────────────────────────────────────
# One evaluated execution
# ─────────────────────────────────────────────────────────────
@dataclass
class EvaluationRun:
    """One (case × repeat × system) execution and its measurement."""

    evaluation_run_id: str
    case_id: str
    case_name: str
    scenario_type: str
    system: str = "insightpulse"      # insightpulse | baseline_llm | baseline_pipeline
    agent_run_id: str = ""
    repeat_index: int = 0

    started_at: str = field(default_factory=_now)
    completed_at: str = ""
    status: str = "running"          # running | completed | failed
    outcome: CaseOutcome = "PARTIAL"

    metrics: dict[str, Any] = field(default_factory=dict)     # name -> MetricResult dict
    claims: list[dict[str, Any]] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    outcome_reasons: list[str] = field(default_factory=list)

    # Provenance / reproducibility (sections 57-58).
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def score(self, name: str) -> float | None:
        m = self.metrics.get(name)
        if not isinstance(m, dict) or not m.get("available"):
            return None
        v = m.get("value")
        return float(v) if isinstance(v, (int, float)) else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Human review (sections 31-32)
# ─────────────────────────────────────────────────────────────
HUMAN_SCORE_FIELDS = (
    "accuracy_score",
    "completion_score",
    "evidence_score",
    "groundedness_score",
    "uncertainty_score",
    "actionability_score",
    "overall_score",
)


@dataclass
class HumanEvaluation:
    evaluation_run_id: str
    reviewer_id: str
    accuracy_score: int = 3
    completion_score: int = 3
    evidence_score: int = 3
    groundedness_score: int = 3
    uncertainty_score: int = 3
    actionability_score: int = 3
    overall_score: int = 3
    decision: Literal["PASS", "PARTIAL", "FAIL"] = "PARTIAL"
    comment: str = ""
    created_at: str = field(default_factory=_now)

    def mean_score(self) -> float:
        vals = [getattr(self, f) for f in HUMAN_SCORE_FIELDS]
        return round(sum(vals) / len(vals), 3)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["mean_score"] = self.mean_score()
        return out


# ─────────────────────────────────────────────────────────────
# Suite result
# ─────────────────────────────────────────────────────────────
@dataclass
class SuiteResult:
    """A whole evaluation suite: aggregated metrics, scenario matrix, baseline."""

    suite_id: str
    started_at: str = field(default_factory=_now)
    completed_at: str = ""
    status: str = "running"
    mode: str = "suite"

    runs: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    scenario_matrix: dict[str, Any] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
