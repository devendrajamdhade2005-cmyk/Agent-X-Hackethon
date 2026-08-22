"""Metric catalogue — the definitions the whole evaluation layer is measured against.

Every metric is declared once, here, with its definition, formula, unit, scope and
direction. The UI, the report and the methodology documentation all read from this
catalogue, so a metric can never be displayed with a different meaning than the one
it was computed under.

Nothing in this module computes anything from live data; it is the specification.
The evaluators in `automated.py` produce `MetricResult`s against these definitions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .schemas import MetricResult

# ─────────────────────────────────────────────────────────────
# Metric names (single source of truth for keys used everywhere)
# ─────────────────────────────────────────────────────────────
ACCURACY = "accuracy"
TASK_COMPLETION = "task_completion"
RELIABILITY = "reliability"
ROBUSTNESS = "robustness"
EVIDENCE_QUALITY = "evidence_quality"
EFFICIENCY = "efficiency"
GROUNDEDNESS = "groundedness"
HALLUCINATION_RATE = "hallucination_rate"
RECOVERY_RATE = "recovery_rate"
CONSISTENCY = "consistency"
LATENCY = "latency"
RESOURCE_EFFICIENCY = "resource_efficiency"
UNCERTAINTY_HANDLING = "uncertainty_handling"
UNSUPPORTED_CONCLUSION_RATE = "unsupported_conclusion_rate"

# Metrics where a lower value is better.
LOWER_IS_BETTER = {HALLUCINATION_RATE, UNSUPPORTED_CONCLUSION_RATE, LATENCY}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    definition: str
    formula: str
    unit: str
    scope: str                 # run | case | suite
    higher_is_better: bool
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "definition": self.definition,
            "formula": self.formula,
            "unit": self.unit,
            "scope": self.scope,
            "higher_is_better": self.higher_is_better,
            "data_source": self.data_source,
        }

    def result(
        self,
        value: float | None,
        *,
        details: dict[str, Any] | None = None,
        notes: str = "",
        method: str = "",
    ) -> MetricResult:
        return MetricResult(
            name=self.name,
            value=value,
            unit=self.unit,
            definition=self.definition,
            method=method or self.formula,
            scope=self.scope,
            available=value is not None,
            unavailable_reason="" if value is not None else "not measurable from this run",
            higher_is_better=self.higher_is_better,
            details=details or {},
            notes=notes,
        )

    def unavailable(self, reason: str) -> MetricResult:
        return MetricResult(
            name=self.name, value=None, unit=self.unit, definition=self.definition,
            method=self.formula, scope=self.scope, available=False,
            unavailable_reason=reason, higher_is_better=self.higher_is_better,
        )


# ─────────────────────────────────────────────────────────────
# The catalogue (section 46 — methodology documentation)
# ─────────────────────────────────────────────────────────────
CATALOGUE: dict[str, MetricSpec] = {
    ACCURACY: MetricSpec(
        name=ACCURACY,
        label="Accuracy",
        definition=(
            "Agreement with the benchmark case's checkable ground truth: the entities "
            "that had to be covered, the source categories that had to be reached, and "
            "any controlled fixture facts. Not a live-world fact check."
        ),
        formula="F1 over (expected_entities ∪ expected_sources ∪ expected_facts) vs. observed",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="case ground truth + run findings/insights",
    ),
    TASK_COMPLETION: MetricSpec(
        name=TASK_COMPLETION,
        label="Task completion",
        definition=(
            "Fraction of the case's required subtasks the run actually performed, "
            "verified from execution evidence rather than an HTTP status."
        ),
        formula="completed_subtasks / required_subtasks",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="case expected_subtasks + run framework/plan/tool/insight evidence",
    ),
    RELIABILITY: MetricSpec(
        name=RELIABILITY,
        label="Reliability",
        definition=(
            "Share of repeated executions of the same case that completed successfully. "
            "Failed and partial repetitions are reported, never hidden."
        ),
        formula="successful_runs / total_runs",
        unit="ratio",
        scope="case",
        higher_is_better=True,
        data_source="repeated run outcomes",
    ),
    ROBUSTNESS: MetricSpec(
        name=ROBUSTNESS,
        label="Robustness",
        definition=(
            "Performance across scenario classes. Aggregated as the unweighted mean of "
            "per-category scores so a strong easy category cannot mask a collapse in a "
            "hard one."
        ),
        formula="mean(per_scenario_category_score)",
        unit="ratio",
        scope="suite",
        higher_is_better=True,
        data_source="per-scenario case outcomes",
    ),
    EVIDENCE_QUALITY: MetricSpec(
        name=EVIDENCE_QUALITY,
        label="Evidence quality",
        definition=(
            "Composite quality of the evidence behind the briefing: source credibility, "
            "relevance, recency, independence (distinct domains) and corroboration."
        ),
        formula=(
            "0.30·credibility + 0.25·relevance + 0.15·recency + 0.15·independence "
            "+ 0.15·corroboration"
        ),
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="run findings (credibility, relevance, dates, providers, corroborated_by)",
    ),
    EFFICIENCY: MetricSpec(
        name=EFFICIENCY,
        label="Efficiency",
        definition=(
            "Useful output per unit of work: relevant findings and insights produced "
            "relative to the tool calls and wall-clock time consumed."
        ),
        formula="normalised(relevant_findings / tool_calls) blended with runtime headroom",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="run metrics (findings_relevant, tool_calls, duration_ms)",
    ),
    GROUNDEDNESS: MetricSpec(
        name=GROUNDEDNESS,
        label="Groundedness",
        definition=(
            "Fraction of factual claims in the final output that are supported by "
            "evidence the run actually collected. Recommendations and explicitly "
            "labelled hypotheses are excluded — they are not factual assertions."
        ),
        formula="(supported + 0.5·partially_supported) / factual_claims",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="insight claims linked to findings by finding_id",
    ),
    HALLUCINATION_RATE: MetricSpec(
        name=HALLUCINATION_RATE,
        label="Hallucination rate",
        definition=(
            "Fraction of factual claims with no sufficient supporting evidence. "
            "Hedged/uncertain statements and labelled hypotheses are not counted."
        ),
        formula="unsupported_factual_claims / factual_claims",
        unit="ratio",
        scope="run",
        higher_is_better=False,
        data_source="insight claims linked to findings by finding_id",
    ),
    RECOVERY_RATE: MetricSpec(
        name=RECOVERY_RATE,
        label="Recovery rate",
        definition=(
            "Share of injected failures the agent genuinely recovered from via retry or "
            "fallback. Returning partial output without attempting recovery does not count."
        ),
        formula="recovered_failures / injected_failures",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="framework tool_errors + fallback_history + injected_events",
    ),
    CONSISTENCY: MetricSpec(
        name=CONSISTENCY,
        label="Consistency",
        definition=(
            "Substantive run-to-run agreement across repetitions — overlap of findings, "
            "agreement of conclusions, priority mix and confidence. Wording is not compared."
        ),
        formula=(
            "0.35·finding_overlap + 0.25·conclusion_agreement + 0.20·priority_agreement "
            "+ 0.10·confidence_agreement + 0.10·completion_agreement"
        ),
        unit="ratio",
        scope="case",
        higher_is_better=True,
        data_source="repeated run findings/insights/metrics",
    ),
    LATENCY: MetricSpec(
        name=LATENCY,
        label="Latency",
        definition="Wall-clock time for the run, plus stage breakdown where instrumented.",
        formula="mean / median / p95 / min / max over samples",
        unit="ms",
        scope="run",
        higher_is_better=False,
        data_source="run metrics duration_ms + framework resource elapsed_ms + tool latencies",
    ),
    RESOURCE_EFFICIENCY: MetricSpec(
        name=RESOURCE_EFFICIENCY,
        label="Resource efficiency",
        definition=(
            "Resource cost of a completed task: tool calls, LLM calls, retries, "
            "fallbacks and estimated spend per successful run."
        ),
        formula="cost_per_completed_task, tool_calls_per_successful_run, llm_calls_per_successful_run",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="framework resource snapshot + tool_executions",
    ),
    UNCERTAINTY_HANDLING: MetricSpec(
        name=UNCERTAINTY_HANDLING,
        label="Uncertainty handling",
        definition=(
            "Whether the agent's expressed certainty matched the strength of its "
            "evidence. Correct uncertainty and correct confidence both pass; confidence "
            "on weak evidence fails, and uncertainty despite strong evidence is flagged "
            "as a calibration issue."
        ),
        formula="calibration match against case expectation and observed evidence strength",
        unit="ratio",
        scope="run",
        higher_is_better=True,
        data_source="framework confidence/uncertainty_flags/hypotheses + evidence quality",
    ),
    UNSUPPORTED_CONCLUSION_RATE: MetricSpec(
        name=UNSUPPORTED_CONCLUSION_RATE,
        label="Unsupported-conclusion rate",
        definition=(
            "How often the agent asserted a conclusion the evidence could not support. "
            "On refusal cases, a correct refusal scores 0 (best)."
        ),
        formula="asserted_unsupported_conclusions / conclusion_opportunities",
        unit="ratio",
        scope="run",
        higher_is_better=False,
        data_source="insight/summary assertion analysis vs. available evidence",
    ),
}


def spec(name: str) -> MetricSpec:
    return CATALOGUE[name]


def catalogue_dicts() -> list[dict[str, Any]]:
    """Methodology payload for the API, dashboard and report."""
    return [s.to_dict() for s in CATALOGUE.values()]


# ─────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────
def distribution(samples: list[float]) -> dict[str, Any]:
    """mean / median / p95 / min / max, with p95 only when the sample supports it."""
    clean = [float(s) for s in samples if isinstance(s, (int, float))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None,
                "p95_note": "no samples"}
    out: dict[str, Any] = {
        "count": len(clean),
        "mean": round(statistics.fmean(clean), 3),
        "median": round(statistics.median(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "p95": None,
        "p95_note": "",
    }
    if len(clean) >= 5:
        ordered = sorted(clean)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        out["p95"] = round(ordered[idx], 3)
    else:
        out["p95_note"] = f"needs >=5 samples, have {len(clean)}"
    return out


def mean_of(values: list[float | None]) -> float | None:
    """Mean of the measurable values only; None when nothing was measurable."""
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(clean), 4) if clean else None
