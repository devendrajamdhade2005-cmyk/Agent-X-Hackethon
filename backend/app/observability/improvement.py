"""Improvement engine — diagnosis in, bounded policy change out, verdict at the end.

This is the half of Task 7 that makes tracing worth doing. A diagnosis is translated
into an `ImprovementPlan` drawn from the controlled policy registry, applied as a new
*version* of runtime configuration, and then judged by re-running the same scenario
and comparing measured outcomes.

Three rules shape the design:

  * **Never rewrite code.** Only `OptimizationPolicy` fields change, and only within
    the bounds declared in `policy.BOUNDS`.
  * **Never claim improvement because one number moved.** Acceptance requires the
    targeted metric to improve *and* quality not to regress, with Task 6 supplying
    the quality verdict. An improvement that trades correctness for speed is
    rejected and reverted.
  * **Always reversible.** Every applied plan can be reverted to the previous
    version, and the rejection path does that automatically.
"""

from __future__ import annotations

from typing import Any

from .policy import registry as policy_registry
from .schemas import Diagnosis, ImprovementPlan, new_id, now_iso

# How much a targeted metric must move before we call it a real change rather than
# noise. Latency in particular jitters between runs.
MIN_LATENCY_GAIN_MS = 50
MIN_RELATIVE_GAIN = 0.05

# Quality guardrails for acceptance (section 28).
MAX_QUALITY_DROP = 0.05          # 5 points of a 0–1 quality score
MAX_GROUNDEDNESS_DROP = 0.05
MAX_TASK_COMPLETION_DROP = 0.05


class ImprovementEngine:
    """Proposes, applies, validates and reverts bounded policy improvements."""

    def __init__(self) -> None:
        self.plans: dict[str, ImprovementPlan] = {}

    # ── propose (section 23) ────────────────────────────────
    def propose(self, diagnosis: Diagnosis) -> ImprovementPlan:
        """Translate a diagnosis into a concrete, bounded configuration change."""
        active = policy_registry.active
        component = diagnosis.affected_component or "unknown"
        improvement_type = diagnosis.improvement_type or ""

        plan = ImprovementPlan(
            improvement_id=new_id("imp-"),
            root_cause_type=diagnosis.root_cause_type,
            improvement_type=improvement_type,
            target_component=component,
            optimization_version=active.version + 1,
            previous_version=active.version,
            reason=f"{diagnosis.root_cause_type} diagnosed on {component} "
                   f"(confidence {diagnosis.confidence:.0%})",
        )

        if improvement_type == "RETRY_POLICY":
            current = self._current_attempts(component)
            proposed = 1
            plan.current_configuration = {f"retry_attempts[{component}]": current}
            plan.proposed_configuration = {f"retry_attempts[{component}]": proposed}
            plan.changed_parameter = f"retry_attempts[{component}]"
            plan.expected_benefit = (
                f"Fail over from {component} after 1 attempt instead of {current}, "
                f"removing the retry latency that did not recover the provider while "
                f"the remaining providers still supply evidence."
            )
            plan.risk = (
                "low — the other providers in the same tool cover this source, and a "
                "rate-limited provider was not returning data anyway"
            )
            if current <= proposed:
                plan.status = "rejected"
                plan.expected_benefit = (
                    f"No change available: {component} already retries only {current} "
                    f"time(s), so there is no retry latency left to remove."
                )
        elif improvement_type == "TIMEOUT":
            current = self._current_timeout(component)
            proposed = max(3.0, round(current * 0.5, 1))
            plan.current_configuration = {f"timeout_seconds[{component}]": current}
            plan.proposed_configuration = {f"timeout_seconds[{component}]": proposed}
            plan.changed_parameter = f"timeout_seconds[{component}]"
            plan.expected_benefit = (
                f"Fail fast on a stalled {component} request ({current}s → {proposed}s) "
                f"so the remaining providers are reached sooner."
            )
            plan.risk = "medium — a slow but healthy provider could be cut off early"
        elif improvement_type == "CACHE_DEDUP":
            plan.current_configuration = {"dedup_identical_tool_calls": active.dedup_identical_tool_calls}
            plan.proposed_configuration = {"dedup_identical_tool_calls": True}
            plan.changed_parameter = "dedup_identical_tool_calls"
            plan.expected_benefit = (
                "Serve a repeated identical query from the first result instead of "
                "issuing a second provider call."
            )
            plan.risk = "low — only exact-duplicate queries within one run are affected"
        elif improvement_type == "RESOURCE_POLICY":
            plan.current_configuration = {"note": "resource ceilings are set per run"}
            plan.proposed_configuration = {"note": "operator decision required"}
            plan.expected_benefit = "Raising the ceiling would let the run finish its plan."
            plan.risk = "medium — spends more budget"
            plan.status = "rejected"
            plan.reason += " — resource ceilings are an operator decision, not automated"
        else:
            plan.status = "rejected"
            plan.expected_benefit = "No safe automated policy change applies to this cause."
            plan.risk = "n/a"
            plan.reason += " — no controlled improvement is registered for this cause"

        if diagnosis.uncertain and plan.status != "rejected":
            plan.validation_required = True
            plan.risk += "; diagnosis is uncertain, so validation is mandatory"

        self.plans[plan.improvement_id] = plan
        return plan

    # ── apply (sections 24-25) ──────────────────────────────
    def apply(self, plan: ImprovementPlan) -> ImprovementPlan:
        """Apply the plan as a new policy version. Refuses rejected plans."""
        if plan.status == "rejected":
            return plan
        component = plan.target_component
        if plan.improvement_type == "RETRY_POLICY":
            proposed = int(list(plan.proposed_configuration.values())[0])
            policy_registry.apply(
                retry_attempts_by_source={component: proposed}, reason=plan.reason
            )
        elif plan.improvement_type == "TIMEOUT":
            proposed = float(list(plan.proposed_configuration.values())[0])
            policy_registry.apply(
                timeout_by_source={component: proposed}, reason=plan.reason
            )
        elif plan.improvement_type == "CACHE_DEDUP":
            policy_registry.apply(dedup_identical_tool_calls=True, reason=plan.reason)
        else:
            plan.status = "rejected"
            return plan

        plan.status = "applied"
        plan.applied_at = now_iso()
        plan.optimization_version = policy_registry.version
        self.plans[plan.improvement_id] = plan
        return plan

    def revert(self, plan: ImprovementPlan) -> ImprovementPlan:
        """Roll the policy back. Used on rejection and by the operator."""
        policy_registry.revert()
        plan.status = "reverted"
        plan.optimization_version = policy_registry.version
        self.plans[plan.improvement_id] = plan
        return plan

    # ── helpers ─────────────────────────────────────────────
    def _current_attempts(self, source: str) -> int:
        from ..sources.registry import registry as sources
        from ..sources.resilience import MAX_ATTEMPTS

        connector = sources.get(source)
        # A connector may already declare a lower ceiling than the global default;
        # the diagnosis must report the value actually in force, not the default.
        declared = getattr(connector, "max_attempts", None) if connector else None
        default = int(declared or MAX_ATTEMPTS)
        return policy_registry.active.retry_attempts_for(source, default)

    def _current_timeout(self, source: str) -> float:
        from ..sources.registry import registry as sources

        connector = sources.get(source)
        default = float(getattr(connector, "timeout_seconds", 12.0)) if connector else 12.0
        return policy_registry.active.timeout_for(source, default)


# ─────────────────────────────────────────────────────────────
# Before / after comparison (sections 26-28)
# ─────────────────────────────────────────────────────────────
# Direction per metric. Never inferred from the sign of a difference.
METRIC_DIRECTION: dict[str, bool] = {
    # higher_is_better
    "task_success": True,
    "findings": True,
    "relevant_findings": True,
    "insights": True,
    "accuracy": True,
    "task_completion": True,
    "groundedness": True,
    "recovery_rate": True,
    "evidence_quality": True,
    "uncertainty_handling": True,
    "evaluation_score": True,
    # lower_is_better
    "duration_ms": False,
    "tool_calls": False,
    "provider_calls": False,
    "errors": False,
    "retries": False,
    "fallbacks": False,
    "llm_calls": False,
    "total_tokens": False,
    "estimated_cost": False,
    "hallucination_rate": False,
}

QUALITY_METRICS = (
    "task_completion", "groundedness", "accuracy", "evidence_quality", "evaluation_score",
)


class ComparisonEngine:
    """Compare two runs of the same scenario and decide whether to keep the change."""

    def compare(
        self,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        primary_metric: str = "duration_ms",
        plan: ImprovementPlan | None = None,
        observed_noise: float = 0.0,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        keys = [k for k in METRIC_DIRECTION if k in before or k in after]
        for key in keys:
            b, a = before.get(key), after.get(key)
            if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                rows.append({
                    "metric": key, "before": b, "after": a, "change": None,
                    "direction": "not_measurable", "higher_is_better": METRIC_DIRECTION[key],
                    "note": "not reported by both runs",
                })
                continue
            higher_better = METRIC_DIRECTION[key]
            change = a - b
            gain = (a - b) if higher_better else (b - a)
            rows.append({
                "metric": key,
                "before": round(float(b), 4),
                "after": round(float(a), 4),
                "change": round(float(change), 4),
                "relative_change": (round(change / abs(b), 4) if b else None),
                "higher_is_better": higher_better,
                "direction": (
                    "improved" if gain > 0 else ("regressed" if gain < 0 else "unchanged")
                ),
            })

        verdict = self._verdict(before, after, primary_metric, rows, observed_noise)
        return {
            "primary_metric": primary_metric,
            "noise_floor_used": verdict["floor"],
            "observed_noise": observed_noise,
            "rows": rows,
            "improvement_verified": verdict["accepted"],
            "verdict": verdict["verdict"],
            "reasons": verdict["reasons"],
            "quality_regressions": verdict["quality_regressions"],
            "improvement_id": plan.improvement_id if plan else "",
            "optimization_version": plan.optimization_version if plan else 0,
            "compared_at": now_iso(),
        }

    def _verdict(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        primary: str,
        rows: list[dict[str, Any]],
        observed_noise: float = 0.0,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        regressions: list[str] = []

        # The floor is the larger of the configured minimum and the noise the
        # workload actually showed. A constant alone would let variance be sold as
        # an improvement on a workload noisier than the constant.
        floor = max(float(MIN_LATENCY_GAIN_MS), float(observed_noise or 0.0))

        row = next((r for r in rows if r["metric"] == primary), None)
        if row is None or row.get("change") is None:
            return {
                "accepted": False, "verdict": "NOT_MEASURABLE", "floor": floor,
                "reasons": [f"the target metric '{primary}' was not measured in both runs"],
                "quality_regressions": [],
            }

        higher_better = row["higher_is_better"]
        b, a = float(row["before"]), float(row["after"])
        gain = (a - b) if higher_better else (b - a)
        relative = (gain / abs(b)) if b else 0.0

        # Is the primary movement real, or noise?
        if primary == "duration_ms":
            material = gain >= floor
            if not material:
                reasons.append(
                    f"latency moved by {gain:.0f}ms, which does not clear the "
                    f"{floor:.0f}ms floor"
                    + (f" (the workload itself varied by {observed_noise:.0f}ms "
                       f"across repeats)" if observed_noise else "")
                )
        else:
            material = relative >= MIN_RELATIVE_GAIN
            if not material:
                reasons.append(
                    f"'{primary}' moved {relative:.1%}, below the "
                    f"{MIN_RELATIVE_GAIN:.0%} floor treated as noise"
                )

        # Did quality hold? This is what stops a speed-for-correctness trade.
        for key in QUALITY_METRICS:
            b_q, a_q = before.get(key), after.get(key)
            if isinstance(b_q, (int, float)) and isinstance(a_q, (int, float)):
                drop = float(b_q) - float(a_q)
                limit = (
                    MAX_GROUNDEDNESS_DROP if key == "groundedness"
                    else MAX_TASK_COMPLETION_DROP if key == "task_completion"
                    else MAX_QUALITY_DROP
                )
                if drop > limit:
                    regressions.append(
                        f"{key} fell {drop:.1%} (limit {limit:.0%})"
                    )

        # Critical errors must not increase.
        errs_before = before.get("errors")
        errs_after = after.get("errors")
        error_regression = (
            isinstance(errs_before, (int, float)) and isinstance(errs_after, (int, float))
            and errs_after > errs_before
        )
        if error_regression:
            regressions.append(f"error count rose from {errs_before} to {errs_after}")

        # Task success must not fall.
        succ_before, succ_after = before.get("task_success"), after.get("task_success")
        if (
            isinstance(succ_before, (int, float)) and isinstance(succ_after, (int, float))
            and float(succ_after) < float(succ_before) - MAX_QUALITY_DROP
        ):
            regressions.append(
                f"task success fell from {float(succ_before):.0%} to {float(succ_after):.0%}"
            )

        if regressions:
            return {
                "accepted": False,
                "verdict": "IMPROVEMENT_REJECTED",
                "floor": floor,
                "reasons": [
                    "the change was reverted because quality regressed beyond the "
                    "configured tolerance"
                ] + regressions + reasons,
                "quality_regressions": regressions,
            }
        if not material:
            return {
                "accepted": False,
                "verdict": "NO_MATERIAL_CHANGE",
                "floor": floor,
                "reasons": reasons or ["the target metric did not move materially"],
                "quality_regressions": [],
            }

        reasons.insert(
            0,
            f"{primary} improved by "
            + (f"{gain:.0f}ms" if primary == 'duration_ms' else f"{relative:.1%}")
            + " with no quality regression beyond tolerance"
            + (f", clearing the {floor:.0f}ms noise floor measured for this workload"
               if primary == "duration_ms" and observed_noise else ""),
        )
        return {
            "accepted": True,
            "verdict": "IMPROVEMENT_VERIFIED",
            "floor": floor,
            "reasons": reasons,
            "quality_regressions": [],
        }


engine = ImprovementEngine()
comparison = ComparisonEngine()
