"""The self-improvement loop — the part that makes tracing consequential.

    TRACE → UNDERSTAND → DIAGNOSE → CHOOSE → APPLY → RE-RUN → MEASURE → VERIFY

Each stage is a real step against the live runtime, not a narration of one:

  * **Trace** runs the actual graph with a deterministic controlled failure armed
    against one provider, keyed to that run's id so nothing leaks into other runs.
  * **Diagnose** reads only what the trace recorded.
  * **Choose / apply** moves a bounded runtime policy value to a new version. No
    source file is written.
  * **Re-run** repeats the *same* scenario, with the same injection configuration,
    so before and after are comparable.
  * **Measure** scores both runs with Task 6's own evaluators, so quality is judged
    by the existing evaluation system rather than by this module's opinion.
  * **Verify** accepts only if the target metric improved *and* no quality metric
    regressed. Otherwise the verdict is `IMPROVEMENT_REJECTED` and the policy is
    rolled back automatically.

The loop reports what it measured, including when a stage could not be measured.
A stage that cannot be measured is reported as such rather than assumed to pass.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from . import controlled_failure as cf
from .analyzer import RootCauseAnalyzer, TraceAnalyzer
from .improvement import ComparisonEngine, ImprovementEngine
from .policy import registry as policy_registry
from .providers import local_provider
from .schemas import now_iso

# The scenario the demo uses by default: one research provider is rate-limited,
# which is the situation the retry policy governs.
DEFAULT_TARGET = "semantic_scholar"
DEFAULT_FAILURE = "rate_limit"
DEFAULT_FAILURE_COUNT = 2
DEFAULT_CASE_ID = "EVAL-001"

# How many times each side of the comparison is run. Two is the minimum that
# produces a measurable spread; the cap keeps a cycle demoable.
DEFAULT_REPEATS = 3
MAX_REPEATS = 5

# Runtime metrics lifted from a run result for comparison. Names match
# `improvement.METRIC_DIRECTION` so direction is always explicit.
_RUNTIME_METRICS = {
    "duration_ms": "duration_ms",
    "tool_calls": "tool_calls",
    "llm_calls": "llm_calls",
    "findings": "findings_total",
    "relevant_findings": "findings_relevant",
    "insights": "insights",
    "estimated_cost": "estimated_cost",
}

# Task 6 metric key → comparison metric name.
_EVAL_METRICS = {
    "accuracy": "accuracy",
    "task_completion": "task_completion",
    "evidence_quality": "evidence_quality",
    "groundedness": "groundedness",
    "hallucination_rate": "hallucination_rate",
    "recovery_rate": "recovery_rate",
    "uncertainty_handling": "uncertainty_handling",
}


class SelfImprovementLoop:
    """Runs the full observe → diagnose → improve → verify cycle."""

    def __init__(self) -> None:
        self.analyzer = TraceAnalyzer()
        self.root_cause = RootCauseAnalyzer()
        self.improver = ImprovementEngine()
        self.comparison = ComparisonEngine()

    # ─────────────────────────────────────────────────────────
    async def execute(
        self,
        *,
        target_source: str = DEFAULT_TARGET,
        failure_type: str = DEFAULT_FAILURE,
        failure_count: int = DEFAULT_FAILURE_COUNT,
        case_id: str = DEFAULT_CASE_ID,
        primary_metric: str = "duration_ms",
        simulation_mode: bool = True,
        validate_with_evaluation: bool = True,
        repeats: int = DEFAULT_REPEATS,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the whole cycle and return every stage's real result.

        `repeats` is why the verdict can be trusted. A single before/after pair
        cannot distinguish a real gain from run-to-run variance — measured at
        roughly 80ms standard deviation on this workload, dominated by the model
        call. Each side is therefore run `repeats` times, compared on medians, and
        the acceptance floor is raised to the *observed* spread rather than a
        constant. One repeat is allowed but reports itself as unqualified.
        """
        case = self._case(case_id)
        cycle_id = f"cyc-{uuid.uuid4().hex[:10]}"
        scenario = f"controlled_failure:{target_source}:{failure_type}"

        def event(name: str, **data: Any) -> None:
            if emit is not None:
                try:
                    emit(name, {"cycle_id": cycle_id, **data})
                except Exception:  # noqa: BLE001 — never let a listener break the loop
                    pass

        report: dict[str, Any] = {
            "cycle_id": cycle_id,
            "scenario": scenario,
            "case_id": case_id,
            "primary_metric": primary_metric,
            "started_at": now_iso(),
            "stages": [],
        }

        # ── 1. TRACE: run with the controlled failure armed ──
        event("cycle_started", scenario=scenario, target=target_source,
              failure_type=failure_type, failure_count=failure_count)
        before_result, before_trace, before_repeats = await self._repeated_runs(
            case, scenario=scenario, label="before", target_source=target_source,
            failure_type=failure_type, failure_count=failure_count,
            simulation_mode=simulation_mode, repeats=repeats,
        )
        report["before_trace_id"] = before_trace.get("trace_id", "")
        report["before_run_id"] = before_result.get("run_id", "")
        event("baseline_traced", trace_id=report["before_trace_id"],
              spans=len(before_trace.get("spans") or []),
              errors=len(before_trace.get("errors") or []))
        report["stages"].append({"stage": "trace", "status": "done",
                                 "detail": f"{len(before_trace.get('spans') or [])} spans, "
                                           f"{len(before_trace.get('errors') or [])} error(s)"})

        # ── 2. UNDERSTAND ──
        analysis = self.analyzer.analyze(before_trace)
        report["analysis"] = analysis
        event("trace_analyzed", errors=analysis["counts"]["errors"],
              wasted_retries=len(analysis["wasted_retries"]))
        report["stages"].append({
            "stage": "understand", "status": "done",
            "detail": f"{analysis['counts']['errors']} error(s), "
                      f"{len(analysis['wasted_retries'])} provider(s) with spent retries",
        })

        # ── 3. DIAGNOSE ──
        diagnosis = self.root_cause.diagnose(before_trace, analysis)
        report["diagnosis"] = diagnosis.to_dict()
        event("root_cause_identified", root_cause=diagnosis.root_cause_type,
              component=diagnosis.affected_component, confidence=diagnosis.confidence)
        report["stages"].append({
            "stage": "diagnose", "status": "done",
            "detail": f"{diagnosis.root_cause_type} on {diagnosis.affected_component} "
                      f"(confidence {diagnosis.confidence:.0%})",
        })

        if diagnosis.root_cause_type in {"UNKNOWN"}:
            report["verdict"] = "NO_DIAGNOSIS"
            report["improvement_verified"] = False
            report["reasons"] = ["the trace showed no actionable inefficiency"]
            report["completed_at"] = now_iso()
            event("cycle_completed", verdict=report["verdict"], verified=False)
            return report

        # ── 4. CHOOSE a bounded improvement ──
        plan = self.improver.propose(diagnosis)
        report["plan"] = plan.to_dict()
        event("improvement_proposed", improvement_type=plan.improvement_type,
              parameter=plan.changed_parameter, status=plan.status)
        if plan.status == "rejected":
            report["verdict"] = "NO_SAFE_IMPROVEMENT"
            report["improvement_verified"] = False
            report["reasons"] = [plan.expected_benefit]
            report["stages"].append({"stage": "choose", "status": "rejected",
                                     "detail": plan.expected_benefit})
            report["completed_at"] = now_iso()
            event("cycle_completed", verdict=report["verdict"], verified=False)
            return report
        report["stages"].append({
            "stage": "choose", "status": "done",
            "detail": f"{plan.changed_parameter}: "
                      f"{list(plan.current_configuration.values())} → "
                      f"{list(plan.proposed_configuration.values())}",
        })

        # ── 5. APPLY (new policy version, reversible) ──
        plan = self.improver.apply(plan)
        report["plan"] = plan.to_dict()
        report["optimization_version"] = policy_registry.version
        event("improvement_applied", version=policy_registry.version,
              parameter=plan.changed_parameter)
        report["stages"].append({
            "stage": "apply", "status": "done",
            "detail": f"runtime policy v{plan.previous_version} → v{policy_registry.version}",
        })

        # ── 6. RE-RUN the same scenario ──
        try:
            after_result, after_trace, after_repeats = await self._repeated_runs(
                case, scenario=scenario, label="after", target_source=target_source,
                failure_type=failure_type, failure_count=failure_count,
                simulation_mode=simulation_mode, repeats=repeats,
            )
        except Exception as exc:  # noqa: BLE001 — a failed re-run must roll back
            self.improver.revert(plan)
            report["plan"] = plan.to_dict()
            report["verdict"] = "RERUN_FAILED"
            report["improvement_verified"] = False
            report["reasons"] = [f"the verification run raised {type(exc).__name__}; "
                                 f"the policy change was reverted"]
            report["completed_at"] = now_iso()
            event("cycle_completed", verdict=report["verdict"], verified=False)
            return report

        report["after_trace_id"] = after_trace.get("trace_id", "")
        report["after_run_id"] = after_result.get("run_id", "")
        report["after_analysis"] = self.analyzer.analyze(after_trace)
        event("rerun_completed", trace_id=report["after_trace_id"])
        report["stages"].append({"stage": "rerun", "status": "done",
                                 "detail": f"same scenario re-run as "
                                           f"{report['after_trace_id']}"})

        # ── 7. MEASURE (runtime + Task 6 evaluation) ──
        before_metrics, before_eval = await self._metrics(
            case, before_result, before_trace, validate=validate_with_evaluation
        )
        after_metrics, after_eval = await self._metrics(
            case, after_result, after_trace, validate=validate_with_evaluation
        )
        report["before_metrics"] = before_metrics
        report["after_metrics"] = after_metrics
        report["evaluation"] = {
            "validated_with_task6": validate_with_evaluation,
            "before": before_eval,
            "after": after_eval,
        }

        # Replace the single-sample latency with the median of the repeats, and
        # measure how much the workload varies on its own. That spread — not a
        # constant — is what the gain has to beat.
        sampling = self._sampling(before_repeats, after_repeats)
        report["sampling"] = sampling
        if sampling["repeats"] > 1:
            before_metrics["duration_ms"] = float(sampling["before_median_ms"])
            after_metrics["duration_ms"] = float(sampling["after_median_ms"])

        event("metrics_collected", validated=validate_with_evaluation,
              repeats=sampling["repeats"], noise_ms=sampling["observed_noise_ms"])
        report["stages"].append({
            "stage": "measure", "status": "done",
            "detail": (
                f"{sampling['repeats']}x per side, median compared; "
                f"observed noise {sampling['observed_noise_ms']}ms"
                + ("; scored by the Task 6 evaluators" if validate_with_evaluation else "")
            ),
        })

        # ── 8. VERIFY, then accept or roll back ──
        verdict = self.comparison.compare(
            before=before_metrics, after=after_metrics,
            primary_metric=primary_metric, plan=plan,
            observed_noise=sampling["observed_noise_ms"],
        )
        verdict["sampling"] = sampling
        report["comparison"] = verdict
        report["verdict"] = verdict["verdict"]
        report["improvement_verified"] = verdict["improvement_verified"]
        report["reasons"] = verdict["reasons"]

        if verdict["improvement_verified"]:
            plan.status = "accepted"
            report["stages"].append({
                "stage": "verify", "status": "accepted",
                "detail": verdict["reasons"][0] if verdict["reasons"] else "verified",
            })
        else:
            # The change did not earn its place. Roll it back so the system is
            # never left running an unverified optimization.
            self.improver.revert(plan)
            report["reverted"] = True
            report["optimization_version"] = policy_registry.version
            report["stages"].append({
                "stage": "verify", "status": "rejected",
                "detail": f"{verdict['verdict']} — policy reverted to "
                          f"v{policy_registry.version}",
            })
        report["plan"] = plan.to_dict()
        report["completed_at"] = now_iso()
        event("cycle_completed", verdict=report["verdict"],
              verified=report["improvement_verified"])
        return report

    # ─────────────────────────────────────────────────────────
    async def _repeated_runs(
        self,
        case: Any,
        *,
        scenario: str,
        label: str,
        target_source: str,
        failure_type: str,
        failure_count: int,
        simulation_mode: bool,
        repeats: int,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Run the scenario `repeats` times and return the median-latency run.

        The median run is the one reported and traced, so the trace shown in the
        dashboard is a real run rather than a synthetic average. The full sample is
        returned so the spread can be used as the acceptance floor.
        """
        repeats = max(1, min(int(repeats), MAX_REPEATS))
        samples: list[dict[str, Any]] = []
        for index in range(repeats):
            result, trace = await self._traced_run(
                case, scenario=scenario, label=f"{label}{index + 1}" if repeats > 1 else label,
                target_source=target_source, failure_type=failure_type,
                failure_count=failure_count, simulation_mode=simulation_mode,
            )
            samples.append({
                "index": index,
                "result": result,
                "trace": trace,
                "duration_ms": int((result.get("metrics") or {}).get("duration_ms") or 0),
                "trace_id": result.get("trace_id", ""),
            })

        ordered = sorted(samples, key=lambda s: s["duration_ms"])
        chosen = ordered[len(ordered) // 2]
        return chosen["result"], chosen["trace"], samples

    async def _traced_run(
        self,
        case: Any,
        *,
        scenario: str,
        label: str,
        target_source: str,
        failure_type: str,
        failure_count: int,
        simulation_mode: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """One graph run with the controlled failure armed against its own run id."""
        from ..graph.runner import run_graph

        run_id = uuid.uuid4().hex[:12]
        # Arming before the run and keying on run_id is what keeps the injection
        # deterministic here and invisible to every other run in the process.
        cf.controller.arm(
            run_id=run_id, target_source=target_source,
            failure_type=failure_type, failure_count=failure_count,
        )
        try:
            result = await run_graph(
                case.user_goal,
                keywords=list(case.keywords),
                competitors=list(case.competitors),
                simulation_mode=simulation_mode,
                scenario=f"{scenario}:{label}",
                run_id=run_id,
                thread_id=f"loop-{label}-{run_id}",
            )
        finally:
            cf.controller.disarm(run_id)

        trace = local_provider.get(str(result.get("trace_id") or "")) or {}
        return result, trace

    def _sampling(
        self, before: list[dict[str, Any]], after: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Summarise the repeats and the noise they revealed."""
        import statistics

        b = sorted(s["duration_ms"] for s in before)
        a = sorted(s["duration_ms"] for s in after)
        repeats = min(len(b), len(a))

        def median(xs: list[int]) -> int:
            return int(statistics.median(xs)) if xs else 0

        # Noise is the larger of the two within-side spreads: if a side varies by
        # 200ms on its own, a 200ms "gain" proves nothing.
        spread_before = (max(b) - min(b)) if len(b) > 1 else 0
        spread_after = (max(a) - min(a)) if len(a) > 1 else 0
        noise = max(spread_before, spread_after)
        return {
            "repeats": repeats,
            "before_samples_ms": b,
            "after_samples_ms": a,
            "before_median_ms": median(b),
            "after_median_ms": median(a),
            "before_spread_ms": spread_before,
            "after_spread_ms": spread_after,
            "observed_noise_ms": noise,
            "qualified": repeats > 1,
            "note": (
                f"each side run {repeats}x; medians compared; the gain must exceed the "
                f"observed {noise}ms spread, not a fixed constant"
                if repeats > 1 else
                "a single run per side — the verdict is not qualified against noise"
            ),
        }

    async def _metrics(
        self, case: Any, result: dict[str, Any], trace: dict[str, Any], *, validate: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Flatten runtime + Task 6 metrics into one comparable dict."""
        runtime = result.get("metrics") or {}
        out: dict[str, Any] = {}
        for name, key in _RUNTIME_METRICS.items():
            value = runtime.get(key)
            if isinstance(value, (int, float)):
                out[name] = float(value)

        analysis = self.analyzer.analyze(trace) if trace else {}
        counts = analysis.get("counts") or {}
        out["errors"] = float(counts.get("errors", 0))
        out["retries"] = float(counts.get("retry_events", 0))
        out["provider_calls"] = float(counts.get("provider_calls", 0))
        tokens = trace.get("token_usage") or {}
        if tokens.get("status") == "measured":
            out["total_tokens"] = float(tokens.get("total_tokens") or 0)

        evaluation: dict[str, Any] = {"available": False}
        if validate:
            evaluation = self._evaluate(case, result)
            for name, key in _EVAL_METRICS.items():
                value = evaluation.get("scores", {}).get(key)
                if isinstance(value, (int, float)):
                    out[name] = float(value)
            if isinstance(evaluation.get("outcome"), str):
                out["task_success"] = 1.0 if evaluation["outcome"] == "PASS" else 0.0
        return out, evaluation

    def _evaluate(self, case: Any, result: dict[str, Any]) -> dict[str, Any]:
        """Score a captured run with Task 6's evaluators.

        Only the *measurement* half of Task 6 is used. Re-executing through
        `evaluate_case` would run the graph a third time and would not carry this
        cycle's controlled-failure configuration, so the comparison would not be
        of the same scenario.
        """
        try:
            from ..evaluation.engine import EvaluationEngine

            engine = EvaluationEngine()
            measured = engine.measure(case, result)
            scores: dict[str, Any] = {}
            unavailable: dict[str, str] = {}
            for name, metric in measured.items():
                data = metric.to_dict()
                if data.get("available") and isinstance(data.get("value"), (int, float)):
                    scores[name] = round(float(data["value"]), 4)
                else:
                    unavailable[name] = str(data.get("reason") or "not measurable")

            # Reuse Task 6's gate logic for the PASS/FAIL verdict.
            from ..evaluation.schemas import EvaluationRun

            eval_run = EvaluationRun(
                evaluation_run_id=f"ev-loop-{uuid.uuid4().hex[:8]}",
                case_id=case.case_id, case_name=case.name,
                scenario_type=case.scenario_type, system="insightpulse",
            )
            eval_run.metrics = {n: m.to_dict() for n, m in measured.items()}
            outcome, reasons, gates = engine.decide_outcome(case, eval_run)
            return {
                "available": True,
                "scores": scores,
                "unavailable": unavailable,
                "outcome": outcome,
                "reasons": reasons,
                "gate_failures": gates,
            }
        except Exception as exc:  # noqa: BLE001 — report, never fabricate
            return {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "Task 6 validation could not run, so quality was not verified",
            }

    def _case(self, case_id: str) -> Any:
        from ..evaluation.dataset import all_cases, get_case

        case = get_case(case_id)
        if case is None:
            cases = all_cases()
            case = cases[0]
        return case


loop = SelfImprovementLoop()
