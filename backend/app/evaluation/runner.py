"""Suite runner — orchestrates cases, repetitions, baselines and aggregation.

Run modes (section 35): single case, full/demo suite, repeated run, baseline
comparison, adversarial suite. Progress is streamed through the same optional queue
pattern the agent SSE endpoints use, so the dashboard can show live progress.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from . import metrics as M
from . import store
from .automated import ConsistencyEvaluator, ReliabilityEvaluator, RobustnessEvaluator
from .baseline import BASELINE_LLM, BASELINE_PIPELINE
from .dataset import all_cases, cases_by_scenario, demo_suite, get_case
from .engine import SYSTEM_INSIGHTPULSE, EvaluationEngine
from .regression import compare_with_previous
from .schemas import SCENARIO_TYPES, EvaluationCase, SuiteResult, Thresholds

# Safe ceiling on repetitions (section 36).
MAX_REPEATS = 10

# Metrics aggregated into the suite headline.
HEADLINE = (
    M.ACCURACY, M.TASK_COMPLETION, M.EVIDENCE_QUALITY, M.GROUNDEDNESS,
    M.HALLUCINATION_RATE, M.RECOVERY_RATE, M.LATENCY, M.RESOURCE_EFFICIENCY,
    M.EFFICIENCY, M.UNCERTAINTY_HANDLING, M.UNSUPPORTED_CONCLUSION_RATE,
)

# Outcome → numeric credit used for scenario-category scoring.
OUTCOME_CREDIT = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0, "ERROR": 0.0}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SuiteRunner:
    def __init__(
        self,
        *,
        thresholds: Thresholds | None = None,
        simulation_mode: bool = True,
        queue: Any = None,
    ) -> None:
        self.thresholds = thresholds or Thresholds()
        self.engine = EvaluationEngine(self.thresholds)
        self.simulation_mode = simulation_mode
        self.queue = queue

    # ── progress ────────────────────────────────────────────
    def _emit(self, event: str, **data: Any) -> None:
        if self.queue is None:
            return
        try:
            self.queue.put_nowait({"type": "evaluation", "event": event, **data})
        except Exception:  # noqa: BLE001 — progress reporting is never load-bearing
            pass

    # ─────────────────────────────────────────────────────────
    # main entry point
    # ─────────────────────────────────────────────────────────
    async def run_suite(
        self,
        *,
        mode: str = "demo",
        case_ids: list[str] | None = None,
        scenario: str = "",
        repeats: int | None = None,
        include_baseline: bool = True,
        baseline_systems: list[str] | None = None,
    ) -> dict[str, Any]:
        started_perf = time.perf_counter()
        cases = self._select_cases(mode=mode, case_ids=case_ids, scenario=scenario)
        suite = SuiteResult(
            suite_id=f"suite-{uuid.uuid4().hex[:10]}",
            mode=mode,
            thresholds=self.thresholds.to_dict(),
        )

        if not cases:
            # Empty suite is a legitimate state, not an error (section 51.11).
            suite.status = "completed"
            suite.completed_at = _now()
            suite.counts = {"cases": 0, "runs": 0, "pass": 0, "partial": 0, "fail": 0, "error": 0}
            suite.aggregate = {"note": "no cases selected", "overall_score": None}
            suite.notes.append("No evaluation cases matched the request, so nothing was measured.")
            suite.scenario_matrix = _empty_matrix()
            store.save_suite(suite.to_dict())
            self._emit("suite_completed", suite_id=suite.suite_id, cases=0)
            return suite.to_dict()

        self._emit("suite_started", suite_id=suite.suite_id, total_cases=len(cases), mode=mode)

        runs: list[dict[str, Any]] = []
        raw_by_case: dict[str, list[dict[str, Any]]] = {}
        eval_by_case: dict[str, list[Any]] = {}

        # ── InsightPulse executions ──
        for position, case in enumerate(cases, start=1):
            n = self._repeats_for(case, repeats)
            self._emit("case_started", case_id=case.case_id, name=case.name,
                       scenario=case.scenario_type, repeats=n, position=position,
                       total=len(cases))
            for idx in range(n):
                eval_run, raw = await self.engine.evaluate_case(
                    case, system=SYSTEM_INSIGHTPULSE, repeat_index=idx,
                    simulation_mode=self.simulation_mode,
                )
                runs.append(eval_run.to_dict())
                eval_by_case.setdefault(case.case_id, []).append(eval_run)
                if raw:
                    raw_by_case.setdefault(case.case_id, []).append(raw)
                self._emit(
                    "case_result", case_id=case.case_id, repeat=idx,
                    outcome=eval_run.outcome, scenario=case.scenario_type,
                    groundedness=eval_run.score(M.GROUNDEDNESS),
                    hallucination=eval_run.score(M.HALLUCINATION_RATE),
                    recovery=eval_run.score(M.RECOVERY_RATE),
                    latency=eval_run.score(M.LATENCY),
                    gate_failures=eval_run.gate_failures,
                )

        # ── reliability + consistency (grouped by stable case_id) ──
        suite.reliability = self._reliability(eval_by_case)
        suite.consistency = self._consistency(raw_by_case)

        # ── baselines ──
        baseline_runs: list[dict[str, Any]] = []
        if include_baseline:
            systems = baseline_systems or [BASELINE_PIPELINE, BASELINE_LLM]
            baseline_cases = self._baseline_cases(cases)
            for system in systems:
                self._emit("baseline_started", system=system, cases=len(baseline_cases))
                for case in baseline_cases:
                    eval_run, _ = await self.engine.evaluate_case(
                        case, system=system, repeat_index=0,
                        simulation_mode=self.simulation_mode,
                    )
                    baseline_runs.append(eval_run.to_dict())
                    self._emit("baseline_result", system=system, case_id=case.case_id,
                               outcome=eval_run.outcome)
            runs.extend(baseline_runs)

        # ── aggregation ──
        primary = [r for r in runs if r.get("system") == SYSTEM_INSIGHTPULSE]
        suite.runs = runs
        suite.aggregate = self._aggregate(primary)
        suite.scenario_matrix = self._scenario_matrix(primary)

        robustness = RobustnessEvaluator().evaluate(suite.scenario_matrix)
        suite.aggregate["robustness"] = robustness.to_dict()
        # Reliability and consistency are per-case (they need repetitions), so they are
        # rolled up into the suite aggregate here. Without this they existed only in
        # the per-case blocks and never reached the metric cards, which is why the
        # dashboard reported them as unavailable even though they had been measured.
        suite.aggregate[M.RELIABILITY] = self._rollup(
            M.RELIABILITY, suite.reliability,
            "no case in this suite was repeated, so repeatability cannot be measured",
        )
        suite.aggregate[M.CONSISTENCY] = self._rollup(
            M.CONSISTENCY, suite.consistency,
            "no case in this suite was repeated, so run-to-run agreement cannot be measured",
        )
        if robustness.available:
            suite.aggregate["overall_score"] = self._overall(suite.aggregate)

        if include_baseline and baseline_runs:
            suite.baseline_comparison = self._baseline_comparison(primary, baseline_runs)

        suite.counts = self._counts(primary, baseline_runs, cases)
        suite.provenance = {
            "suite_id": suite.suite_id,
            "mode": mode,
            "simulation_mode": self.simulation_mode,
            "case_ids": [c.case_id for c in cases],
            "thresholds": self.thresholds.to_dict(),
            "evaluated_at": _now(),
            "duration_ms": int((time.perf_counter() - started_perf) * 1000),
            "framework_version": "langgraph-stategraph-1",
            "storage": store.status(),
        }
        suite.regression = compare_with_previous(suite.suite_id, suite.aggregate)
        suite.status = "completed"
        suite.completed_at = _now()

        store.save_suite(suite.to_dict())
        self._emit("suite_completed", suite_id=suite.suite_id,
                   overall=suite.aggregate.get("overall_score"),
                   counts=suite.counts)
        return suite.to_dict()

    # ─────────────────────────────────────────────────────────
    # single case + repeat helpers
    # ─────────────────────────────────────────────────────────
    async def run_single(self, case_id: str, *, repeats: int = 1) -> dict[str, Any]:
        return await self.run_suite(
            mode="single", case_ids=[case_id], repeats=repeats, include_baseline=False
        )

    async def run_repeated(self, case_id: str, *, repeats: int = 3) -> dict[str, Any]:
        return await self.run_suite(
            mode="repeated", case_ids=[case_id], repeats=repeats, include_baseline=False
        )

    async def run_adversarial(self) -> dict[str, Any]:
        ids = [
            c.case_id for c in all_cases()
            if c.scenario_type in {"ADVERSARIAL", "TOOL_FAILURE", "CONTRADICTORY"}
        ]
        return await self.run_suite(mode="adversarial", case_ids=ids, include_baseline=False)

    # ─────────────────────────────────────────────────────────
    # selection
    # ─────────────────────────────────────────────────────────
    def _select_cases(
        self, *, mode: str, case_ids: list[str] | None, scenario: str
    ) -> list[EvaluationCase]:
        if case_ids:
            return [c for cid in case_ids if (c := get_case(cid)) is not None]
        if scenario:
            return cases_by_scenario(scenario)
        if mode == "full":
            return all_cases()
        return demo_suite()

    def _repeats_for(self, case: EvaluationCase, override: int | None) -> int:
        n = override if isinstance(override, int) and override > 0 else case.repeat_count
        return max(1, min(int(n), MAX_REPEATS))

    def _baseline_cases(self, cases: list[EvaluationCase]) -> list[EvaluationCase]:
        """Baselines run the same cases, minus those that are structurally unfair.

        A fault-injection case cannot be run against a system with no tools, so it is
        excluded and the exclusion is reported rather than scored as a baseline loss.
        """
        return [
            c for c in cases
            if c.scenario_type not in {"TOOL_FAILURE", "ADVERSARIAL"}
        ]

    # ─────────────────────────────────────────────────────────
    # aggregation
    # ─────────────────────────────────────────────────────────
    def _aggregate(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in HEADLINE:
            values = [
                m.get("value") for r in runs
                if (m := (r.get("metrics") or {}).get(name)) and m.get("available")
                and isinstance(m.get("value"), (int, float))
            ]
            spec = M.spec(name)
            if not values:
                out[name] = spec.unavailable("no run produced this metric").to_dict()
                continue
            mean = M.mean_of(values)
            entry = spec.result(mean, details={"samples": len(values)}).to_dict()
            if name == M.LATENCY:
                entry["details"].update(M.distribution(values))
            out[name] = entry
        return out

    def _rollup(self, name: str, per_case: dict[str, Any], empty_reason: str) -> dict[str, Any]:
        """Aggregate a per-case metric block (keyed by case_id) into one metric.

        The block holds one `MetricResult` dict per repeated case. The mean across
        cases is the suite figure; the per-case detail is kept so the report can show
        which case contributed what.
        """
        spec = M.spec(name)
        entries = {
            case_id: entry for case_id, entry in (per_case or {}).items()
            if isinstance(entry, dict) and entry.get("available")
            and isinstance(entry.get("value"), (int, float))
        }
        if not entries:
            return spec.unavailable(empty_reason).to_dict()
        values = [float(e["value"]) for e in entries.values()]
        details: dict[str, Any] = {
            "cases_measured": sorted(entries),
            "per_case": {cid: e.get("value") for cid, e in entries.items()},
        }
        # Surface the run counts the reliability evaluator recorded, so the dashboard
        # can show successful/partial/failed without another request.
        for case_id, entry in entries.items():
            d = entry.get("details") or {}
            for key in ("total_runs", "successful_runs", "partial_runs", "failed_runs",
                        "mean_time_to_completion_ms", "repetitions", "pairs_compared"):
                if key in d:
                    details.setdefault(key, d[key])
            details.setdefault("primary_case", case_id)
        return spec.result(M.mean_of(values), details=details).to_dict()

    def _overall(self, aggregate: dict[str, Any]) -> float | None:
        """Single headline score: mean of quality metrics, with lower-is-better
        metrics inverted so the direction is consistent."""
        parts: list[float] = []
        for name in (M.ACCURACY, M.TASK_COMPLETION, M.EVIDENCE_QUALITY, M.GROUNDEDNESS,
                     M.UNCERTAINTY_HANDLING, M.RELIABILITY, M.CONSISTENCY, "robustness"):
            m = aggregate.get(name) or {}
            if m.get("available") and isinstance(m.get("value"), (int, float)):
                parts.append(float(m["value"]))
        h = aggregate.get(M.HALLUCINATION_RATE) or {}
        if h.get("available") and isinstance(h.get("value"), (int, float)):
            parts.append(1.0 - float(h["value"]))
        return round(M.mean_of(parts), 4) if parts else None

    def _scenario_matrix(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        matrix = _empty_matrix()
        for run in runs:
            key = str(run.get("scenario_type") or "NORMAL")
            bucket = matrix.setdefault(key, {"total": 0, "passed": 0, "partial": 0,
                                             "failed": 0, "error": 0, "score": 0.0})
            bucket["total"] += 1
            outcome = str(run.get("outcome") or "PARTIAL")
            bucket[{"PASS": "passed", "PARTIAL": "partial",
                    "FAIL": "failed", "ERROR": "error"}.get(outcome, "partial")] += 1
        for bucket in matrix.values():
            if bucket["total"]:
                credit = bucket["passed"] * 1.0 + bucket["partial"] * 0.5
                bucket["score"] = round(credit / bucket["total"], 4)
        return matrix

    def _counts(
        self, primary: list[dict[str, Any]], baseline: list[dict[str, Any]],
        cases: list[EvaluationCase],
    ) -> dict[str, int]:
        return {
            "cases": len(cases),
            "runs": len(primary),
            "baseline_runs": len(baseline),
            "pass": sum(1 for r in primary if r.get("outcome") == "PASS"),
            "partial": sum(1 for r in primary if r.get("outcome") == "PARTIAL"),
            "fail": sum(1 for r in primary if r.get("outcome") == "FAIL"),
            "error": sum(1 for r in primary if r.get("outcome") == "ERROR"),
        }

    # ── reliability / consistency ───────────────────────────
    def _reliability(self, eval_by_case: dict[str, list[Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for case_id, eval_runs in eval_by_case.items():
            if len(eval_runs) < 2:
                continue
            result = ReliabilityEvaluator().evaluate(
                [r.outcome for r in eval_runs],
                [(r.provenance or {}).get("system", "") for r in eval_runs],
                [v for r in eval_runs if (v := r.score(M.LATENCY)) is not None],
            )
            out[case_id] = result.to_dict()
        if not out:
            out["note"] = "no case was repeated in this suite, so reliability is not measurable"
        return out

    def _consistency(self, raw_by_case: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for case_id, raws in raw_by_case.items():
            if len(raws) < 2:
                continue
            out[case_id] = ConsistencyEvaluator().evaluate(raws).to_dict()
        if not out:
            out["note"] = "no case was repeated in this suite, so consistency is not measurable"
        return out

    # ── baseline comparison (section 28) ────────────────────
    def _baseline_comparison(
        self, primary: list[dict[str, Any]], baseline_runs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        systems: dict[str, list[dict[str, Any]]] = {}
        for run in baseline_runs:
            systems.setdefault(str(run.get("system")), []).append(run)

        # Compare only on the cases both sides actually ran, so the difference is fair.
        comparisons: dict[str, Any] = {}
        for system, runs in systems.items():
            shared = {str(r.get("case_id")) for r in runs}
            ours = [r for r in primary if str(r.get("case_id")) in shared]
            blocked_reasons = sorted({
                str((r.get("provenance") or {}).get("baseline_blocked_reason") or "")
                for r in runs
                if (r.get("provenance") or {}).get("baseline_blocked")
            } - {""})
            rows: list[dict[str, Any]] = []
            for name in HEADLINE:
                spec = M.spec(name)
                b_vals = _values(runs, name)
                i_vals = _values(ours, name)
                b_reason = _first_unavailable_reason(runs, name)

                # A baseline that could not run at all did not score badly — it did
                # not score. Its output-derived metrics collapse to zero because there
                # is no output, and comparing against that would manufacture an
                # improvement. So when the baseline is blocked, nothing is comparable.
                if blocked_reasons:
                    rows.append({
                        "metric": name, "label": spec.label, "unit": spec.unit,
                        "higher_is_better": spec.higher_is_better,
                        "baseline": None, "insightpulse": M.mean_of(i_vals),
                        "difference": None, "relative_improvement": None,
                        "direction": "not_comparable", "available": False,
                        "unavailable_reason": (
                            f"baseline unavailable — {'; '.join(blocked_reasons)}; "
                            f"a system that produced no output cannot be scored"
                        ),
                    })
                    continue

                row: dict[str, Any] = {
                    "metric": name,
                    "label": spec.label,
                    "unit": spec.unit,
                    "higher_is_better": spec.higher_is_better,
                    "baseline": M.mean_of(b_vals),
                    "insightpulse": M.mean_of(i_vals),
                    "difference": None,
                    "relative_improvement": None,
                    "available": bool(b_vals and i_vals),
                    "unavailable_reason": "" if (b_vals and i_vals) else (
                        b_reason or "metric not produced by both systems on the shared cases"
                    ),
                }
                if row["available"]:
                    b, i = float(row["baseline"]), float(row["insightpulse"])
                    diff = i - b
                    row["difference"] = round(diff, 4)
                    # For lower-is-better metrics, improvement means going down.
                    gain = (b - i) if not spec.higher_is_better else (i - b)
                    # Tri-state: equal is not a loss, and reporting it as one would
                    # overstate the comparison in either direction.
                    row["direction"] = (
                        "equal" if abs(gain) < 1e-9 else ("better" if gain > 0 else "worse")
                    )
                    row["improved"] = gain > 0
                    if b != 0:
                        row["relative_improvement"] = round(gain / abs(b), 4)
                    else:
                        row["relative_improvement"] = None
                        row["relative_note"] = "baseline is zero, so relative change is undefined"
                rows.append(row)
            comparisons[system] = {
                "system": system,
                "blocked": bool(blocked_reasons),
                "blocked_reason": "; ".join(blocked_reasons),
                "blocked_note": (
                    "This baseline could not produce output during the run, so its scores "
                    "reflect an unavailable system rather than a measured quality gap."
                    if blocked_reasons else ""
                ),
                "cases_compared": sorted(shared),
                "excluded_cases": sorted(
                    {str(r.get("case_id")) for r in primary} - shared
                ),
                "exclusion_reason": (
                    "fault-injection and adversarial cases are not run against baselines, "
                    "because a system with no tool layer cannot be fairly subjected to "
                    "tool failure"
                ),
                "rows": rows,
            }
        return comparisons


def _values(runs: list[dict[str, Any]], metric: str) -> list[float]:
    out: list[float] = []
    for run in runs:
        m = (run.get("metrics") or {}).get(metric) or {}
        if m.get("available") and isinstance(m.get("value"), (int, float)):
            out.append(float(m["value"]))
    return out


def _first_unavailable_reason(runs: list[dict[str, Any]], metric: str) -> str:
    for run in runs:
        m = (run.get("metrics") or {}).get(metric) or {}
        if not m.get("available") and m.get("unavailable_reason"):
            return str(m["unavailable_reason"])
    return ""


def _empty_matrix() -> dict[str, Any]:
    return {
        s: {"total": 0, "passed": 0, "partial": 0, "failed": 0, "error": 0, "score": 0.0}
        for s in SCENARIO_TYPES
    }
