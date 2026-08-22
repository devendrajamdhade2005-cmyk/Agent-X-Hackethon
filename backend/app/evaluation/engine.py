"""Evaluation engine — execute a case for real, then measure what happened.

The engine never simulates a successful agent result. It calls the actual runtime
(`run_graph`, i.e. Tasks 1–5 end to end), captures the full execution record, and
hands that record to the deterministic evaluators. Fault injection reuses the
existing Task 5 adversarial knobs, so evaluation exercises the production recovery
path rather than a parallel one.

Outcome logic is transparent and gate-based: a critical breach fails the case
outright, so one excellent metric can never mask a serious failure (section 48).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ..graph.adversarial import AdversarialConfig
from ..graph.runner import run_graph
from . import metrics as M
from .automated import (
    ClaimExtractor,
    CorrectnessEvaluator,
    EfficiencyEvaluator,
    EvidenceEvaluator,
    GroundednessEvaluator,
    HallucinationEvaluator,
    LatencyEvaluator,
    RecoveryEvaluator,
    RefusalEvaluator,
    ResourceEvaluator,
    TaskCompletionEvaluator,
    UncertaintyEvaluator,
)
from .baseline import (
    BASELINE_LLM,
    BASELINE_PIPELINE,
    run_baseline_llm,
    run_baseline_pipeline,
    unavailable_reason,
)
from .schemas import EvaluationCase, EvaluationRun, Thresholds

SYSTEM_INSIGHTPULSE = "insightpulse"

# The framework version this evaluation measured, recorded for provenance.
FRAMEWORK_VERSION = "langgraph-stategraph-1"


class EvaluationEngine:
    """Executes and scores one (case, system, repeat) at a time."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds()
        self.claims = ClaimExtractor()
        self.correctness = CorrectnessEvaluator()
        self.completion = TaskCompletionEvaluator()
        self.evidence = EvidenceEvaluator()
        self.groundedness = GroundednessEvaluator()
        self.hallucination = HallucinationEvaluator()
        self.recovery = RecoveryEvaluator()
        self.latency = LatencyEvaluator()
        self.resource = ResourceEvaluator()
        self.efficiency = EfficiencyEvaluator()
        self.uncertainty = UncertaintyEvaluator()
        self.refusal = RefusalEvaluator()

    # ─────────────────────────────────────────────────────────
    # execution
    # ─────────────────────────────────────────────────────────
    async def execute(
        self,
        case: EvaluationCase,
        *,
        system: str = SYSTEM_INSIGHTPULSE,
        repeat_index: int = 0,
        simulation_mode: bool = True,
    ) -> dict[str, Any]:
        """Run the case against a system and return the raw execution record."""
        adversarial = self._adversarial_for(case)
        if system == BASELINE_LLM:
            return await run_baseline_llm(
                case.user_goal, keywords=case.keywords, competitors=case.competitors
            )
        if system == BASELINE_PIPELINE:
            return await run_baseline_pipeline(
                case.user_goal, keywords=case.keywords, competitors=case.competitors,
                simulation_mode=simulation_mode,
            )
        return await run_graph(
            case.user_goal,
            keywords=case.keywords,
            competitors=case.competitors,
            simulation_mode=simulation_mode,
            adversarial=adversarial,
            thread_id=f"eval-{case.case_id}-{repeat_index}-{uuid.uuid4().hex[:6]}",
        )

    def _adversarial_for(self, case: EvaluationCase) -> AdversarialConfig | None:
        """Translate the case's declared injections into the Task 5 knobs."""
        spec = case.failure_injections or {}
        if not spec:
            return None
        scenario = str(spec.get("scenario") or "full")
        competitor = case.competitors[0] if case.competitors else "OpenAI"
        cfg = AdversarialConfig.named(scenario, competitor)
        override = spec.get("budget_override")
        if isinstance(override, dict):
            cfg.budget_override = {**cfg.budget_override, **override}
        return cfg

    # ─────────────────────────────────────────────────────────
    # measurement
    # ─────────────────────────────────────────────────────────
    async def evaluate_case(
        self,
        case: EvaluationCase,
        *,
        system: str = SYSTEM_INSIGHTPULSE,
        repeat_index: int = 0,
        simulation_mode: bool = True,
    ) -> tuple[EvaluationRun, dict[str, Any]]:
        """Execute + score. Returns (evaluation run, raw agent result)."""
        started = time.perf_counter()
        eval_run = EvaluationRun(
            evaluation_run_id=f"ev-{uuid.uuid4().hex[:12]}",
            case_id=case.case_id,
            case_name=case.name,
            scenario_type=case.scenario_type,
            system=system,
            repeat_index=repeat_index,
        )

        try:
            run = await self.execute(
                case, system=system, repeat_index=repeat_index,
                simulation_mode=simulation_mode,
            )
        except Exception as exc:  # noqa: BLE001 — a broken case must not end the suite
            eval_run.status = "failed"
            eval_run.outcome = "ERROR"
            eval_run.error = f"{type(exc).__name__}: {exc}"
            eval_run.completed_at = _now_iso()
            eval_run.outcome_reasons = ["the execution raised before producing a result"]
            return eval_run, {}

        eval_run.agent_run_id = str(run.get("run_id") or "")
        results = self.measure(case, run, system=system)
        eval_run.metrics = {name: m.to_dict() for name, m in results.items()}
        eval_run.claims = [c.to_dict() for c in self._last_claims]
        eval_run.status = "completed"

        outcome, reasons, gates = self.decide_outcome(case, eval_run)
        eval_run.outcome = outcome
        eval_run.outcome_reasons = reasons
        eval_run.gate_failures = gates
        eval_run.completed_at = _now_iso()
        eval_run.provenance = self._provenance(case, run, system, repeat_index, simulation_mode)
        eval_run.provenance["evaluation_time_ms"] = int((time.perf_counter() - started) * 1000)
        return eval_run, run

    def measure(
        self, case: EvaluationCase, run: dict[str, Any], *, system: str = SYSTEM_INSIGHTPULSE
    ) -> dict[str, Any]:
        """Run every applicable evaluator over one captured execution."""
        claims = self.claims.extract(run)
        self._last_claims = claims

        evidence = self.evidence.evaluate(case, run)
        evidence_value = evidence.value if evidence.available else None

        out = {
            M.ACCURACY: self.correctness.evaluate(case, run),
            M.TASK_COMPLETION: self.completion.evaluate(case, run),
            M.EVIDENCE_QUALITY: evidence,
            M.GROUNDEDNESS: self.groundedness.evaluate(case, claims),
            M.HALLUCINATION_RATE: self.hallucination.evaluate(case, claims),
            M.RECOVERY_RATE: self.recovery.evaluate(case, run),
            M.LATENCY: self.latency.evaluate(case, run),
            M.RESOURCE_EFFICIENCY: self.resource.evaluate(case, run),
            M.EFFICIENCY: self.efficiency.evaluate(case, run),
            M.UNCERTAINTY_HANDLING: self.uncertainty.evaluate(case, run, evidence_value),
            M.UNSUPPORTED_CONCLUSION_RATE: self.refusal.evaluate(case, run),
        }

        # Replace metrics that are structurally inapplicable to a baseline with an
        # explicit, reasoned 'unavailable' rather than a misleading number.
        if system in {BASELINE_LLM, BASELINE_PIPELINE}:
            for name in list(out.keys()):
                reason = unavailable_reason(system, name)
                if reason:
                    out[name] = M.spec(name).unavailable(reason)
        return out

    # ─────────────────────────────────────────────────────────
    # outcome (sections 48-49)
    # ─────────────────────────────────────────────────────────
    def decide_outcome(
        self, case: EvaluationCase, eval_run: EvaluationRun
    ) -> tuple[str, list[str], list[str]]:
        t = self.thresholds
        reasons: list[str] = []
        gates: list[str] = []
        critical: list[str] = []

        ground = eval_run.score(M.GROUNDEDNESS)
        halluc = eval_run.score(M.HALLUCINATION_RATE)
        completion = eval_run.score(M.TASK_COMPLETION)
        recovery = eval_run.score(M.RECOVERY_RATE)
        evidence = eval_run.score(M.EVIDENCE_QUALITY)
        latency = eval_run.score(M.LATENCY)
        uncertainty = eval_run.score(M.UNCERTAINTY_HANDLING)
        unsupported = eval_run.score(M.UNSUPPORTED_CONCLUSION_RATE)

        # ── hard gates ──
        if halluc is not None and halluc > t.critical_hallucination_rate:
            critical.append(
                f"hallucination rate {halluc:.0%} exceeds the critical ceiling "
                f"{t.critical_hallucination_rate:.0%}"
            )
        if ground is not None and ground < t.critical_groundedness:
            critical.append(
                f"groundedness {ground:.0%} is below the critical floor "
                f"{t.critical_groundedness:.0%}"
            )
        # A case built to test refusal fails outright if the agent asserted anyway.
        if case.expected_refusal and unsupported is not None and unsupported > 0:
            critical.append("asserted a conclusion the evidence does not support")
        # A case built to test recovery fails if nothing recovered.
        if case.expected_recovery and recovery is not None and recovery < 0.5:
            critical.append(f"recovery rate {recovery:.0%} on a case that injected a failure")
        # A case built to test conflict handling fails if the contradiction was never
        # detected. Read from the completion evaluator's per-subtask evidence, which
        # checks the framework's `conflicting_evidence` directly.
        if case.expects_conflict_detection and not self._subtask_done(eval_run, "contradiction"):
            critical.append("did not detect the injected contradiction")

        # ── soft gates ──
        def soft(name: str, value: float | None, ok: bool, message: str) -> None:
            if value is None:
                return
            if not ok:
                gates.append(name)
                reasons.append(message)

        soft("groundedness", ground, (ground or 0) >= t.min_groundedness,
             f"groundedness {(ground or 0):.0%} below target {t.min_groundedness:.0%}")
        soft("hallucination", halluc, (halluc if halluc is not None else 0) <= t.max_hallucination_rate,
             f"hallucination {(halluc or 0):.0%} above target {t.max_hallucination_rate:.0%}")
        soft("task_completion", completion, (completion or 0) >= t.min_task_completion,
             f"task completion {(completion or 0):.0%} below target {t.min_task_completion:.0%}")
        soft("recovery", recovery, (recovery or 0) >= t.min_recovery_rate,
             f"recovery {(recovery or 0):.0%} below target {t.min_recovery_rate:.0%}")
        soft("evidence_quality", evidence, (evidence or 0) >= t.min_evidence_quality,
             f"evidence quality {(evidence or 0):.0%} below target {t.min_evidence_quality:.0%}")
        soft("latency", latency, (latency or 0) <= t.max_latency_ms,
             f"latency {int(latency or 0)}ms above target {t.max_latency_ms}ms")
        if uncertainty is not None and uncertainty < 0.5:
            gates.append("uncertainty_handling")
            reasons.append("expressed certainty was not calibrated to evidence strength")

        if critical:
            return "FAIL", critical + reasons, ["critical:" + c for c in critical] + gates
        if gates:
            return "PARTIAL", reasons, gates
        return "PASS", ["all configured quality gates met"], []

    @staticmethod
    def _subtask_done(eval_run: EvaluationRun, needle: str) -> bool:
        """Was a subtask matching `needle` verified as completed?"""
        details = (eval_run.metrics.get(M.TASK_COMPLETION) or {}).get("details") or {}
        for check in details.get("checks") or []:
            if needle in str(check.get("subtask", "")).lower():
                return bool(check.get("completed"))
        return False

    # ─────────────────────────────────────────────────────────
    def _provenance(
        self, case: EvaluationCase, run: dict[str, Any], system: str,
        repeat_index: int, simulation_mode: bool,
    ) -> dict[str, Any]:
        fw = run.get("framework") or {}
        mx = run.get("metrics") or {}
        return {
            "evaluation_case_id": case.case_id,
            "agent_run_id": run.get("run_id"),
            "thread_id": run.get("thread_id"),
            "system": system,
            "scenario_type": case.scenario_type,
            "framework_version": FRAMEWORK_VERSION,
            "runtime": fw.get("runtime") or mx.get("runtime"),
            "evaluated_at": _now_iso(),
            "simulation_mode": simulation_mode,
            "repeat_index": repeat_index,
            "adversarial": (fw.get("adversarial") or {}).get("scenario") if fw.get("adversarial") else None,
            "tool_configuration": sorted({
                str(t.get("tool_name")) for t in (fw.get("tool_executions") or []) if t.get("tool_name")
            }),
            "reasoner": mx.get("reasoner") or (mx.get("llm") or {}).get("provider"),
            # Baselines report whether an external condition (quota, credential)
            # prevented them from answering at all, so a blocked baseline is never
            # mistaken for a measured quality gap.
            "baseline_blocked": bool(run.get("baseline_blocked")),
            "baseline_blocked_reason": str(run.get("baseline_blocked_reason") or ""),
            "baseline_notes": str(run.get("baseline_notes") or ""),
            "deterministic": simulation_mode,
            "determinism_note": (
                "simulation mode: providers return deterministic fixtures, so repeated "
                "evaluation is stable"
                if simulation_mode else
                "live mode: external APIs make repeated results non-deterministic"
            ),
        }


def _now_iso() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat(timespec="seconds")
