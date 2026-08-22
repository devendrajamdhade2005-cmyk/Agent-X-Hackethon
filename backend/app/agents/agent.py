"""InsightPulse agent — the ReAct loop.

    GOAL
      ↓
    REASON / PLAN
      ↓
    DECIDE NEXT ACTION ──────────────┐
      ↓                              │
    SELECT TOOL → CALL TOOL          │
      ↓                              │
    OBSERVE RESULT → ANALYZE RESULT ─┘  (more information needed?)
      ↓ no
    GENERATE PRIORITIZED INSIGHTS

Nothing in this file decides *which* tool to call — that is the decision engine's
job, driven by state. This file owns the loop, the iteration budget, the error
containment and the narration.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import settings
from ..services.activity_logger import ActivityEntry, ActivityLogger
from ..sources.registry import build_http_client, registry as source_registry
from ..tools.base import ToolContext, ToolInput, ToolResult
from ..tools.registry import ToolRegistry, tool_registry
from .decision_engine import (
    MIN_TOTAL_RELEVANT,
    NEED_TO_TOOL,
    TOOL_TO_NEED,
    DecisionEngine,
    ObservationAnalyzer,
)
from .insight_generator import HIGH, LOW, MEDIUM, InsightGenerator, SummaryWriter
from .llm import LLMClient
from .planner import Planner
from .state import MAX_ITERATIONS, AgentState, Decision, ToolCallRecord


@dataclass
class AgentRunRequest:
    goal: str
    keywords: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    max_iterations: int | None = None
    simulation_mode: bool | None = None


@dataclass
class AgentRunResult:
    status: str
    run_id: str
    goal: str
    activity_log: list[dict[str, Any]]
    tools_used: list[str]
    findings: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    summary: str
    state: dict[str, Any]
    metrics: dict[str, Any]
    activity_text: str = ""
    insights_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "goal": self.goal,
            "activity_log": self.activity_log,
            "tools_used": self.tools_used,
            "findings": self.findings,
            "insights": self.insights,
            "summary": self.summary,
            "state": self.state,
            "metrics": self.metrics,
            "activity_text": self.activity_text,
            "insights_text": self.insights_text,
        }


class InsightPulseAgent:
    """One instance per run. Safe to construct without any API keys."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        llm: LLMClient | None = None,
        logger: ActivityLogger | None = None,
        simulation_mode: bool | None = None,
        echo: bool = False,
        on_event: Callable[[ActivityEntry], None] | None = None,
        queue: asyncio.Queue | None = None,
    ) -> None:
        self.tools = tools or tool_registry
        self.llm = llm if llm is not None else LLMClient()
        self.state = AgentState()
        self.logger = logger or ActivityLogger(
            self.state.run_id, sink=on_event, queue=queue, echo=echo
        )
        self.simulation_mode = (
            settings.simulation_mode if simulation_mode is None else simulation_mode
        )
        self.planner = Planner(self.llm)
        self.decider = DecisionEngine(self.tools, self.llm)
        self.analyzer = ObservationAnalyzer()
        self.insight_writer = InsightGenerator(self.llm)
        self.summary_writer = SummaryWriter(self.llm)

    # ─────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        started = time.perf_counter()
        state = self.state
        state.user_goal = (request.goal or "").strip()
        state.max_iterations = max(1, min(int(request.max_iterations or MAX_ITERATIONS), 25))
        if request.simulation_mode is not None:
            self.simulation_mode = request.simulation_mode

        if not state.user_goal:
            state.status = "failed"
            self.logger.error("No goal provided", "The agent needs a tracking goal to work on.")
            return self._result(started)

        self.logger.start(state.user_goal, run_id=state.run_id)
        await self._check_reasoner()

        try:
            async with build_http_client(timeout=settings.collect_timeout_seconds) as client:
                ctx = ToolContext(
                    http_client=client,
                    registry=source_registry,
                    simulation_mode=self.simulation_mode,
                )
                await self._understand_goal(request)
                await self._plan()
                await self._loop(ctx)
                await self._finalize()
        except asyncio.CancelledError:
            state.status = "failed"
            state.stop_reason = "cancelled"
            self.logger.error("Run cancelled", "The request was cancelled before completion.")
            raise
        except Exception as exc:  # noqa: BLE001 — always return something useful
            state.record_error("agent.run", f"{type(exc).__name__}: {exc}", recovered=False)
            self.logger.error(
                "Unexpected failure in the agent loop",
                f"{type(exc).__name__}: {exc}. Reporting on whatever was collected.",
            )
            try:
                await self._finalize()
            except Exception:  # noqa: BLE001
                state.status = "failed"

        return self._result(started)

    async def _check_reasoner(self) -> None:
        """Confirm the LLM credential before planning, and report what we got.

        Listing models is not proof of generation access, so `verify()` actually
        generates. If the credential is rejected the agent keeps running on its
        deterministic reasoner — but it must never claim a model was involved when
        it wasn't, so the outcome is written to the activity log either way.
        """
        state = self.state
        if not self.llm.configured:
            state.reasoner = "heuristic"
            self.logger.thought(
                "Reasoning with the built-in heuristic reasoner",
                "No LLM credential is configured, so planning, decisions and "
                "prioritization use the deterministic reasoner.",
                reasoner="heuristic",
            )
            return

        ok, reason = await self.llm.verify()
        state.reasoner = self.llm.reasoner_name
        state.llm_calls = self.llm.usage.calls
        if ok:
            self.logger.thought(
                f"Reasoning with {self.llm.provider.name} ({self.llm.model})",
                "The model plans, chooses each next action and writes the insight "
                "priorities. The deterministic reasoner stays as a fallback.",
                reasoner=state.reasoner,
            )
            switches = getattr(self.llm.provider, "model_switches", [])
            if switches:
                self.logger.warning(
                    "Switched model mid-verification",
                    "; ".join(switches) + " — the configured model was not usable "
                    "with this credential.",
                )
        else:
            state.record_error("llm.verify", reason)
            self.logger.warning(
                f"LLM unavailable → continuing with the heuristic reasoner",
                f"{self.llm.provider.name}: {reason}",
                reasoner="heuristic",
            )

    # ─────────────────────────────────────────────────────────
    # 1. GOAL
    # ─────────────────────────────────────────────────────────
    async def _understand_goal(self, request: AgentRunRequest) -> None:
        state = self.state
        state.current_step = "goal"
        state.keywords = [k.strip() for k in request.keywords if str(k).strip()][:10]
        state.competitors = [c.strip() for c in request.competitors if str(c).strip()][:8]
        state.tracking_topics = [t.strip() for t in request.topics if str(t).strip()][:8]

        availability = self.tools.availability()
        state.available_tools = [n for n, a in availability.items() if a.available]
        state.unavailable_tools = {
            n: a.reason or "unavailable" for n, a in availability.items() if not a.available
        }

        detail_bits = [f"Goal: {state.user_goal}"]
        if state.keywords:
            detail_bits.append(f"Keywords supplied: {', '.join(state.keywords)}")
        if state.competitors:
            detail_bits.append(f"Competitors to track: {', '.join(state.competitors)}")
        self.logger.goal(
            "Goal understood",
            " · ".join(detail_bits),
            keywords=state.keywords,
            competitors=state.competitors,
            available_tools=state.available_tools,
        )

        if state.unavailable_tools:
            names = ", ".join(state.unavailable_tools)
            rest = ", ".join(state.available_tools) or "no tools"
            self.logger.warning(
                f"{names} unavailable → continuing with {rest}",
                "; ".join(f"{k}: {v}" for k, v in state.unavailable_tools.items()),
                unavailable=state.unavailable_tools,
            )
        if self.simulation_mode:
            self.logger.warning(
                "Simulation mode is on",
                "All providers will return deterministic synthetic data, clearly labelled.",
            )

    # ─────────────────────────────────────────────────────────
    # 2. PLAN
    # ─────────────────────────────────────────────────────────
    async def _plan(self) -> None:
        state = self.state
        state.current_step = "plan"
        state.status = "planning"

        try:
            plan = await self.planner.build(
                state.user_goal,
                keywords=state.keywords,
                competitors=state.competitors,
                topics=state.tracking_topics,
            )
        except Exception as exc:  # noqa: BLE001
            state.record_error("planner", f"{type(exc).__name__}: {exc}")
            self.logger.warning(
                "Planner failed — falling back to a minimal plan",
                f"{type(exc).__name__}: {exc}",
            )
            plan = Planner(None)._heuristic_plan(  # noqa: SLF001 — deliberate fallback
                state.user_goal, state.keywords, state.competitors, state.tracking_topics
            )

        state.plan = plan
        derived = self.planner.derived()
        state.keywords = state.keywords or derived.get("keywords", [])
        state.competitors = state.competitors or derived.get("competitors", [])
        state.tracking_topics = state.tracking_topics or derived.get("topics", [])

        required = [n.key for n in plan.needs if n.required]
        optional = [n.key for n in plan.needs if not n.required]
        detail = plan.interpretation
        if plan.opening_move:
            detail += f" Opening move: {plan.opening_move}."
        self.logger.plan(
            f"Plan built — must satisfy: {', '.join(required) or 'none'}",
            detail,
            author=plan.author,
            required_needs=required,
            optional_needs=optional,
            needs=[n.to_dict() for n in plan.needs],
            keywords=state.keywords,
            competitors=state.competitors,
        )
        for need in plan.needs:
            if not need.required:
                self.logger.thought(
                    f"Holding back {NEED_TO_TOOL.get(need.key, need.key)}",
                    need.reason,
                    need=need.key,
                )

    # ─────────────────────────────────────────────────────────
    # 3-8. THE LOOP
    # ─────────────────────────────────────────────────────────
    async def _loop(self, ctx: ToolContext) -> None:
        state = self.state
        state.status = "running"

        while True:
            state.current_step = "decide"
            decision = await self._safe_decide()
            state.decisions.append(decision)

            if decision.action != "call_tool" or not decision.tool or not decision.tool_input:
                state.final_decision = decision.reasoning
                state.stop_reason = decision.reasoning
                self.logger.final(
                    "Enough information collected"
                    if "limit" not in decision.reasoning.lower()
                    else "Stopping at the iteration limit",
                    decision.reasoning,
                    author=decision.author,
                    iterations_used=state.iteration_count,
                )
                return

            state.iteration_count += 1
            iteration = state.iteration_count

            self.logger.decision(
                f"Next action → {decision.tool}",
                decision.reasoning,
                iteration=iteration,
                tool=decision.tool,
                author=decision.author,
                confidence=round(decision.confidence, 2),
                tool_input=decision.tool_input.describe(),
            )

            result = await self._call_tool(decision, ctx, iteration)
            self._observe(decision, result, iteration)

            if state.iteration_count >= state.max_iterations:
                state.stop_reason = f"iteration limit ({state.max_iterations}) reached"
                state.final_decision = state.stop_reason
                self.logger.warning(
                    f"Iteration limit reached ({state.max_iterations})",
                    "Stopping the collection loop and summarizing what was gathered.",
                    iteration=iteration,
                )
                return

    async def _safe_decide(self) -> Decision:
        try:
            return await self.decider.decide(self.state)
        except Exception as exc:  # noqa: BLE001 — a broken decision must not end the run
            self.state.record_error("decision_engine", f"{type(exc).__name__}: {exc}")
            self.logger.warning(
                "Decision step failed — finalizing safely",
                f"{type(exc).__name__}: {exc}",
            )
            return Decision(
                action="finalize",
                reasoning="The decision step errored; summarizing what has been collected.",
                author="guardrail",
                iteration=self.state.iteration_count,
            )

    # ── 4-5. SELECT TOOL + CALL TOOL ────────────────────────
    async def _call_tool(
        self, decision: Decision, ctx: ToolContext, iteration: int
    ) -> ToolResult:
        state = self.state
        state.current_step = "act"
        assert decision.tool and decision.tool_input

        tool = self.tools.get(decision.tool)
        if tool is None:
            state.record_error("tool_registry", f"unknown tool '{decision.tool}'")
            return ToolResult(tool=decision.tool, ok=False, error="unknown tool")

        self.logger.action(
            f"Calling {tool.display_name}",
            _describe_call(decision.tool_input),
            iteration=iteration,
            tool=tool.name,
            tool_input=decision.tool_input.describe(),
        )

        state.call_signatures.add(decision.tool_input.signature(decision.tool))
        result = await tool.run(decision.tool_input, ctx)
        state.absorb_result(result)
        return result

    # ── 6-7. OBSERVE + ANALYZE ──────────────────────────────
    def _observe(self, decision: Decision, result: ToolResult, iteration: int) -> None:
        state = self.state
        state.current_step = "observe"
        need_key = TOOL_TO_NEED.get(result.tool, "")

        observation = self.analyzer.analyze(state, result, need_key=need_key)

        state.tool_calls.append(
            ToolCallRecord(
                iteration=iteration,
                tool=result.tool,
                tool_input=decision.tool_input.describe() if decision.tool_input else {},
                reasoning=decision.reasoning,
                ok=result.ok,
                items_returned=result.count,
                new_items=observation.new_items,
                duplicates=observation.duplicates,
                latency_ms=result.latency_ms,
                providers_used=result.providers_used,
                providers_failed=result.providers_failed,
                simulated=result.simulated,
                error=result.error,
                note=result.note,
            )
        )

        # Provider-level degradation is reported but never fatal.
        for failure in result.providers_failed:
            state.record_error(
                f"provider:{failure['provider']}", failure.get("error", "unknown")
            )
        if result.providers_failed:
            failed = ", ".join(p["provider"] for p in result.providers_failed)
            survivors = ", ".join(result.providers_used) or "none"
            self.logger.warning(
                f"Provider(s) degraded: {failed} → continued with {survivors}",
                "; ".join(
                    f"{p['provider']}: {p.get('error') or p.get('note')}"
                    for p in result.providers_failed
                ),
                iteration=iteration,
            )

        if not result.ok and not result.items:
            self.logger.error(
                f"{result.tool} returned nothing usable",
                result.error or "no results; the agent will try a different angle.",
                iteration=iteration,
            )
        else:
            self.logger.observation(
                f"{result.count} item(s) from {result.tool}"
                + (f" — {observation.relevant_items} relevant" if result.count else ""),
                observation.summary,
                iteration=iteration,
                tool=result.tool,
                new_items=observation.new_items,
                duplicates=observation.duplicates,
                relevant=observation.relevant_items,
                yield_quality=observation.yield_quality,
                signals=observation.signals,
                top_titles=observation.top_titles,
                providers_used=result.providers_used,
                simulated=result.simulated,
                latency_ms=result.latency_ms,
            )

        # ANALYZE: state the conclusion drawn from this observation.
        state.current_step = "analyze"
        self.logger.thought(
            self._analysis_headline(observation),
            self._analysis_detail(observation),
            iteration=iteration,
            coverage=state.coverage(),
            needs_satisfied=[n.key for n in state.plan.needs if n.satisfied],
            relevant_total=len(state.relevant_findings()),
        )

    def _analysis_headline(self, obs: Any) -> str:
        state = self.state
        need = state.plan.need(TOOL_TO_NEED.get(obs.tool, ""))
        if obs.yield_quality == "failed":
            return "That tool is not usable right now"
        if need is not None and need.satisfied:
            return f"'{need.key}' need is now satisfied"
        if obs.yield_quality == "empty":
            return "Nothing came back for that query"
        if obs.yield_quality == "thin":
            return "Low relevance — that angle is weak"
        return f"{obs.relevant_items} relevant item(s) added"

    def _analysis_detail(self, obs: Any) -> str:
        state = self.state
        bits: list[str] = []
        outstanding = [n.key for n in state.plan.unsatisfied_required()]
        if outstanding:
            bits.append(f"still required: {', '.join(outstanding)}")
        else:
            bits.append("all required needs satisfied")
        relevant = len(state.relevant_findings())
        bits.append(f"{relevant}/{MIN_TOTAL_RELEVANT} relevant items toward a useful report")
        if obs.signals:
            bits.append(f"new signals to consider: {', '.join(obs.signals)}")
        if state.competitors:
            missing = state.uncovered_competitors()
            if missing:
                bits.append(f"no coverage yet for {', '.join(missing)}")
        return "; ".join(bits) + "."

    # ─────────────────────────────────────────────────────────
    # 9. FINAL INSIGHTS
    # ─────────────────────────────────────────────────────────
    async def _finalize(self) -> None:
        state = self.state
        state.current_step = "finalize"
        state.status = "finalizing"

        self.logger.insight(
            "Analyzing and prioritizing findings",
            f"{len(state.findings)} item(s) collected, "
            f"{len(state.relevant_findings())} above the relevance bar.",
        )

        try:
            insights = await self.insight_writer.generate(state)
        except Exception as exc:  # noqa: BLE001
            state.record_error("insight_generator", f"{type(exc).__name__}: {exc}")
            self.logger.warning(
                "Insight generation failed — falling back to the deterministic writer",
                f"{type(exc).__name__}: {exc}",
            )
            insights = await InsightGenerator(None).generate(state)

        state.final_insights = [i.to_dict() for i in insights]
        self._insights_text = "\n\n".join(i.render() for i in insights)

        counts = {
            HIGH: sum(1 for i in insights if i.priority == HIGH),
            MEDIUM: sum(1 for i in insights if i.priority == MEDIUM),
            LOW: sum(1 for i in insights if i.priority == LOW),
        }
        self.logger.insight(
            f"{len(insights)} prioritized insight(s): "
            f"{counts[HIGH]} high, {counts[MEDIUM]} medium, {counts[LOW]} low",
            "; ".join(
                f"[{i.priority}] {i.title[:80]}" for i in insights[:3]
            )
            or "no insights produced",
            priority_counts=counts,
        )

        # The insight call returns the executive summary in the same round trip.
        # Only fall back to a dedicated summary pass if it did not.
        try:
            prewritten = getattr(self.insight_writer, "executive_summary", "")
            if prewritten:
                state.summary = prewritten
            else:
                state.summary = await self.summary_writer.write(state, insights)
        except Exception as exc:  # noqa: BLE001
            state.record_error("summary_writer", f"{type(exc).__name__}: {exc}")
            state.summary = await SummaryWriter(None).write(state, insights)

        if not state.final_decision:
            state.final_decision = state.stop_reason or "Collection complete."

        limited = "iteration limit" in (state.stop_reason or "").lower()
        state.status = "completed_partial" if limited or not insights else "completed"
        state.current_step = "done"
        state.finished_at = datetime.now(UTC).isoformat(timespec="seconds")

        self.logger.done(
            "Task completed" if state.status == "completed" else "Task completed (partial)",
            state.summary.split("\n\n")[0] if state.summary else "",
            status=state.status,
            insights=len(insights),
            findings=len(state.findings),
            iterations=state.iteration_count,
        )

    # ─────────────────────────────────────────────────────────
    # Result assembly
    # ─────────────────────────────────────────────────────────
    def _result(self, started: float) -> AgentRunResult:
        state = self.state
        duration_ms = int((time.perf_counter() - started) * 1000)
        findings = sorted(state.findings, key=lambda f: f.relevance, reverse=True)

        return AgentRunResult(
            status=state.status,
            run_id=state.run_id,
            goal=state.user_goal,
            activity_log=self.logger.as_dicts(),
            tools_used=state.tools_used(),
            findings=[f.public() for f in findings],
            insights=state.final_insights,
            summary=state.summary,
            state=state.snapshot(),
            metrics={
                "duration_ms": duration_ms,
                "iterations": state.iteration_count,
                "max_iterations": state.max_iterations,
                "tool_calls": len(state.tool_calls),
                "tools_used": state.tools_used(),
                "findings_total": len(state.findings),
                "findings_relevant": len(state.relevant_findings()),
                "duplicates_suppressed": sum(c.duplicates for c in state.tool_calls),
                "insights": len(state.final_insights),
                "priority_counts": {
                    "HIGH": sum(1 for i in state.final_insights if i.get("priority") == HIGH),
                    "MEDIUM": sum(1 for i in state.final_insights if i.get("priority") == MEDIUM),
                    "LOW": sum(1 for i in state.final_insights if i.get("priority") == LOW),
                },
                "errors": len(state.errors),
                "reasoner": state.reasoner,
                "llm": self.llm.usage.to_dict(),
                "simulated_data_used": state.simulated_data_used,
                "coverage": state.coverage(),
                "competitor_coverage": state.competitor_coverage(),
                "signals_detected": sorted(state.detected_signals),
            },
            activity_text=self.logger.render(),
            insights_text=getattr(self, "_insights_text", ""),
        )


# ─────────────────────────────────────────────────────────────
# convenience
# ─────────────────────────────────────────────────────────────
def _describe_call(tool_input: ToolInput) -> str:
    bits = [f'query "{tool_input.query}"']
    if tool_input.keywords:
        bits.append(f"keywords {tool_input.keywords}")
    if tool_input.competitors:
        bits.append(f"companies {tool_input.competitors}")
    bits.append(f"last {tool_input.since_days} days, max {tool_input.limit}")
    return " · ".join(bits)


async def run_agent(
    goal: str,
    *,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    max_iterations: int | None = None,
    simulation_mode: bool | None = None,
    echo: bool = False,
    queue: asyncio.Queue | None = None,
) -> AgentRunResult:
    """One-shot helper used by the API, the CLI demo and the tests."""
    agent = InsightPulseAgent(echo=echo, queue=queue, simulation_mode=simulation_mode)
    return await agent.run(
        AgentRunRequest(
            goal=goal,
            keywords=keywords or [],
            competitors=competitors or [],
            max_iterations=max_iterations,
            simulation_mode=simulation_mode,
        )
    )
