"""Per-run execution engine — the live objects the graph nodes operate through.

The `GraphState` (checkpointed) holds only serialisable data. Everything alive — the
HTTP client, the LLM client, the activity logger wired to the SSE queue, the Task 4
memory manager and working memory, the source registry — lives here on the engine
and is passed to nodes via `config["configurable"]["engine"]`. It is never
checkpointed, so on resume a fresh engine is built while the durable state is
reloaded from the checkpointer.

`GraphHost` is the shim that lets the existing `SpecialistAgent.execute` run inside a
graph node. It subclasses `InsightPulseAgent` purely to reuse the battle-tested
`_observe` (relevance scoring, dedup, signal detection, plan-need updates) and swaps
in a *scoped* `AgentState` so two agents can run in parallel without racing. Its one
real override is `_call_tool`, which is where adversarial faults are injected and
where every tool execution is recorded as structured data.
"""

from __future__ import annotations

import time
from typing import Any

from ..agents.agent import InsightPulseAgent
from ..agents.state import AgentState, Decision
from ..memory import MemoryManager
from ..observability.instrument import traced_tool_call
from ..services.activity_logger import ActivityLogger, Phase
from ..sources.resilience import registry as resilience_registry
from ..tools.base import ToolContext, ToolInput, ToolResult
from .adversarial import AdversarialConfig, AdversarialController, FALLBACK_PROVIDERS
from .governor import ProgressMonitor, ResourceGovernor

# Framework event → activity-log phase. The frontend framework panel filters
# activity entries on data["fw_event"]; the phase keeps them coherent in the main
# reasoning trail too.
FW_EVENT_PHASE: dict[str, Phase] = {
    "planner_started": "plan",
    "plan_created": "plan",
    "task_decomposed": "plan",
    "parallel_tasks_started": "orchestration",
    "agent_started": "delegation",
    "tool_started": "action",
    "tool_succeeded": "observation",
    "tool_failed": "warning",
    "tool_timeout": "warning",
    "retry_started": "thought",
    "fallback_started": "thought",
    "fallback_succeeded": "observation",
    "evaluation_started": "thought",
    "evaluation_completed": "thought",
    "conflict_detected": "warning",
    "conflict_resolved": "observation",
    "verification_started": "orchestration",
    "verification_completed": "observation",
    "hypothesis_evaluated": "thought",
    "replan_triggered": "orchestration",
    "checkpoint_saved": "memory",
    "checkpoint_resumed": "memory",
    "budget_constraint_detected": "warning",
    "deadlock_detected": "warning",
    "resource_status": "thought",
    "final_synthesis_started": "insight",
    "run_completed": "done",
    "memory_updated": "consolidation",
}


class GraphEngine:
    """Owns the live per-run objects. One per graph execution."""

    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        logger: ActivityLogger,
        simulation_mode: bool,
        adversarial: AdversarialConfig | None = None,
        budget: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.logger = logger
        self.simulation_mode = simulation_mode
        # The master host accumulates all findings across nodes and owns the shared
        # plan, memory manager and the insight/summary writers reused at finalize.
        self.host = InsightPulseAgent(
            logger=logger, simulation_mode=simulation_mode
        )
        self.master_state: AgentState = self.host.state
        self.master_state.run_id = run_id
        self.memory_manager: MemoryManager = self.host.memory_manager
        self.working_memory: Any | None = None
        self.ctx: ToolContext | None = None
        self.adversarial = AdversarialController(adversarial or AdversarialConfig())
        self.governor = ResourceGovernor(budget or {}, started_at=time.monotonic())
        self.progress = ProgressMonitor()
        # Live objects handed from the (parallel) agent nodes to the (sequential)
        # observer node. Real FindingRecords and AgentReports cannot live in the
        # checkpointed state, so they ride on the engine instead. Keyed by agent.
        self.pending_findings: dict[str, list[Any]] = {}
        self.pending_reports: dict[str, Any] = {}

    # ── framework event emission (activity log + SSE + panel tag) ──
    def fw(
        self,
        fw_event: str,
        title: str,
        detail: str = "",
        *,
        agent: str = "",
        **data: Any,
    ) -> None:
        phase = FW_EVENT_PHASE.get(fw_event, "thought")
        payload = {"fw_event": fw_event, **data}
        if agent:
            payload["agent"] = agent
        try:
            self.logger.log(phase, title, detail, **payload)
        except Exception:  # noqa: BLE001 — logging must never break the graph
            pass

    def scoped_host(self, agent_key: str, prior_finding_ids: set[str]) -> "GraphHost":
        """A host with its own AgentState for race-free parallel execution.

        Seeded with the ids already known to the run so a re-dispatched agent
        (after a replan) reports only genuinely new findings rather than
        re-surfacing what the run already has.
        """
        scoped = AgentState()
        scoped.run_id = self.run_id
        scoped.user_goal = self.master_state.user_goal
        scoped.keywords = list(self.master_state.keywords)
        scoped.competitors = list(self.master_state.competitors)
        scoped.tracking_topics = list(self.master_state.tracking_topics)
        scoped.plan = self.master_state.plan
        scoped.max_iterations = self.master_state.max_iterations
        scoped.available_tools = list(self.master_state.available_tools)
        # Pre-seed dedup so duplicates across dispatches are suppressed.
        scoped.seen_finding_ids.update(prior_finding_ids)
        return GraphHost(self, agent_key=agent_key, scoped_state=scoped)


class GraphHost(InsightPulseAgent):
    """A specialist's execution host inside a graph node.

    Reuses `InsightPulseAgent._observe` verbatim; overrides `_call_tool` to inject
    adversarial faults and to record every tool execution as structured evidence of
    what happened (used for failure recovery, the resource picture and the UI).
    """

    def __init__(self, engine: GraphEngine, *, agent_key: str, scoped_state: AgentState) -> None:
        super().__init__(
            tools=engine.host.tools,
            llm=engine.host.llm,
            logger=engine.logger,
            simulation_mode=engine.simulation_mode,
        )
        self.state = scoped_state
        self.engine = engine
        self.agent_key = agent_key
        self.tool_executions: list[dict[str, Any]] = []
        self.tool_errors: list[dict[str, Any]] = []
        self.fallback_history: list[dict[str, Any]] = []
        self.injected_events: list[dict[str, Any]] = []

    # ── the one real override ───────────────────────────────
    @traced_tool_call
    async def _call_tool(self, decision: Decision, ctx: ToolContext, iteration: int) -> ToolResult:
        tool_name = decision.tool or ""
        fault = self.engine.adversarial.fault_for(tool_name)
        if fault is not None:
            return await self._injected_call(decision, ctx, iteration, fault)
        result = await super()._call_tool(decision, ctx, iteration)
        self._record_exec(tool_name, decision, result, attempt=1, fallback=False)
        return result

    async def _injected_call(
        self, decision: Decision, ctx: ToolContext, iteration: int, fault: Any
    ) -> ToolResult:
        """A real failure → retry → fallback → recovery sequence.

        The failure is injected, but the recovery is genuine: the fallback call is
        an ordinary tool execution that really returns data. We also toggle the
        production resilience breaker so the source-health board reflects the event.
        """
        tool_name = decision.tool or ""
        eng = self.engine
        primary = fault.provider or FALLBACK_PROVIDERS.get(tool_name, ("primary", "fallback"))[0]
        fb = fault.fallback_provider or FALLBACK_PROVIDERS.get(tool_name, ("primary", "fallback"))[1]

        eng.fw("tool_started", f"Calling {tool_name}", f"primary provider: {primary}",
               agent=self.agent_key, tool=tool_name, iteration=iteration)

        # Exercise the real resilience layer so the event is not purely synthetic.
        try:
            resilience_registry.record_failure(primary, "adversarial: injected failure")
        except Exception:  # noqa: BLE001
            pass

        eng.fw("tool_failed", f"{tool_name} failed on {primary}",
               "Injected fault: primary source unavailable.",
               agent=self.agent_key, tool=tool_name, provider=primary, iteration=iteration)
        self.tool_errors.append(
            {"tool": tool_name, "provider": primary, "error": "injected failure", "attempt": 1}
        )
        self.injected_events.append({"type": "tool_failed", "tool": tool_name, "provider": primary})
        self._record_exec(tool_name, decision, None, attempt=1, fallback=False,
                          status="failed", error="injected failure", source=primary)

        # Retry the primary once.
        eng.fw("retry_started", f"Retrying {tool_name}",
               "Transient failure — one retry before falling back.",
               agent=self.agent_key, tool=tool_name, iteration=iteration)
        if fault.timeout:
            eng.fw("tool_timeout", f"{tool_name} timed out on retry",
                   f"{primary} exceeded its deadline again.",
                   agent=self.agent_key, tool=tool_name, provider=primary, iteration=iteration)
            self.tool_errors.append(
                {"tool": tool_name, "provider": primary, "error": "timeout", "attempt": 2}
            )
            self.injected_events.append(
                {"type": "tool_timeout", "tool": tool_name, "provider": primary}
            )
            self._record_exec(tool_name, decision, None, attempt=2, fallback=False,
                              status="failed", error="timeout", source=primary)

        # Fall back to the alternate source. From here it is an ordinary, successful
        # tool call — the fault is one-shot and now marked recovered.
        eng.fw("fallback_started", f"Falling back to {fb}",
               f"Switching {tool_name} from {primary} to {fb}.",
               agent=self.agent_key, tool=tool_name, provider=fb, iteration=iteration)
        eng.adversarial.mark_recovered(tool_name)
        try:
            resilience_registry.set_forced_failure(primary, False)
        except Exception:  # noqa: BLE001
            pass

        result = await super()._call_tool(decision, ctx, iteration)
        attempt = 3 if fault.timeout else 2
        self.fallback_history.append(
            {"tool": tool_name, "from": primary, "to": fb, "recovered": True}
        )
        self.injected_events.append(
            {"type": "fallback_succeeded", "tool": tool_name, "from": primary, "to": fb}
        )
        eng.fw("fallback_succeeded", f"{tool_name} recovered via {fb}",
               f"Fallback returned {result.count} item(s).",
               agent=self.agent_key, tool=tool_name, provider=fb, iteration=iteration)
        self._record_exec(tool_name, decision, result, attempt=attempt, fallback=True,
                          status="ok", source=fb)
        return result

    def _record_exec(
        self,
        tool_name: str,
        decision: Decision,
        result: ToolResult | None,
        *,
        attempt: int,
        fallback: bool,
        status: str | None = None,
        error: str = "",
        source: str = "",
    ) -> None:
        ti: ToolInput | None = decision.tool_input
        if result is not None:
            status = status or ("ok" if result.count else "empty")
            self.tool_executions.append(
                {
                    "tool_name": tool_name,
                    "agent": self.agent_key,
                    "query": (ti.query if ti else ""),
                    "status": status,
                    "source": source or (result.providers_used[0] if result.providers_used else ""),
                    "providers_used": list(result.providers_used),
                    "providers_failed": [p.get("provider", "") for p in result.providers_failed],
                    "result_count": result.count,
                    "latency_ms": result.latency_ms,
                    "error": error or result.error,
                    "attempt": attempt,
                    "fallback_used": fallback,
                    "simulated": result.simulated,
                }
            )
        else:
            self.tool_executions.append(
                {
                    "tool_name": tool_name,
                    "agent": self.agent_key,
                    "query": (ti.query if ti else ""),
                    "status": status or "failed",
                    "source": source,
                    "providers_used": [],
                    "providers_failed": [source] if source else [],
                    "result_count": 0,
                    "latency_ms": 0,
                    "error": error,
                    "attempt": attempt,
                    "fallback_used": fallback,
                    "simulated": False,
                }
            )
