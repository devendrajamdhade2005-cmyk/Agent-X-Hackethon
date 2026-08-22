"""High-level entry point for the LangGraph runtime.

`run_graph` is what the API route and the tests call. It owns the pieces that must
live for exactly one run: the activity logger (optionally wired to an SSE queue),
the HTTP client, the per-run engine and the checkpointer configuration. It invokes
the compiled graph on a stable `thread_id` and projects the final `GraphState` into
the same result shape the existing dashboard and report already consume, with an
added `framework` block describing the graph execution.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from ..agents.insight_generator import HIGH, LOW, MEDIUM
from ..config import settings
from ..services.activity_logger import ActivityLogger
from ..sources.registry import build_http_client, registry as source_registry
from ..tools.base import ToolContext
from .adversarial import AdversarialConfig
from .builder import RECURSION_LIMIT, build_graph
from .engine import GraphEngine
from .state import DEFAULT_BUDGET, new_state

# In-process checkpointer shared across invocations so a run can be resumed from a
# saved checkpoint within the same process. Durable cross-process persistence uses
# a SqliteSaver (see `make_checkpointer`).
_MEMORY_SAVER = MemorySaver()


def make_checkpointer(kind: str = "memory") -> Any:
    """Return a checkpointer. `memory` (default) is in-process and resumable;
    `sqlite` is durable across restarts."""
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

            from ..config import DATA_DIR

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # from_conn_string yields a context-managed saver; we keep it open for
            # the process lifetime deliberately (single-writer demo/durability).
            cm = SqliteSaver.from_conn_string(str(DATA_DIR / "graph_checkpoints.db"))
            return cm.__enter__()
        except Exception:  # noqa: BLE001 — fall back to in-process rather than fail
            return _MEMORY_SAVER
    return _MEMORY_SAVER


def _resolve_adversarial(adversarial: Any) -> AdversarialConfig | None:
    if adversarial is None:
        return None
    if isinstance(adversarial, AdversarialConfig):
        return adversarial
    if isinstance(adversarial, dict):
        return AdversarialConfig.from_dict(adversarial)
    return None


async def run_graph(
    goal: str,
    *,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    topics: list[str] | None = None,
    simulation_mode: bool | None = None,
    max_iterations: int | None = None,
    adversarial: Any = None,
    queue: Any = None,
    thread_id: str | None = None,
    checkpointer: Any = None,
    on_event: Any = None,
) -> dict[str, Any]:
    """Run the agent graph end to end and return a UI/report-compatible result."""
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    thread_id = thread_id or run_id
    sim = settings.simulation_mode if simulation_mode is None else simulation_mode

    adv = _resolve_adversarial(adversarial)
    budget = {**DEFAULT_BUDGET, **((adv.budget_override if adv else {}) or {})}

    logger = ActivityLogger(run_id, queue=queue, sink=on_event)
    logger.start(goal, run_id=run_id)

    engine = GraphEngine(
        run_id=run_id, thread_id=thread_id, logger=logger,
        simulation_mode=sim, adversarial=adv, budget=budget,
    )
    if max_iterations:
        engine.master_state.max_iterations = max(1, min(int(max_iterations), 25))

    graph = build_graph(checkpointer or _MEMORY_SAVER)
    config = {
        "configurable": {"engine": engine, "thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    initial = new_state(
        run_id=run_id, thread_id=thread_id, goal=goal,
        keywords=keywords, competitors=competitors, topics=topics,
        simulation_mode=sim, budget=budget,
        adversarial=(adv.to_dict() if adv else None),
    )

    final: dict[str, Any]
    try:
        async with build_http_client(timeout=settings.collect_timeout_seconds) as client:
            engine.ctx = ToolContext(
                http_client=client, registry=source_registry, simulation_mode=sim
            )
            final = await graph.ainvoke(initial, config=config)
    except Exception as exc:  # noqa: BLE001 — always return something usable
        logger.error("Graph execution failed",
                     f"{type(exc).__name__}: {exc}. Reporting on partial state.")
        final = dict(initial)
        final["status"] = "failed"
        final["termination_reason"] = f"{type(exc).__name__}: {exc}"

    return _assemble(final, engine, logger, started)


# ─────────────────────────────────────────────────────────────
# result projection
# ─────────────────────────────────────────────────────────────
def _assemble(
    state: dict, engine: GraphEngine, logger: ActivityLogger, started: float
) -> dict[str, Any]:
    duration_ms = int((time.perf_counter() - started) * 1000)
    ms = engine.master_state
    findings = sorted(state.get("findings") or [], key=lambda f: f.get("relevance", 0), reverse=True)
    insights = state.get("final_insights") or []
    gov = engine.governor.snapshot(state)

    framework = {
        "runtime": "langgraph",
        "graph_steps": state.get("graph_step_count", 0),
        "plan_version": state.get("plan_version", 0),
        "replan_count": state.get("replan_count", 0),
        "verify_count": state.get("verify_count", 0),
        "selected_agents": state.get("selected_agents") or [],
        "completed_agents": state.get("completed_agents") or [],
        "current_tasks": state.get("current_tasks") or [],
        "tool_executions": state.get("tool_executions") or [],
        "tool_errors": state.get("tool_errors") or [],
        "fallback_history": state.get("fallback_history") or [],
        "conflicting_evidence": state.get("conflicting_evidence") or [],
        "uncertainty_flags": state.get("uncertainty_flags") or [],
        "verification_status": state.get("verification_status", "not_started"),
        "verification_findings": state.get("verification_findings") or [],
        "hypotheses": state.get("hypotheses") or [],
        "evaluation": state.get("evaluation_results") or {},
        "overall_confidence": state.get("overall_confidence", 0.0),
        "confidence_scores": state.get("confidence_scores") or {},
        "checkpoints": state.get("checkpoints") or [],
        "route_decisions": state.get("route_decisions") or [],
        "progress_history": state.get("progress_history") or [],
        "deadlock_detected": state.get("deadlock_detected", False),
        "termination_reason": state.get("termination_reason", ""),
        "resource": gov,
        "adversarial": state.get("adversarial") or {},
        "injected_events": state.get("injected_events") or [],
    }

    return {
        "status": state.get("status", "completed"),
        "run_id": state.get("run_id", engine.run_id),
        "thread_id": state.get("thread_id", engine.thread_id),
        "goal": state.get("user_goal", ""),
        "activity_log": logger.as_dicts(),
        "tools_used": sorted({t.get("tool_name", "") for t in (state.get("tool_executions") or []) if t.get("tool_name")}),
        "findings": findings,
        "insights": insights,
        "summary": state.get("summary", ""),
        "execution_plan": state.get("execution_plan") or [],
        "agents": state.get("agent_reports") or [],
        "collaboration_events": state.get("collaboration_events") or [],
        "memory": state.get("memory_context") or {},
        "framework": framework,
        "state": {
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id"),
            "status": state.get("status"),
            "plan": ms.plan.to_dict(),
            "selected_agents": state.get("selected_agents") or [],
            "graph_step_count": state.get("graph_step_count", 0),
        },
        "metrics": {
            "runtime": "langgraph",
            "duration_ms": duration_ms,
            "graph_steps": state.get("graph_step_count", 0),
            "tool_calls": len(state.get("tool_executions") or []),
            "llm_calls": state.get("llm_call_count", 0),
            "tools_used": sorted({t.get("tool_name", "") for t in (state.get("tool_executions") or []) if t.get("tool_name")}),
            "findings_total": len(findings),
            "findings_relevant": sum(1 for f in findings if f.get("relevance", 0) >= 0.35),
            "insights": len(insights),
            "priority_counts": {
                "HIGH": sum(1 for i in insights if i.get("priority") == HIGH),
                "MEDIUM": sum(1 for i in insights if i.get("priority") == MEDIUM),
                "LOW": sum(1 for i in insights if i.get("priority") == LOW),
            },
            "replans": state.get("replan_count", 0),
            "verifications": state.get("verify_count", 0),
            "parallel_agents": len(state.get("completed_agents") or []),
            "overall_confidence": state.get("overall_confidence", 0.0),
            "estimated_cost": gov.get("estimated_cost", 0.0),
            "simulated_data_used": ms.simulated_data_used,
            "agents_used": state.get("completed_agents") or [],
            "collaboration_events": len(state.get("collaboration_events") or []),
            "adversarial": bool((state.get("adversarial") or {}).get("enabled")),
        },
        "activity_text": logger.render(),
        "insights_text": "\n\n".join(
            f"[{i.get('priority')}] {i.get('title','')}" for i in insights[:5]
        ),
    }
