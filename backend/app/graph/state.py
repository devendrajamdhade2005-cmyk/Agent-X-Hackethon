"""Shared LangGraph state — the single source of truth for one graph run.

Everything here is plain JSON/msgpack-serialisable data (dicts, lists, numbers,
strings, bools). That is a hard requirement: the state is what LangGraph writes to
the checkpointer after every superstep, and what it reloads on resume. Live objects
(the HTTP client, the LLM client, the activity logger, the memory manager) never go
in the state — they live on the per-run `GraphEngine` passed through `config`.

Reducers matter because the graph fans out: the research and competitive agents run
in the same superstep and both write `findings`, `tool_executions`, `evidence_items`
and so on. Without a reducer LangGraph raises `InvalidUpdateError` on a concurrent
write; with the wrong reducer (e.g. last-writer-wins) one branch's results are lost.
Every channel two branches can touch has an explicit, order-independent reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# ─────────────────────────────────────────────────────────────
# Reducers
# ─────────────────────────────────────────────────────────────
def add_list(left: list | None, right: list | None) -> list:
    """Concatenate. Safe when either side is missing (initial superstep)."""
    return (left or []) + (right or [])


def merge_findings(left: list | None, right: list | None) -> list:
    """Union of finding dicts, de-duplicated on `id`, first occurrence wins.

    The two agents legitimately surface the same development; this keeps one copy
    while `corroborated_by` (set in the observer) records that both saw it.
    """
    out = list(left or [])
    seen = {f.get("id") for f in out if isinstance(f, dict)}
    for f in right or []:
        fid = f.get("id") if isinstance(f, dict) else None
        if fid is not None and fid in seen:
            continue
        seen.add(fid)
        out.append(f)
    return out


def merge_reports(left: list | None, right: list | None) -> list:
    """Agent reports keyed by `agent`; a later report replaces an earlier one.

    A follow-up report (sequential, post-replan) supersedes the primary; parallel
    reports are for different agents, so nothing is lost either way.
    """
    by_agent: dict[str, dict] = {}
    order: list[str] = []
    for r in [*(left or []), *(right or [])]:
        key = r.get("agent") or r.get("from_agent") or id(r)
        if key not in by_agent:
            order.append(key)
        by_agent[key] = r
    return [by_agent[k] for k in order]


def merge_unique(left: list | None, right: list | None) -> list:
    """Append items not already present, preserving order (e.g. agent keys)."""
    out = list(left or [])
    for x in right or []:
        if x not in out:
            out.append(x)
    return out


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Shallow dict merge; right wins on key collision (e.g. agent_statuses)."""
    out = dict(left or {})
    out.update(right or {})
    return out


# ─────────────────────────────────────────────────────────────
# Sub-record shapes (documentation; stored as plain dicts)
# ─────────────────────────────────────────────────────────────
class PlanTask(TypedDict, total=False):
    """One unit of work the planner/decomposer produced."""

    task_id: str
    agent: str            # research_agent | competitive_agent | verification
    objective: str
    need_keys: list[str]
    allowed_tools: list[str]
    status: str           # pending | running | completed | failed | skipped
    reason: str
    priority: int
    depends_on: list[str]
    kind: str             # primary | follow_up | verification


class ToolExecution(TypedDict, total=False):
    tool_name: str
    agent: str
    query: str
    status: str           # ok | empty | failed
    source: str
    providers_used: list[str]
    providers_failed: list[str]
    result_count: int
    latency_ms: int
    error: str
    attempt: int
    fallback_used: bool
    simulated: bool


class EvidenceItem(TypedDict, total=False):
    finding_id: str
    title: str
    source: str           # research | news | competitor | patent | web
    provider: str
    agent: str
    competitor: str
    signals: list[str]
    relevance: float
    credibility: str
    simulated: bool
    published_date: str | None
    claim_key: str        # normalised claim fingerprint for conflict detection


class Hypothesis(TypedDict, total=False):
    hypothesis_id: str
    statement: str
    origin: str
    status: str           # PROPOSED | SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | INCONCLUSIVE
    supporting: list[str]
    refuting: list[str]
    confidence: float
    reason: str


# ─────────────────────────────────────────────────────────────
# The graph state
# ─────────────────────────────────────────────────────────────
class GraphState(TypedDict, total=False):
    # ── identity / isolation ────────────────────────────────
    run_id: str
    thread_id: str
    user_goal: str
    keywords: list[str]
    competitors: list[str]
    topics: list[str]
    simulation_mode: bool

    # ── understanding + memory (Task 4) ─────────────────────
    task_context: dict[str, Any]
    memory_context: dict[str, Any]
    retrieved_memory: list[dict[str, Any]]

    # ── planning / decomposition ────────────────────────────
    execution_plan: list[dict[str, Any]]     # ExecutionPlanEntry dicts (UI-compatible)
    plan_interpretation: str
    plan_author: str
    plan_version: int
    # Replaced wholesale by the (sequential) decompose/replan nodes — no reducer.
    current_tasks: list[PlanTask]
    completed_tasks: Annotated[list[str], merge_unique]
    failed_tasks: Annotated[list[str], merge_unique]

    # ── agents ──────────────────────────────────────────────
    selected_agents: list[str]
    active_agents: Annotated[list[str], merge_unique]
    completed_agents: Annotated[list[str], merge_unique]
    agent_statuses: Annotated[dict[str, str], merge_dict]
    agent_reports: Annotated[list[dict[str, Any]], merge_reports]

    # ── findings + evidence ─────────────────────────────────
    findings: Annotated[list[dict[str, Any]], merge_findings]
    evidence_items: Annotated[list[EvidenceItem], add_list]
    corroborated_finding_ids: Annotated[list[str], merge_unique]
    collaboration_events: Annotated[list[dict[str, Any]], add_list]

    # ── tools + failure recovery ────────────────────────────
    tool_executions: Annotated[list[ToolExecution], add_list]
    tool_errors: Annotated[list[dict[str, Any]], add_list]
    fallback_history: Annotated[list[dict[str, Any]], add_list]

    # ── conflict / uncertainty / verification ───────────────
    conflicting_evidence: list[dict[str, Any]]
    hypotheses: list[Hypothesis]
    confidence_scores: Annotated[dict[str, float], merge_dict]
    uncertainty_flags: Annotated[list[str], merge_unique]
    verification_status: str          # not_started | pending | in_progress | resolved | unresolved
    verification_findings: Annotated[list[dict[str, Any]], add_list]

    # ── self-evaluation ─────────────────────────────────────
    evaluation_results: dict[str, Any]
    overall_confidence: float

    # ── resource governance ─────────────────────────────────
    budget: dict[str, Any]
    tool_call_count: int
    llm_call_count: int
    # Incremented by every node, including the two parallel agent branches in one
    # superstep, so it needs an additive reducer: each node contributes +1.
    graph_step_count: Annotated[int, operator.add]
    elapsed_time_ms: int
    estimated_cost: float

    # ── control / loop detection ────────────────────────────
    iteration_count: int
    replan_count: int
    verify_count: int
    progress_history: Annotated[list[dict[str, Any]], add_list]
    action_history: Annotated[list[str], add_list]
    route_decisions: Annotated[list[dict[str, Any]], add_list]
    deadlock_detected: bool
    termination_reason: str
    next_route: str          # decided by self_evaluator, read by the conditional edge

    # ── adversarial demo ────────────────────────────────────
    adversarial: dict[str, Any]
    injected_events: Annotated[list[dict[str, Any]], add_list]

    # ── output ──────────────────────────────────────────────
    final_insights: list[dict[str, Any]]
    summary: str
    status: str                       # running | completed | completed_partial | failed
    checkpoints: Annotated[list[dict[str, Any]], add_list]


Route = Literal["finalize", "verify", "replan", "research_agent", "competitive_agent"]


# ─────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────
DEFAULT_BUDGET = {
    "max_graph_steps": 60,
    "max_tool_calls": 14,
    "max_llm_calls": 40,
    "max_runtime_seconds": 120,
    "max_replans": 2,
    "max_verifications": 2,
    "max_concurrent_tasks": 3,
    "usd_ceiling": 0.50,
}


def new_state(
    *,
    run_id: str,
    thread_id: str,
    goal: str,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    topics: list[str] | None = None,
    simulation_mode: bool = False,
    budget: dict[str, Any] | None = None,
    adversarial: dict[str, Any] | None = None,
) -> GraphState:
    """A fresh, fully-initialised state. Every channel starts at a valid empty value
    so nodes never have to guard against missing keys."""
    merged_budget = {**DEFAULT_BUDGET, **(budget or {})}
    return GraphState(
        run_id=run_id,
        thread_id=thread_id,
        user_goal=goal,
        keywords=list(keywords or []),
        competitors=list(competitors or []),
        topics=list(topics or []),
        simulation_mode=simulation_mode,
        task_context={},
        memory_context={},
        retrieved_memory=[],
        execution_plan=[],
        plan_interpretation="",
        plan_author="",
        plan_version=0,
        current_tasks=[],
        completed_tasks=[],
        failed_tasks=[],
        selected_agents=[],
        active_agents=[],
        completed_agents=[],
        agent_statuses={},
        agent_reports=[],
        findings=[],
        evidence_items=[],
        corroborated_finding_ids=[],
        collaboration_events=[],
        tool_executions=[],
        tool_errors=[],
        fallback_history=[],
        conflicting_evidence=[],
        hypotheses=[],
        confidence_scores={},
        uncertainty_flags=[],
        verification_status="not_started",
        verification_findings=[],
        evaluation_results={},
        overall_confidence=0.0,
        budget=merged_budget,
        tool_call_count=0,
        llm_call_count=0,
        graph_step_count=0,
        elapsed_time_ms=0,
        estimated_cost=0.0,
        iteration_count=0,
        replan_count=0,
        verify_count=0,
        progress_history=[],
        action_history=[],
        route_decisions=[],
        deadlock_detected=False,
        termination_reason="",
        next_route="",
        adversarial=dict(adversarial or {}),
        injected_events=[],
        final_insights=[],
        summary="",
        status="running",
        checkpoints=[],
    )
