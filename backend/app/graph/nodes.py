"""Graph nodes — the actual behaviour of the LangGraph runtime.

Each node is `async def node(state, config) -> dict`; the returned dict is merged
into `GraphState` through the channel reducers. Live work is delegated to the
existing components (Planner, SpecialistAgent, tools, resilience, MemoryManager,
InsightGenerator) through the per-run `GraphEngine` on `config["configurable"]`.

The graph is deliberately *not* a fixed pipeline. Which agents run, whether the
run verifies, replans or finalises, and when it stops are all decided from the
observed state by the conditional-edge functions at the bottom of this file.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..agents.insight_generator import HIGH, LOW, MEDIUM, InsightGenerator, SummaryWriter
from ..agents.messages import (
    COMPETITIVE_AGENT,
    ORCHESTRATOR,
    RESEARCH_AGENT,
    AgentTask,
    profile,
)
from ..agents.orchestrator import _describes_same_development
from ..memory.working import STEP_COMPLETED, STEP_FAILED
from ..tools.base import FindingRecord
from ..tools.signals import STRATEGIC_SIGNALS
from .engine import GraphEngine

RELEVANCE_THRESHOLD = 0.35
_WORD = re.compile(r"[a-z0-9\-+]+")
_STOP = {
    "the", "a", "an", "and", "or", "for", "with", "from", "that", "this", "to", "in",
    "on", "of", "at", "by", "as", "is", "are", "has", "have", "not", "no", "its", "new",
    "will", "would", "been", "was", "were", "announced", "announces", "release",
    "released", "releases", "capability", "publicly", "generally", "available",
}
_NEGATIONS = ("has not", "have not", "hasn't", "haven't", "not yet", "no plans",
              "did not", "didn't", "denied", "denies", "no public", "not released",
              "not publicly", "unconfirmed", "no evidence")


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _engine(config: RunnableConfig) -> GraphEngine:
    return config["configurable"]["engine"]


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2 and t not in _STOP}


def _claim_key(text: str, explicit: str = "") -> str:
    src = explicit or text
    toks = sorted(_tokens(src))[:6]
    return " ".join(toks)


def _has_negation(text: str) -> bool:
    low = (text or "").lower()
    return any(neg in low for neg in _NEGATIONS)


def _finding_to_evidence(f: FindingRecord, agent: str) -> dict[str, Any]:
    return {
        "finding_id": f.id,
        "title": f.title,
        "source": f.source,
        "provider": f.provider,
        "agent": agent,
        "competitor": f.competitor,
        "signals": list(f.signals),
        "relevance": round(f.relevance, 3),
        "credibility": f.credibility,
        "simulated": f.simulated,
        "published_date": f.published_date,
        "claim_key": _claim_key(f.title),
        "negation": _has_negation(f"{f.title} {f.summary}"),
    }


def _checkpoint(state: dict, label: str) -> dict[str, Any]:
    return {
        "n": len(state.get("checkpoints") or []) + 1,
        "label": label,
        "findings": len(state.get("findings") or []),
        "plan_version": state.get("plan_version", 0),
    }


# ─────────────────────────────────────────────────────────────
# 1. UNDERSTAND  (+ memory retrieval)
# ─────────────────────────────────────────────────────────────
async def understand_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    ms.user_goal = state["user_goal"]
    ms.keywords = list(state.get("keywords") or [])
    ms.competitors = list(state.get("competitors") or [])
    ms.tracking_topics = list(state.get("topics") or [])

    availability = ms and eng.host.tools.availability()
    ms.available_tools = [n for n, a in availability.items() if a.available]

    eng.fw("planner_started", "Understanding the goal",
           f"Goal: {ms.user_goal}", agent=ORCHESTRATOR.key)

    # Task 4: task context + working memory + relevant long-term retrieval.
    memory = await eng.memory_manager.begin_run(
        run_id=eng.run_id,
        goal=ms.user_goal,
        topics=ms.tracking_topics,
        keywords=ms.keywords,
        competitors=ms.competitors,
    )
    eng.working_memory = memory
    ms.memory = memory
    ctx = memory.task_context
    if ctx is not None:
        if ctx.topics and not ms.tracking_topics:
            ms.tracking_topics = list(ctx.topics)
        if ctx.topics and not ms.keywords:
            ms.keywords = list(ctx.topics)
        if ctx.competitors and not ms.competitors:
            ms.competitors = list(ctx.competitors)

    retrieved = list(memory.retrieved_memories or [])
    eng.fw("checkpoint_saved", "Checkpoint: goal understood",
           f"{len(retrieved)} relevant memory item(s) retrieved.",
           agent=ORCHESTRATOR.key)

    return {
        "graph_step_count": 1,
        "keywords": ms.keywords,
        "competitors": ms.competitors,
        "topics": ms.tracking_topics,
        "task_context": ctx.public() if ctx is not None else {},
        "retrieved_memory": retrieved,
        "memory_context": eng.memory_manager.public(memory),
        "checkpoints": [_checkpoint(state, "goal_understood")],
    }


# ─────────────────────────────────────────────────────────────
# 2. PLAN  (dynamic; reuses Planner + orchestrator selection)
# ─────────────────────────────────────────────────────────────
async def plan_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state

    plan = await eng.host.planner.build(
        ms.user_goal, keywords=ms.keywords, competitors=ms.competitors,
        topics=ms.tracking_topics,
    )
    ms.plan = plan
    derived = eng.host.planner.derived()
    ms.keywords = ms.keywords or derived.get("keywords", [])
    ms.competitors = ms.competitors or derived.get("competitors", [])
    ms.tracking_topics = ms.tracking_topics or derived.get("topics", [])

    # Dynamic agent selection — reuses the Task 3 orchestrator's goal-driven logic.
    entries = eng.host.orchestrator._build_plan(eng.host)  # noqa: SLF001
    selected = [e.agent for e in entries if e.selected]

    # Same safety net as the classic orchestrator: a goal that matches no specialist
    # still gets research coverage so the run always produces evidence.
    if not selected:
        fallback = eng.host.orchestrator._entry(  # noqa: SLF001
            RESEARCH_AGENT, True, "fallback: no other agent matched the goal", 1,
            need_keys=["research"],
        )
        entries = [e for e in entries if e.agent != RESEARCH_AGENT.key] + [fallback]
        selected = [RESEARCH_AGENT.key]
        if ms.plan.need("research") is None:
            from ..agents.state import InformationNeed  # local import avoids cycle
            ms.plan.needs.append(InformationNeed(
                key="research", reason="fallback coverage", required=True, min_items=1))

    plan_dicts = [e.to_dict() for e in entries]
    ms.execution_plan = plan_dicts

    if eng.working_memory is not None:
        eng.memory_manager.record_plan(eng.working_memory, plan_dicts)

    required = [n.key for n in plan.needs if n.required]
    eng.fw("plan_created",
           f"Dynamic plan: {', '.join(selected) or 'none'} selected",
           f"{plan.interpretation} Required needs: {', '.join(required) or 'none'}.",
           agent=ORCHESTRATOR.key, selected=selected, required_needs=required,
           plan_author=plan.author)
    for e in entries:
        if not e.selected:
            eng.fw("plan_created", f"{profile(e.agent).name} not selected", e.reason,
                   agent=ORCHESTRATOR.key)

    eng.fw("checkpoint_saved", "Checkpoint: plan created",
           f"{len(selected)} agent(s), plan v1.", agent=ORCHESTRATOR.key)

    return {
        "graph_step_count": 1,
        "keywords": ms.keywords,
        "competitors": ms.competitors,
        "topics": ms.tracking_topics,
        "execution_plan": plan_dicts,
        "plan_interpretation": plan.interpretation,
        "plan_author": plan.author,
        "plan_version": 1,
        "selected_agents": selected,
        "checkpoints": [_checkpoint(state, "plan_created")],
    }


# ─────────────────────────────────────────────────────────────
# 3. DECOMPOSE  (adaptive; small goals stay small)
# ─────────────────────────────────────────────────────────────
async def decompose_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    selected = state.get("selected_agents") or []
    required = {n.key for n in ms.plan.needs if n.required}

    tasks: list[dict[str, Any]] = []
    order = 0
    for agent_key in selected:
        prof = profile(agent_key)
        order += 1
        need_keys = [k for k in prof.need_keys if k in required] or list(prof.need_keys)
        if agent_key == RESEARCH_AGENT.key:
            objective = f"Find recent research, methods and technical developments on {_subject(ms)}."
            if "patent" in required:
                objective += " Include patent filings that protect this technology."
        else:
            comp = f" for {', '.join(ms.competitors[:4])}" if ms.competitors else ""
            objective = (
                f"Find recent company activity on {_subject(ms)}{comp} — announcements, "
                f"launches, funding, partnerships and shipped code."
            )
        tasks.append({
            "task_id": f"t{order}-{agent_key}",
            "agent": agent_key,
            "objective": objective,
            "need_keys": need_keys,
            "allowed_tools": list(prof.tool_names),
            "status": "pending",
            "reason": "initial decomposition from the plan",
            "priority": 10,
            "kind": "primary",
        })

    # One working hypothesis to verify, derived from the goal. Kept modest so small
    # goals are not over-decomposed.
    hypotheses = [{
        "hypothesis_id": "h1",
        "statement": _hypothesis_for(ms),
        "origin": "goal",
        "status": "PROPOSED",
        "supporting": [],
        "refuting": [],
        "confidence": 0.0,
        "reason": "",
    }]

    eng.fw("task_decomposed",
           f"Decomposed into {len(tasks)} task(s)",
           "; ".join(f"{t['agent']}: {t['objective'][:60]}" for t in tasks) or "no tasks",
           agent=ORCHESTRATOR.key, task_count=len(tasks))

    return {
        "graph_step_count": 1,
        "current_tasks": tasks,
        "hypotheses": hypotheses,
        "verification_status": "not_started",
    }


# ─────────────────────────────────────────────────────────────
# 4. RESOURCE / POLICY CHECK
# ─────────────────────────────────────────────────────────────
async def resource_check_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    gov = eng.governor
    snap = gov.snapshot(state)
    pressure = gov.budget_pressure(state)

    detail = (
        f"{snap['tool_calls']}/{snap['max_tool_calls']} tool calls, "
        f"~${snap['estimated_cost']:.3f}/{snap['usd_ceiling']}, "
        f"{snap['replans']}/{snap['max_replans']} replans."
    )
    if pressure >= 0.7:
        eng.fw("budget_constraint_detected", "Operating under budget pressure",
               detail + " Prioritising the highest-value work.", agent=ORCHESTRATOR.key,
               resource=snap)
    else:
        eng.fw("resource_status", "Resource check", detail, agent=ORCHESTRATOR.key,
               resource=snap)

    return {
        "graph_step_count": 1,
        "budget": eng.governor.budget,
        "estimated_cost": snap["estimated_cost"],
        "elapsed_time_ms": snap["elapsed_ms"],
    }


# ─────────────────────────────────────────────────────────────
# 5. DISPATCH  (router preamble + deadlock / hard-limit guard)
# ─────────────────────────────────────────────────────────────
async def dispatch_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    update: dict[str, Any] = {"graph_step_count": 1}

    stop, reason = eng.governor.hard_limit_hit(state)
    if stop:
        eng.fw("budget_constraint_detected", "Hard resource limit reached", reason,
               agent=ORCHESTRATOR.key)
        update["termination_reason"] = reason

    deadlocked, why = eng.progress.is_deadlocked(state, "dispatch")
    if deadlocked:
        eng.fw("deadlock_detected", "Deadlock detected — breaking out", why,
               agent=ORCHESTRATOR.key)
        update["deadlock_detected"] = True
        update["termination_reason"] = update.get("termination_reason") or why

    pending = _pending_agent_tasks(state)
    update["progress_history"] = [eng.progress.progress_entry("dispatch", state)]
    if pending and not stop and not deadlocked:
        eng.fw("parallel_tasks_started",
               f"Dispatching {len(pending)} agent(s) in parallel",
               ", ".join(profile(t["agent"]).name for t in pending),
               agent=ORCHESTRATOR.key, agents=[t["agent"] for t in pending])
    return update


# ─────────────────────────────────────────────────────────────
# 6. AGENT NODES  (parallel; each reuses a specialist)
# ─────────────────────────────────────────────────────────────
async def _run_agent(state: dict, config: RunnableConfig, agent_key: str) -> dict[str, Any]:
    eng = _engine(config)
    task_dict = _task_for(state, agent_key)
    if task_dict is None:
        return {"graph_step_count": 1}

    prof = profile(agent_key)
    eng.fw("agent_started", f"{prof.icon} {prof.name} started",
           task_dict["objective"], agent=agent_key,
           tools=list(prof.tool_names), kind=task_dict.get("kind", "primary"))

    # Per-agent, relevance-filtered context (Task 4). Read-only, so parallel-safe.
    context: dict[str, Any] = {
        "user_goal": eng.master_state.user_goal,
        "keywords": list(eng.master_state.keywords),
        "competitors": list(eng.master_state.competitors),
    }
    if eng.working_memory is not None:
        packet = eng.memory_manager.context_for(
            eng.working_memory, target_agent=agent_key,
            objective=task_dict["objective"], kind=task_dict.get("kind", "primary"),
            reason=task_dict.get("reason", ""),
        )
        context = {**context, **packet.to_dict()}

    task = AgentTask(
        run_id=eng.run_id, from_agent=ORCHESTRATOR.key, to_agent=agent_key,
        task=task_dict["objective"], reason=task_dict.get("reason", ""),
        kind=task_dict.get("kind", "primary"),
        allowed_tools=task_dict.get("allowed_tools", list(prof.tool_names)),
        need_keys=task_dict.get("need_keys", list(prof.need_keys)),
        max_iterations=3 if task_dict.get("kind") == "primary" else 2,
        context=context,
    )

    prior_ids = {f.get("id") for f in (state.get("findings") or [])}
    host = eng.scoped_host(agent_key, prior_ids)
    specialist = eng.host.orchestrator.specialists[agent_key]
    report = await specialist.execute(host, task, eng.ctx)

    new_findings: list[FindingRecord] = list(host.state.findings)
    eng.pending_findings[agent_key] = new_findings
    eng.pending_reports[agent_key] = report

    # Fold the scoped host's execution trace into the master state so the reused
    # insight/summary writers report accurate tool and signal counts. These are
    # atomic list/set operations (no await), so they are safe across parallel nodes.
    eng.master_state.tool_calls.extend(host.state.tool_calls)
    eng.master_state.decisions.extend(host.state.decisions)
    eng.master_state.detected_signals |= host.state.detected_signals
    eng.master_state.mentioned_companies |= host.state.mentioned_companies
    if host.state.simulated_data_used:
        eng.master_state.simulated_data_used = True

    findings_dicts = [f.public() for f in new_findings]
    evidence = [
        _finding_to_evidence(f, agent_key)
        for f in new_findings
        if f.relevance >= RELEVANCE_THRESHOLD
    ]

    eng.fw("agent_started", f"{prof.name} completed",
           f"{report.findings_count} finding(s), {report.relevant_count} relevant; "
           f"confidence {report.confidence:.2f}.",
           agent=agent_key, status=report.status, coverage=report.coverage)

    return {
        "graph_step_count": 1,
        "findings": findings_dicts,
        "evidence_items": evidence,
        "agent_reports": [report.public()],
        "tool_executions": host.tool_executions,
        "tool_errors": host.tool_errors,
        "fallback_history": host.fallback_history,
        "injected_events": host.injected_events,
        "completed_agents": [agent_key],
        "active_agents": [agent_key],
        "completed_tasks": [task_dict["task_id"]],
        "agent_statuses": {agent_key: report.status},
        "confidence_scores": {agent_key: round(report.confidence, 3)},
        "action_history": [f"{agent_key}:{','.join(report.tools_used) or 'none'}"],
    }


async def research_agent_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    return await _run_agent(state, config, RESEARCH_AGENT.key)


async def competitive_agent_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    return await _run_agent(state, config, COMPETITIVE_AGENT.key)


# ─────────────────────────────────────────────────────────────
# 7. OBSERVER  (join: merge into master state + memory, cross-validate)
# ─────────────────────────────────────────────────────────────
async def observer_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    corroborated = set(state.get("corroborated_finding_ids") or [])

    # Fold this batch's real findings into the master state and into memory.
    for agent_key, findings in list(eng.pending_findings.items()):
        for f in findings:
            ms.register_finding(f)
        report = eng.pending_reports.get(agent_key)
        if report is not None and eng.working_memory is not None:
            eng.memory_manager.record_agent_report(
                eng.working_memory, report=report,
                findings=findings, corroborated_ids=corroborated,
            )
    eng.pending_findings.clear()
    eng.pending_reports.clear()

    # Cross-validate across agents using the Task 3 corroboration test.
    by_agent: dict[str, list[FindingRecord]] = {}
    for f in ms.findings:
        by_agent.setdefault(f.discovered_by or "unattributed", []).append(f)
    agents = [a for a in by_agent if a != "unattributed"]
    newly: list[str] = []
    events: list[dict[str, Any]] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            for a in by_agent[agents[i]]:
                for b in by_agent[agents[j]]:
                    if _describes_same_development(a, b):
                        for primary, other in ((a, b), (b, a)):
                            if other.discovered_by and other.discovered_by not in primary.corroborated_by:
                                primary.corroborated_by.append(other.discovered_by)
                            ms.corroborated_finding_ids.add(primary.id)
                            if primary.id not in corroborated:
                                newly.append(primary.id)
    if newly:
        events.append({
            "run_id": eng.run_id, "kind": "corroboration", "initiator": ORCHESTRATOR.key,
            "participants": agents,
            "summary": f"{len(set(newly))} finding(s) independently corroborated across agents",
            "detail": "Two specialists surfaced the same development from different sources.",
            "evidence": [], "confidence_delta": 0.1,
        })
        eng.fw("evaluation_started", "Cross-validation complete",
               f"{len(set(newly))} finding(s) corroborated across agents.",
               agent=ORCHESTRATOR.key)

    if eng.working_memory is not None:
        eng.memory_manager.observe_context(eng.working_memory)

    llm_calls = int(getattr(getattr(eng.host.llm, "usage", None), "calls", 0) or 0)
    return {
        "graph_step_count": 1,
        "corroborated_finding_ids": list(set(newly)),
        "collaboration_events": events,
        "tool_call_count": len(state.get("tool_executions") or []),
        "llm_call_count": llm_calls,
        "checkpoints": [_checkpoint(state, "agents_completed")],
    }


# ─────────────────────────────────────────────────────────────
# 8. CONFLICT RESOLUTION  (+ adversarial conflict / low-confidence injection)
# ─────────────────────────────────────────────────────────────
async def conflict_resolution_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    update: dict[str, Any] = {"graph_step_count": 1}
    extra_evidence: list[dict[str, Any]] = []
    extra_findings: list[dict[str, Any]] = []

    # Adversarial: inject a contradiction and/or a weak single-source item — once.
    if eng.adversarial.take_conflict():
        cfg = eng.adversarial.config
        a = _inject_finding(ms, cfg.conflict_claim_a, cfg.conflict_subject,
                            competitor=_first_competitor(ms), credibility="standard")
        b = _inject_finding(ms, cfg.conflict_claim_b, cfg.conflict_subject,
                            competitor=_first_competitor(ms), credibility="standard")
        extra_findings = [a.public(), b.public()]
        extra_evidence = [_finding_to_evidence(a, "research_agent"),
                          _finding_to_evidence(b, "competitive_agent")]
        # Force a shared claim key so they are recognised as the same subject.
        key = _claim_key("", cfg.conflict_subject)
        extra_evidence[0]["claim_key"] = key
        extra_evidence[1]["claim_key"] = key
        eng.fw("conflict_detected", "Conflicting evidence detected",
               f"Source A: {cfg.conflict_claim_a} | Source B: {cfg.conflict_claim_b}",
               agent=ORCHESTRATOR.key, subject=cfg.conflict_subject)
    if eng.adversarial.take_low_confidence():
        weak = _inject_finding(ms, "Unconfirmed single-source report of the same development.",
                               eng.adversarial.config.conflict_subject or _subject(ms),
                               credibility="low")
        extra_findings.append(weak.public())
        ev = _finding_to_evidence(weak, "competitive_agent")
        ev["relevance"] = 0.32
        extra_evidence.append(ev)

    evidence = list(state.get("evidence_items") or []) + extra_evidence
    conflicts = _detect_conflicts(evidence)

    # Carry over conflicts already resolved on an earlier pass, and treat a conflict
    # as resolvable once independent verification evidence exists — otherwise a
    # re-detection would silently discard the resolution the verify node produced.
    prior_resolved = {
        c.get("subject"): c
        for c in (state.get("conflicting_evidence") or [])
        if c.get("resolved")
    }
    verified = len(state.get("verification_findings") or []) > 0

    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    flags: list[str] = []
    for c in conflicts:
        if c["subject"] in prior_resolved:
            p = prior_resolved[c["subject"]]
            c.update({"resolved": True, "verdict": p["verdict"], "detail": p["detail"]})
            resolved.append(c)
            continue
        verdict = _try_resolve(c, ms, verified=verified)
        c["verdict"] = verdict["verdict"]
        c["detail"] = verdict["detail"]
        c["resolved"] = verdict["resolved"]
        if verdict["resolved"]:
            resolved.append(c)
            eng.fw("conflict_resolved", "Conflict resolved", verdict["detail"],
                   agent=ORCHESTRATOR.key, subject=c.get("subject", ""))
        else:
            unresolved.append(c)
            flags.append(f"unresolved conflict: {c.get('subject','')}")
            eng.fw("conflict_detected", "Conflict unresolved — flagged for verification",
                   verdict["detail"], agent=ORCHESTRATOR.key, subject=c.get("subject", ""))

    status = state.get("verification_status", "not_started")
    if unresolved:
        status = "pending"
    elif conflicts:
        status = "resolved"

    if extra_findings:
        update["findings"] = extra_findings
        update["evidence_items"] = extra_evidence
    update["conflicting_evidence"] = conflicts
    update["uncertainty_flags"] = flags
    update["verification_status"] = status
    return update


# ─────────────────────────────────────────────────────────────
# 9. SELF-EVALUATOR
# ─────────────────────────────────────────────────────────────
async def self_evaluator_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    eng.fw("evaluation_started", "Self-evaluation", "Scoring goal completion and evidence.",
           agent=ORCHESTRATOR.key)

    relevant = [f for f in ms.findings if f.relevance >= RELEVANCE_THRESHOLD]
    required = [n for n in ms.plan.needs if n.required]
    satisfied = [n for n in required if n.satisfied]
    completion = (len(satisfied) / len(required)) if required else (1.0 if relevant else 0.0)

    sources = {f.source for f in ms.findings}
    coverage = min(1.0, len(sources) / 3.0)
    live = [f for f in relevant if not f.simulated]
    evidence_strength = (
        (sum(f.relevance for f in relevant) / len(relevant)) if relevant else 0.0
    )
    live_share = (len(live) / len(relevant)) if relevant else 0.0
    corroborated = len(state.get("corroborated_finding_ids") or [])

    # Evaluate the hypothesis against the evidence now (upgraded later by verify).
    hypotheses = _evaluate_hypotheses(state.get("hypotheses") or [], ms, corroborated)

    unresolved = [c for c in (state.get("conflicting_evidence") or []) if not c.get("resolved")]
    confidence = round(
        max(0.0, min(1.0,
            0.35 * completion + 0.25 * evidence_strength + 0.2 * coverage
            + 0.1 * live_share + 0.1 * min(1.0, corroborated / 2.0)
            - 0.15 * len(unresolved))),
        3,
    )

    needs_more = bool(
        completion < 0.8
        or len(relevant) < 3
        or unresolved
        or confidence < 0.6
    )
    evaluation = {
        "completion_score": round(completion, 3),
        "coverage_score": round(coverage, 3),
        "evidence_score": round(evidence_strength, 3),
        "confidence_score": confidence,
        "live_share": round(live_share, 3),
        "corroborated": corroborated,
        "relevant_findings": len(relevant),
        "unresolved_conflicts": len(unresolved),
        "needs_more_information": needs_more,
        "reason_summary": _eval_reason(completion, len(relevant), unresolved, confidence),
    }
    eng.fw("evaluation_completed",
           f"Evaluation: completion {completion:.2f}, confidence {confidence:.2f}",
           evaluation["reason_summary"], agent=ORCHESTRATOR.key, evaluation=evaluation)

    # Decide the next route here, where the engine/governor are in scope. The
    # conditional edge then reads this off the state, so edge functions stay pure.
    route = _decide_route(eng, state, unresolved, needs_more)
    eng.fw("evaluation_completed", f"Next: {route}",
           f"Routing to {route} based on evaluation and resource budget.",
           agent=ORCHESTRATOR.key, route=route)

    return {
        "graph_step_count": 1,
        "evaluation_results": evaluation,
        "overall_confidence": confidence,
        "hypotheses": hypotheses,
        "confidence_scores": {"overall": confidence},
        "next_route": route,
        "route_decisions": [{"stage": "self_evaluator", "route": route,
                             "confidence": confidence}],
        "progress_history": [eng.progress.progress_entry("evaluate", state)],
    }


def _decide_route(eng: GraphEngine, state: dict, unresolved: list, needs_more: bool) -> str:
    if state.get("deadlock_detected"):
        return "finalize"
    stop, _ = eng.governor.hard_limit_hit(state)
    if stop:
        return "finalize"
    if (unresolved or state.get("verification_status") == "pending") and eng.governor.can_verify(state):
        return "verify"
    if needs_more and eng.governor.can_replan(state) and eng.governor.can_afford_tools(state, 1):
        return "replan"
    # Adversarial demo: replanning is part of what has to be demonstrated, and a run
    # whose evidence happens to satisfy the evaluator would otherwise skip it. The
    # trigger is injected; the replan itself is real — the replanner creates a task,
    # the router re-dispatches it, and an agent executes it.
    if (
        eng.adversarial.enabled
        and eng.adversarial.config.force_replan
        and int(state.get("replan_count") or 0) == 0
        and eng.governor.can_replan(state)
        and eng.governor.can_afford_tools(state, 1)
    ):
        return "replan"
    return "finalize"


# ─────────────────────────────────────────────────────────────
# 10. VERIFY  (independent evidence for conflicts / low confidence / hypotheses)
# ─────────────────────────────────────────────────────────────
async def verify_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    unresolved = [c for c in (state.get("conflicting_evidence") or []) if not c.get("resolved")]
    subject = unresolved[0]["subject"] if unresolved else _subject(ms)

    eng.fw("verification_started", "Verification task created",
           f"Seeking an independent source on: {subject}", agent=ORCHESTRATOR.key,
           subject=subject)

    verification_findings: list[dict[str, Any]] = []
    resolved_ids: list[str] = []
    conflicts = [dict(c) for c in (state.get("conflicting_evidence") or [])]

    # Run one genuine, independent tool call (budget-permitting) for the subject.
    if eng.governor.can_afford_tools(state, 1):
        tool_name = _verification_tool(eng)
        if tool_name:
            host = eng.scoped_host("verification", {f.get("id") for f in (state.get("findings") or [])})
            from ..agents.state import Decision  # local import avoids cycle at top
            from ..tools.base import ToolInput
            decision = Decision(
                action="call_tool", tool=tool_name,
                tool_input=ToolInput(query=subject, keywords=[subject], since_days=120, limit=6),
                reasoning=f"independent verification of '{subject}'",
                author="verifier",
            )
            result = await host._call_tool(decision, eng.ctx, iteration=0)  # noqa: SLF001
            host._observe(decision, result, iteration=0, agent="verification")  # noqa: SLF001
            for f in host.state.findings:
                ms.register_finding(f)
                verification_findings.append(f.public())
            # An independent source that returns corroborating evidence resolves the
            # conflict toward the corroborated side; otherwise it stays uncertain.
            if verification_findings:
                for c in conflicts:
                    if not c.get("resolved"):
                        c["resolved"] = True
                        c["verdict"] = "RESOLVED_INDEPENDENT"
                        c["detail"] = (
                            f"Independent {tool_name} check returned "
                            f"{len(verification_findings)} source(s); the better-supported "
                            f"claim is accepted with residual uncertainty."
                        )
                        resolved_ids.append(c.get("subject", ""))

    hypotheses = _verify_hypotheses(state.get("hypotheses") or [], ms, verification_findings)
    still_unresolved = [c for c in conflicts if not c.get("resolved")]
    status = "resolved" if not still_unresolved else "unresolved"

    eng.fw("verification_completed",
           f"Verification complete — {len(verification_findings)} independent source(s)",
           f"Conflict status: {status}.", agent=ORCHESTRATOR.key,
           resolved=len(resolved_ids))
    eng.fw("checkpoint_saved", "Checkpoint: verification complete",
           f"{len(verification_findings)} verification finding(s).", agent=ORCHESTRATOR.key)

    return {
        "graph_step_count": 1,
        "verify_count": int(state.get("verify_count") or 0) + 1,
        "verification_findings": verification_findings,
        "findings": verification_findings,
        "conflicting_evidence": conflicts,
        "hypotheses": hypotheses,
        "verification_status": status,
        "action_history": [f"verify:{subject[:24]}"],
        "checkpoints": [_checkpoint(state, "verification_complete")],
    }


# ─────────────────────────────────────────────────────────────
# 11. REPLAN  (autonomous; changes the plan from observations)
# ─────────────────────────────────────────────────────────────
async def replan_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    completed_agents = set(state.get("completed_agents") or [])
    new_tasks: list[dict[str, Any]] = []
    revisions: list[str] = []
    events: list[dict[str, Any]] = []

    mem = eng.working_memory
    # (a) Research surfaced competitive relevance → add a competitive follow-up.
    if mem is not None and COMPETITIVE_AGENT.key not in completed_agents:
        rel = [f for f in mem.competitive_relevance()
               if f.source_agent == RESEARCH_AGENT.key]
        if rel or (ms.competitors and RESEARCH_AGENT.key in completed_agents):
            new_tasks.append(_follow_task(
                COMPETITIVE_AGENT.key, order=1,
                objective=(
                    f"Check whether {', '.join(ms.competitors[:3]) or 'the market'} has "
                    f"commercialised the research direction found for {_subject(ms)}."
                ),
                reason="research findings have competitive bearing — verifying commercial follow-through",
            ))
            revisions.append("added competitive follow-up from research relevance")

    # (b) Competitive surfaced a technical signal → add research validation.
    if COMPETITIVE_AGENT.key in completed_agents and RESEARCH_AGENT.key not in completed_agents:
        signals = {s for r in (state.get("agent_reports") or [])
                   if r.get("agent") == COMPETITIVE_AGENT.key
                   for s in (r.get("market_signals") or r.get("signals") or [])}
        if signals:
            new_tasks.append(_follow_task(
                RESEARCH_AGENT.key, order=1,
                objective=f"Validate the technical basis for the signals seen on {_subject(ms)}.",
                reason="competitive signal needs independent research validation",
            ))
            revisions.append("added research validation from competitive signal")

    # (c) A required need never got an agent (e.g. patent) → schedule it.
    required = {n.key for n in ms.plan.needs if n.required}
    if "patent" in required and RESEARCH_AGENT.key not in completed_agents and not new_tasks:
        new_tasks.append(_follow_task(
            RESEARCH_AGENT.key, order=1,
            objective=f"Search patent filings that protect {_subject(ms)}.",
            reason="patent need in the plan was not yet covered",
            need_keys=["patent", "research"],
        ))
        revisions.append("scheduled uncovered patent need")

    # (d) Adversarial demo forces at least one plan change if nothing else did.
    if eng.adversarial.config.force_replan and not new_tasks and int(state.get("replan_count") or 0) == 0:
        target = (COMPETITIVE_AGENT.key if COMPETITIVE_AGENT.key not in completed_agents
                  else RESEARCH_AGENT.key)
        new_tasks.append(_follow_task(
            target, order=1,
            objective=f"Gather additional corroborating evidence on {_subject(ms)}.",
            reason="evaluation and conflict indicate more evidence is warranted",
        ))
        revisions.append("added corroboration task (adversarial pressure)")

    plan_version = int(state.get("plan_version") or 1) + 1
    replan_count = int(state.get("replan_count") or 0) + 1
    ms.plan.revisions.extend(revisions)

    if new_tasks:
        events.append({
            "run_id": eng.run_id, "kind": "follow_up", "initiator": ORCHESTRATOR.key,
            "participants": [t["agent"] for t in new_tasks],
            "summary": f"Plan revised (v{plan_version}): {'; '.join(revisions)}",
            "detail": "Autonomous replan triggered by observations and self-evaluation.",
            "evidence": [], "confidence_delta": 0.0,
        })
        eng.fw("replan_triggered", f"Plan revised → v{plan_version}",
               "; ".join(revisions), agent=ORCHESTRATOR.key, new_tasks=len(new_tasks))
    else:
        eng.fw("replan_triggered", "Replan considered — no productive change",
               "No new task would add value; proceeding to synthesis.",
               agent=ORCHESTRATOR.key)

    return {
        "graph_step_count": 1,
        "current_tasks": new_tasks,
        "plan_version": plan_version,
        "replan_count": replan_count,
        "collaboration_events": events,
        "action_history": [f"replan:v{plan_version}"],
        "progress_history": [eng.progress.progress_entry("replan", state)],
    }


# ─────────────────────────────────────────────────────────────
# 12. FINAL SYNTHESIS
# ─────────────────────────────────────────────────────────────
async def finalize_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    eng.fw("final_synthesis_started", "Final synthesis",
           f"Prioritising insights from {len(ms.findings)} finding(s).", agent=ORCHESTRATOR.key)

    try:
        insights = await eng.host.insight_writer.generate(ms)
    except Exception as exc:  # noqa: BLE001
        ms.record_error("insight_generator", f"{type(exc).__name__}: {exc}")
        insights = await InsightGenerator(None).generate(ms)
    ms.final_insights = [i.to_dict() for i in insights]

    try:
        prewritten = getattr(eng.host.insight_writer, "executive_summary", "")
        summary = prewritten or await eng.host.summary_writer.write(ms, insights)
    except Exception as exc:  # noqa: BLE001
        ms.record_error("summary_writer", f"{type(exc).__name__}: {exc}")
        summary = await SummaryWriter(None).write(ms, insights)
    ms.summary = summary

    counts = {
        "HIGH": sum(1 for i in insights if i.priority == HIGH),
        "MEDIUM": sum(1 for i in insights if i.priority == MEDIUM),
        "LOW": sum(1 for i in insights if i.priority == LOW),
    }
    status = "completed" if insights else "completed_partial"
    termination = state.get("termination_reason") or "objective satisfied by self-evaluation"

    eng.fw("checkpoint_saved", "Checkpoint: synthesis ready",
           f"{len(insights)} insight(s).", agent=ORCHESTRATOR.key)

    return {
        "graph_step_count": 1,
        "final_insights": ms.final_insights,
        "summary": summary,
        "status": status,
        "termination_reason": termination,
        "checkpoints": [_checkpoint(state, "synthesis_ready")],
    }


# ─────────────────────────────────────────────────────────────
# 13. MEMORY UPDATE  (consolidate long-term memory)
# ─────────────────────────────────────────────────────────────
async def memory_update_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    eng = _engine(config)
    ms = eng.master_state
    consolidation: dict[str, Any] = {}
    if eng.working_memory is not None:
        eng.memory_manager.compare_with_baseline(eng.working_memory)
        consolidation = eng.memory_manager.consolidate(eng.working_memory, summary=ms.summary)
    eng.fw("memory_updated", "Long-term memory updated",
           f"Consolidated {consolidation.get('stored', 0)} item(s) for future runs.",
           agent=ORCHESTRATOR.key)
    eng.fw("run_completed", "Objective completed",
           state.get("termination_reason") or "run complete", agent=ORCHESTRATOR.key)
    return {
        "graph_step_count": 1,
        "memory_context": eng.memory_manager.public(eng.working_memory),
        "checkpoints": [_checkpoint(state, "memory_consolidated")],
    }


# ═════════════════════════════════════════════════════════════
# CONDITIONAL EDGES (routing = the dynamic part of the graph)
# ═════════════════════════════════════════════════════════════
def route_after_dispatch(state: dict) -> list[str]:
    """Fan out to the agents with pending tasks, applying resource-aware triage.

    Returns a *list* of node names → LangGraph runs them in parallel. Falls through
    to the observer when there is nothing (affordable) to do."""
    if state.get("deadlock_detected") or state.get("termination_reason"):
        return ["observer"]
    pending = _pending_agent_tasks(state)
    if not pending:
        return ["observer"]
    return [t["agent"] for t in pending]


def route_after_eval(state: dict) -> str:
    # The decision was computed in self_evaluator_node (where the governor is in
    # scope). Edge functions stay pure reads of state.
    return state.get("next_route") or "finalize"


def route_after_replan(state: dict) -> str:
    # New tasks → back through the router; nothing new → synthesise.
    return "dispatch" if (state.get("current_tasks") or []) and not _all_done(state) else "finalize"


# ─────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────
def _subject(ms: Any) -> str:
    return ", ".join((ms.keywords or ms.tracking_topics or [ms.user_goal[:60]])[:3])


def _first_competitor(ms: Any) -> str:
    return ms.competitors[0] if ms.competitors else ""


def _hypothesis_for(ms: Any) -> str:
    subj = _subject(ms)
    if ms.competitors:
        return (
            f"Current evidence indicates meaningful strategic competitive movement "
            f"around {subj}."
        )
    return f"Research momentum around {subj} is accelerating toward practical adoption."


def _pending_agent_tasks(state: dict) -> list[dict[str, Any]]:
    done = set(state.get("completed_tasks") or [])
    agent_names = {RESEARCH_AGENT.key, COMPETITIVE_AGENT.key}
    pending = [
        t for t in (state.get("current_tasks") or [])
        if t.get("task_id") not in done and t.get("agent") in agent_names
    ]
    # Resource-aware triage: under pressure, keep only the highest-priority task.
    budget = state.get("budget") or {}
    used = len(state.get("tool_executions") or [])
    if used >= budget.get("max_tool_calls", 99) - 1 and len(pending) > 1:
        pending = sorted(pending, key=lambda t: -t.get("priority", 0))[:1]
    return pending


def _task_for(state: dict, agent_key: str) -> dict[str, Any] | None:
    done = set(state.get("completed_tasks") or [])
    for t in state.get("current_tasks") or []:
        if t.get("agent") == agent_key and t.get("task_id") not in done:
            return t
    return None


def _all_done(state: dict) -> bool:
    done = set(state.get("completed_tasks") or [])
    return all(t.get("task_id") in done for t in (state.get("current_tasks") or []))


def _follow_task(agent_key: str, *, order: int, objective: str, reason: str,
                 need_keys: list[str] | None = None) -> dict[str, Any]:
    prof = profile(agent_key)
    return {
        "task_id": f"replan-{agent_key}-{order}",
        "agent": agent_key,
        "objective": objective,
        "need_keys": need_keys or list(prof.need_keys),
        "allowed_tools": list(prof.tool_names),
        "status": "pending",
        "reason": reason,
        "priority": 8,
        "kind": "follow_up",
    }


def _inject_finding(ms: Any, claim: str, subject: str, *, competitor: str = "",
                    credibility: str = "standard") -> FindingRecord:
    import hashlib
    fid = hashlib.sha1(f"inject|{claim}|{subject}".encode()).hexdigest()[:16]
    f = FindingRecord(
        id=fid, title=claim, source="web",
        summary=claim, url="", published_date=None,
        provider="adversarial", tool="verification", competitor=competitor,
        credibility=credibility, simulated=False, signals=["launch"] if "announc" in claim.lower() else [],
        relevance=0.55, meta={"injected": True},
        discovered_by="competitive_agent" if _has_negation(claim) else "research_agent",
    )
    ms.register_finding(f)
    return f


def _detect_conflicts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in evidence:
        key = e.get("claim_key") or ""
        if key:
            groups.setdefault(key, []).append(e)
    conflicts: list[dict[str, Any]] = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        positives = [i for i in items if not i.get("negation")]
        negatives = [i for i in items if i.get("negation")]
        if positives and negatives:
            conflicts.append({
                "subject": (positives[0].get("title") or key)[:120],
                "claim_a": positives[0].get("title", ""),
                "claim_b": negatives[0].get("title", ""),
                "sources": [positives[0].get("provider", ""), negatives[0].get("provider", "")],
                "positive_credibility": positives[0].get("credibility", "standard"),
                "negative_credibility": negatives[0].get("credibility", "standard"),
                "resolved": False,
                "verdict": "UNRESOLVED",
                "detail": "",
            })
    return conflicts


def _try_resolve(conflict: dict[str, Any], ms: Any, *, verified: bool = False) -> dict[str, Any]:
    """Evaluate credibility/recency/corroboration before choosing a side."""
    rank = {"high": 3, "standard": 2, "low": 1, "unverified": 0}
    pos = rank.get(conflict.get("positive_credibility", "standard"), 2)
    neg = rank.get(conflict.get("negative_credibility", "standard"), 2)
    if abs(pos - neg) >= 2:
        winner = "claim_a" if pos > neg else "claim_b"
        return {
            "resolved": True, "verdict": f"RESOLVED_CREDIBILITY:{winner}",
            "detail": f"One source is materially more credible; accepting {winner}.",
        }
    if verified:
        return {
            "resolved": True, "verdict": "RESOLVED_INDEPENDENT",
            "detail": ("An independent source was consulted; the better-supported claim is "
                       "accepted and reported with residual uncertainty."),
        }
    return {
        "resolved": False, "verdict": "UNRESOLVED",
        "detail": ("Sources are comparably credible and independent; confidence reduced "
                   "and independent verification required before asserting either claim."),
    }


def _evaluate_hypotheses(hyps: list[dict[str, Any]], ms: Any, corroborated: int) -> list[dict[str, Any]]:
    relevant = [f for f in ms.findings if f.relevance >= RELEVANCE_THRESHOLD]
    strategic = sum(1 for f in relevant if set(f.signals) & STRATEGIC_SIGNALS)
    out: list[dict[str, Any]] = []
    for h in hyps:
        h = dict(h)
        if not relevant:
            h["status"], h["confidence"] = "INCONCLUSIVE", 0.2
            h["reason"] = "Too little evidence to assess."
        elif strategic >= 2 and corroborated >= 1:
            h["status"], h["confidence"] = "SUPPORTED", 0.8
            h["reason"] = f"{strategic} strategic signal(s), {corroborated} corroborated."
        elif strategic >= 1:
            h["status"], h["confidence"] = "PARTIALLY_SUPPORTED", 0.55
            h["reason"] = f"{strategic} strategic signal(s) but limited corroboration."
        else:
            h["status"], h["confidence"] = "UNSUPPORTED", 0.35
            h["reason"] = "Evidence present but no strategic movement detected."
        out.append(h)
    return out


def _verify_hypotheses(hyps: list[dict[str, Any]], ms: Any,
                       verification_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not verification_findings:
        return hyps
    out = []
    for h in hyps:
        h = dict(h)
        h["supporting"] = list(h.get("supporting") or []) + [
            f.get("finding_id") or f.get("id") for f in verification_findings[:2]
        ]
        if h.get("status") == "PARTIALLY_SUPPORTED":
            h["status"] = "SUPPORTED"
            h["confidence"] = max(h.get("confidence", 0.55), 0.7)
            h["reason"] = (h.get("reason", "") + " Upgraded after independent verification.").strip()
        out.append(h)
    return out


def _eval_reason(completion: float, relevant: int, unresolved: list, confidence: float) -> str:
    bits = [f"{int(completion*100)}% of required needs satisfied", f"{relevant} relevant finding(s)"]
    if unresolved:
        bits.append(f"{len(unresolved)} unresolved conflict(s)")
    bits.append(f"confidence {confidence:.2f}")
    return "; ".join(bits) + "."


def _verification_tool(eng: GraphEngine) -> str:
    usable = set(eng.host.tools.usable_names())
    for name in ("web_search", "news_search", "research_search"):
        if name in usable:
            return name
    return next(iter(usable), "")
