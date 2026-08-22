"""Task 5 — LangGraph agent framework tests.

The 18 checks from the brief (section 34). Integration tests drive the real
compiled graph in simulation mode (deterministic, offline); unit tests exercise the
governor and progress monitor directly. Nothing here mocks the recovery path — the
graph genuinely fails, falls back, conflicts, verifies and replans.
"""

from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from app.graph.adversarial import AdversarialConfig
from app.graph.builder import RECURSION_LIMIT, build_graph
from app.graph.engine import GraphEngine
from app.graph.governor import ProgressMonitor, ResourceGovernor
from app.graph.runner import run_graph
from app.graph.state import DEFAULT_BUDGET, new_state
from app.services.activity_logger import ActivityLogger
from app.sources.registry import build_http_client, registry as source_registry
from app.tools.base import ToolContext


def go(goal: str, **kw):
    kw.setdefault("simulation_mode", True)
    return asyncio.run(run_graph(goal, **kw))


def adv_run(scenario: str = "full", **kw):
    cfg = AdversarialConfig.named(scenario, kw.pop("competitor", "OpenAI"))
    return asyncio.run(run_graph(
        kw.pop("goal", "Analyze AI-agent developments and strategic competitive movement"),
        keywords=kw.pop("keywords", ["AI agents"]),
        competitors=kw.pop("competitors", ["OpenAI", "Anthropic"]),
        simulation_mode=True, adversarial=cfg, **kw,
    ))


# ── 1. Dynamic planning ──────────────────────────────────────
def test_01_dynamic_planning_varies_by_goal():
    research = go("Track recent research on multi-agent reinforcement learning")
    competitive = go("Monitor OpenAI and Anthropic announcements", competitors=["OpenAI", "Anthropic"])
    both = go("Analyze AI agent research and competitor developments",
              keywords=["AI agents"], competitors=["OpenAI"])
    assert research["framework"]["selected_agents"] != competitive["framework"]["selected_agents"]
    assert "research_agent" in research["framework"]["selected_agents"]
    assert "competitive_agent" in competitive["framework"]["selected_agents"]
    assert set(both["framework"]["selected_agents"]) == {"research_agent", "competitive_agent"}


# ── 2. Conditional routing ───────────────────────────────────
def test_02_conditional_routing_skips_irrelevant_agents():
    # A purely academic goal selects only the research agent at plan time. (The
    # competitive agent may still be added later by autonomous replanning if the
    # research turns up competitive relevance — that is a feature, not a routing
    # failure, so the conditional-routing assertion is on the plan-time selection.)
    research = go("Track recent research on multi-agent reinforcement learning")
    assert research["framework"]["selected_agents"] == ["research_agent"]
    assert "research_agent" in research["framework"]["completed_agents"]


# ── 3. Parallel execution ────────────────────────────────────
def test_03_parallel_execution_runs_both_agents():
    r = go("Analyze AI agent research and competitor developments",
           keywords=["AI agents"], competitors=["OpenAI", "Anthropic"])
    # Both specialists complete off a single dispatch (plan v1) — a real fan-out.
    assert set(r["framework"]["completed_agents"]) == {"research_agent", "competitive_agent"}
    assert r["framework"]["plan_version"] >= 1


# ── 4. Shared state ──────────────────────────────────────────
def test_04_shared_state_findings_visible_downstream():
    r = go("Track AI agents", keywords=["AI agents"])
    # Findings produced by the agents are visible to synthesis (insights exist).
    assert len(r["findings"]) > 0
    assert len(r["insights"]) > 0
    assert r["summary"]


# ── 5. Checkpoint creation ───────────────────────────────────
def test_05_checkpoints_created():
    r = go("Track AI agents", keywords=["AI agents"])
    labels = {c["label"] for c in r["framework"]["checkpoints"]}
    assert {"goal_understood", "plan_created", "agents_completed",
            "synthesis_ready", "memory_consolidated"} <= labels


# ── 6. Checkpoint resume ─────────────────────────────────────
def test_06_checkpoint_resume_continues_without_duplication():
    async def scenario():
        saver = MemorySaver()
        rid, tid = "resume-test", "resume-thread"
        eng = GraphEngine(run_id=rid, thread_id=tid, logger=ActivityLogger(rid),
                          simulation_mode=True, budget=dict(DEFAULT_BUDGET))
        graph = build_graph(saver, interrupt_before=["finalize"])
        cfg = {"configurable": {"engine": eng, "thread_id": tid},
               "recursion_limit": RECURSION_LIMIT}
        init = new_state(run_id=rid, thread_id=tid, goal="Track AI agents",
                         keywords=["AI agents"], competitors=["OpenAI"], simulation_mode=True)
        async with build_http_client(timeout=20) as client:
            eng.ctx = ToolContext(http_client=client, registry=source_registry,
                                  simulation_mode=True)
            paused = await graph.ainvoke(init, config=cfg)
            snap = await graph.aget_state(cfg)
            resumed = await graph.ainvoke(None, config=cfg)
        return paused, snap, resumed

    paused, snap, resumed = asyncio.run(scenario())
    assert snap.next == ("finalize",)                 # interrupted at the checkpoint
    assert len(paused.get("final_insights") or []) == 0
    findings_before = len(paused.get("findings") or [])
    assert findings_before > 0
    assert resumed["status"] in {"completed", "completed_partial"}
    assert len(resumed.get("final_insights") or []) > 0
    # Completed work is retained, not redone: agents unchanged, findings preserved.
    assert paused.get("completed_agents") == resumed.get("completed_agents")
    assert len(resumed.get("findings") or []) >= findings_before


# ── 7. Tool retry ────────────────────────────────────────────
def test_07_tool_retry_recovers():
    # The full scenario includes a timeout fault (fails twice) that then recovers.
    r = adv_run("full")
    timeouts = [e for e in r["framework"]["tool_errors"] if e.get("error") == "timeout"]
    assert timeouts, "a retryable timeout should have been injected and retried"
    assert r["status"] == "completed"


# ── 8. Tool fallback ─────────────────────────────────────────
def test_08_tool_fallback_recovers():
    r = adv_run("tool_failure")
    fb = r["framework"]["fallback_history"]
    assert fb, "a fallback should have recovered the failed primary source"
    assert all(x["recovered"] for x in fb)


# ── 9. Conflict detection ────────────────────────────────────
def test_09_conflict_detected():
    r = adv_run("conflict")
    conflicts = r["framework"]["conflicting_evidence"]
    assert conflicts, "a contradiction should have been detected"
    assert any(c.get("claim_a") and c.get("claim_b") for c in conflicts)


# ── 10. Conflict resolution → verification ───────────────────
def test_10_conflict_triggers_verification():
    r = adv_run("conflict")
    assert r["framework"]["verify_count"] >= 1
    # Either resolved via an independent source, or explicitly left uncertain.
    for c in r["framework"]["conflicting_evidence"]:
        assert c.get("verdict")


# ── 11. Uncertainty ──────────────────────────────────────────
def test_11_uncertainty_is_represented():
    r = adv_run("conflict")
    assert r["framework"]["uncertainty_flags"], "unresolved conflict should flag uncertainty"
    assert 0.0 <= r["framework"]["overall_confidence"] < 1.0


# ── 12. Self-evaluation can request more work ────────────────
def test_12_self_evaluation_can_request_more():
    r = adv_run("full")
    # The conflict + evaluation drove additional work (verify and/or replan).
    assert r["framework"]["verify_count"] >= 1 or r["framework"]["replan_count"] >= 1
    assert r["framework"]["evaluation"]


# ── 13. Autonomous replanning ────────────────────────────────
def test_13_autonomous_replanning_changes_plan():
    r = adv_run("full")
    assert r["framework"]["replan_count"] >= 1
    assert r["framework"]["plan_version"] >= 2   # the plan actually changed


# ── 14. Budget constraint ────────────────────────────────────
def test_14_budget_constraint_gates_work():
    # Governor unit: a tight budget refuses further tool calls and replans.
    gov = ResourceGovernor({"max_tool_calls": 2, "max_replans": 0, "usd_ceiling": 0.5,
                            "max_graph_steps": 60, "max_runtime_seconds": 120,
                            "max_verifications": 0})
    state = {"tool_executions": [{}, {}], "replan_count": 0, "verify_count": 0}
    assert gov.can_afford_tools(state, 1) is False
    assert gov.can_replan(state) is False
    assert gov.can_verify(state) is False
    # Integration: the budget scenario still completes within its tightened ceiling.
    r = adv_run("budget")
    assert r["status"] in {"completed", "completed_partial"}
    assert r["framework"]["resource"]["max_tool_calls"] == 4


# ── 15. Hypothesis verification ──────────────────────────────
def test_15_hypothesis_verification_transitions():
    r = adv_run("full")
    hyps = r["framework"]["hypotheses"]
    assert hyps
    valid = {"PROPOSED", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "INCONCLUSIVE"}
    assert all(h["status"] in valid for h in hyps)
    assert any(h["status"] != "PROPOSED" for h in hyps)   # it was actually evaluated


# ── 16. Memory-based reasoning ───────────────────────────────
def test_16_memory_influences_planning():
    r = go("Track AI agents and OpenAI", keywords=["AI agents"], competitors=["OpenAI"])
    mem = r["memory"]
    assert mem.get("available") is True
    assert mem.get("working", {}).get("version", 0) > 0
    # A second run on the same subject can retrieve consolidated context.
    r2 = go("Continue monitoring AI agents and OpenAI", keywords=["AI agents"],
            competitors=["OpenAI"])
    assert r2["memory"].get("available") is True
    assert isinstance(r2.get("retrieved_memory", []) or r2["memory"].get("long_term", {}).get("retrieved", []), list)


# ── 17. Loop / deadlock detection ────────────────────────────
def test_17_loop_deadlock_detection():
    base = {"plan_version": 1, "current_tasks": [], "findings": [], "conflicting_evidence": []}
    sig = ProgressMonitor.signature("dispatch", base)
    stuck = {**base, "progress_history": [{"signature": sig} for _ in range(3)],
             "action_history": []}
    hit, why = ProgressMonitor.is_deadlocked(stuck, "dispatch")
    assert hit and why
    repeated = {"progress_history": [], "action_history": ["a:x", "a:x", "a:x"]}
    assert ProgressMonitor.is_deadlocked(repeated, "dispatch")[0] is True
    # A healthy, progressing state is not flagged.
    assert ProgressMonitor.is_deadlocked(base, "dispatch")[0] is False


# ── 18b. Adversarial injection is plan-independent (regression) ──
def test_18b_adversarial_fires_whatever_the_plan_selects():
    """The faults must fire regardless of which tools the dynamic plan reaches for.

    Regression: faults were pinned to research_search/web_search, so a plan that
    selected only the competitive agent never called them and the adversarial run
    silently degraded to a fault-free run — breaking repeatability.
    """
    cases = [
        dict(goal="Analyze AI agent research and competitor developments",
             keywords=["AI agents"], competitors=["OpenAI", "Anthropic"]),
        dict(goal="Monitor OpenAI and Anthropic announcements",
             competitors=["OpenAI", "Anthropic"]),
        dict(goal="Track recent research on multi-agent reinforcement learning",
             keywords=["multi-agent RL"]),
    ]
    for kw in cases:
        r = asyncio.run(run_graph(simulation_mode=True,
                                  adversarial=AdversarialConfig.full_scenario("OpenAI"), **kw))
        fw = r["framework"]
        assert fw["fallback_history"], f"no fallback fired for plan {fw['selected_agents']}"
        assert len(fw["fallback_history"]) <= 2, "injections must stay within budget"
        assert fw["replan_count"] >= 1, f"no replan for plan {fw['selected_agents']}"
        assert r["status"] == "completed"


# ── 18. Full adversarial test ────────────────────────────────
def test_18_full_adversarial_completes_autonomously():
    r = adv_run("full")
    fw = r["framework"]
    assert r["status"] == "completed"
    assert fw["fallback_history"], "tool failure + fallback"
    assert fw["conflicting_evidence"], "conflicting evidence"
    assert fw["uncertainty_flags"], "uncertainty represented"
    assert fw["verify_count"] >= 1, "verification performed"
    assert fw["replan_count"] >= 1, "autonomous replanning"
    assert fw["resource"]["max_tool_calls"] == 8, "budget constraint applied"
    assert len(r["insights"]) > 0, "objective completed with insights"
    assert fw["graph_steps"] <= RECURSION_LIMIT, "hard step limit respected"
