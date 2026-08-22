"""Task 3 acceptance tests — multi-agent architecture.

Covers the seven required scenarios and, more importantly, the properties that
distinguish real orchestration from three functions wearing agent costumes:

  * specialisation is *structural* — an agent physically cannot call another
    agent's tools
  * the orchestrator selects agents from the goal, and records why it rejected one
  * one agent's result can create work for another (observation-driven delegation)
  * overlapping evidence is merged with confidence raised and sources preserved
  * a failing agent does not stop the run or block the other agent's findings
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.agent import run_agent
from app.agents.messages import (
    COMPETITIVE_AGENT,
    ORCHESTRATOR,
    RESEARCH_AGENT,
    AgentReport,
    AgentTask,
    CollaborationEvent,
)
from app.agents.orchestrator import IntelligenceOrchestrator, _describes_same_development
from app.tools.base import FindingRecord
from app.tools.registry import tool_registry


def go(goal: str, **kw):
    return asyncio.run(run_agent(goal, simulation_mode=True, **kw))


def selected(result) -> list[str]:
    return [e["agent"] for e in result.execution_plan if e["selected"]]


def skipped(result) -> list[str]:
    return [e["agent"] for e in result.execution_plan if not e["selected"]]


def agent_card(result, key: str) -> dict | None:
    return next((a for a in result.agents if a["agent"] == key), None)


# ─────────────────────────────────────────────────────────────
# Structural specialisation
# ─────────────────────────────────────────────────────────────
def test_agents_own_disjoint_tool_sets():
    research = set(RESEARCH_AGENT.tool_names)
    competitive = set(COMPETITIVE_AGENT.tool_names)
    assert research and competitive
    assert not (research & competitive), "specialists must not share tools"
    # every tool in the product belongs to exactly one specialist
    assert research | competitive == set(tool_registry.names())


def test_orchestrator_owns_no_tools():
    assert ORCHESTRATOR.tool_names == ()
    assert ORCHESTRATOR.need_keys == ()


def test_scoped_engine_cannot_see_other_agents_tools():
    """Specialisation is enforced by the decision engine, not by convention."""
    from app.agents.decision_engine import DecisionEngine
    from app.agents.planner import Planner
    from app.agents.state import AgentState
    from app.tools.registry import ToolRegistry

    state = AgentState(user_goal="Track AI agents and monitor OpenAI")
    planner = Planner(None)
    state.plan = asyncio.run(planner.build(state.user_goal, competitors=["OpenAI"]))
    state.keywords = planner.derived()["keywords"]
    state.competitors = ["OpenAI"]

    engine = DecisionEngine(ToolRegistry(), None, allowed_tools=set(RESEARCH_AGENT.tool_names))
    tools = {c.tool for c in engine.candidates(state)}
    assert tools <= set(RESEARCH_AGENT.tool_names), f"research scope leaked: {tools}"
    assert "web_search" not in tools
    assert "competitor_search" not in tools


# ─────────────────────────────────────────────────────────────
# 1. Research-only goal
# ─────────────────────────────────────────────────────────────
def test_research_only_goal_skips_the_competitive_agent():
    r = go("Track recent research on multi-agent reinforcement learning")
    assert RESEARCH_AGENT.key in selected(r)
    assert COMPETITIVE_AGENT.key in skipped(r), (
        f"a purely academic goal must not deploy competitor monitoring: {r.execution_plan}")

    # and the orchestrator must explain the rejection
    reason = next(e["reason"] for e in r.execution_plan if e["agent"] == COMPETITIVE_AGENT.key)
    assert reason, "a skipped agent needs a recorded reason"

    assert "web_search" not in r.tools_used
    assert "competitor_search" not in r.tools_used


def test_patent_goal_uses_research_agent_without_web_search():
    r = go("Monitor generative AI patents", keywords=["generative AI"])
    assert RESEARCH_AGENT.key in selected(r)
    assert COMPETITIVE_AGENT.key in skipped(r)
    assert "patent_search" in r.tools_used
    assert "web_search" not in r.tools_used, "Tavily must not be called for an IP-only goal"


# ─────────────────────────────────────────────────────────────
# 2. Competitor-only goal
# ─────────────────────────────────────────────────────────────
def test_competitor_goal_selects_the_competitive_agent():
    r = go("Track recent announcements from OpenAI", keywords=["AI agents"],
           competitors=["OpenAI"])
    assert COMPETITIVE_AGENT.key in selected(r)
    card = agent_card(r, COMPETITIVE_AGENT.key)
    assert card is not None
    assert card["findings_count"] > 0
    assert set(card["tools_used"]) <= set(COMPETITIVE_AGENT.tool_names)


def test_tavily_is_used_by_the_competitive_agent_when_justified():
    r = go("Track recent announcements from OpenAI", keywords=["AI agents"],
           competitors=["OpenAI"])
    card = agent_card(r, COMPETITIVE_AGENT.key)
    assert "web_search" in card["tools_used"], (
        "current corporate announcements justify live web search")


def test_tavily_is_not_used_for_an_academic_goal():
    r = go("Track scientific research on multi-agent reinforcement learning")
    assert "web_search" not in r.tools_used


# ─────────────────────────────────────────────────────────────
# 3. Mixed goal
# ─────────────────────────────────────────────────────────────
def test_mixed_goal_deploys_both_specialists():
    r = go("Track AI agent technology, OpenAI, Anthropic, research and competitor developments",
           keywords=["AI agents"], competitors=["OpenAI", "Anthropic"])
    assert set(selected(r)) == {RESEARCH_AGENT.key, COMPETITIVE_AGENT.key}

    for key in (RESEARCH_AGENT.key, COMPETITIVE_AGENT.key):
        card = agent_card(r, key)
        assert card is not None, f"{key} must report"
        assert card["status"] in {"completed", "partial"}
        assert card["findings_count"] > 0

    # each agent's findings are attributed to it
    discovered = {f["discovered_by"] for f in r.findings}
    assert RESEARCH_AGENT.key in discovered
    assert COMPETITIVE_AGENT.key in discovered


def test_each_agent_reports_a_structured_result():
    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    for card in r.agents:
        for key in ("agent", "name", "status", "responsibility", "tools_used",
                    "findings_count", "coverage", "confidence", "summary"):
            assert key in card, f"{card.get('agent')} card missing {key!r}"
        assert 0.0 <= card["confidence"] <= 1.0

    research = agent_card(r, RESEARCH_AGENT.key)
    if research and research["findings_count"]:
        assert "research_trends" in research
        assert "key_developments" in research
    competitive = agent_card(r, COMPETITIVE_AGENT.key)
    if competitive and competitive["findings_count"]:
        assert "competitors_analyzed" in competitive
        assert "market_signals" in competitive


# ─────────────────────────────────────────────────────────────
# 4. Cross-agent collaboration
# ─────────────────────────────────────────────────────────────
def test_orchestrator_delegates_a_follow_up_from_an_observation():
    """A competitor-only goal has no research need; a detected signal creates one."""
    r = go("Track recent announcements from OpenAI", keywords=["AI agents"],
           competitors=["OpenAI"])

    follow_ups = [e for e in r.collaboration_events if e["kind"] == "follow_up"]
    assert follow_ups, "a strategic signal should trigger cross-agent validation"

    event = follow_ups[0]
    assert set(event["participants"]) >= {COMPETITIVE_AGENT.key, RESEARCH_AGENT.key}
    assert event["detail"], "the collaboration must record why it happened"

    # the follow-up actually ran: the research agent was not in the original plan
    assert RESEARCH_AGENT.key in skipped(r)
    assert RESEARCH_AGENT.key in r.state["completed_agents"]


def test_follow_up_revises_the_plan_so_the_agent_has_work():
    r = go("Track recent announcements from OpenAI", keywords=["AI agents"],
           competitors=["OpenAI"])
    revisions = r.state["plan"]["revisions"]
    assert len(revisions) > 1, f"the orchestrator should record a plan revision: {revisions}"
    assert any("research" in rev for rev in revisions[1:])


def test_agent_messages_form_a_request_response_trail():
    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    messages = r.state["agent_messages"]
    assert messages, "delegation must be recorded as messages"

    tasks = [m for m in messages if m.get("from_agent") == ORCHESTRATOR.key]
    reports = [m for m in messages if m.get("to_agent") == ORCHESTRATOR.key]
    assert tasks and reports
    for t in tasks:
        assert t["task"] and t["reason"] and t["allowed_tools"]
        assert t["context"]["user_goal"] == r.goal
    for rep in reports:
        assert rep["status"] in {"completed", "partial", "degraded", "failed", "skipped"}


def test_activity_log_uses_the_orchestration_event_taxonomy():
    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    types = {e["event_type"] for e in r.activity_log}
    assert {"ORCHESTRATION", "DELEGATION", "TOOL_CALL", "OBSERVATION", "RESULT"} <= types

    # entries are attributed to the agent that emitted them
    agents = {e["agent"] for e in r.activity_log if e["agent"]}
    assert ORCHESTRATOR.key in agents
    assert agents & {RESEARCH_AGENT.key, COMPETITIVE_AGENT.key}


# ─────────────────────────────────────────────────────────────
# 7. Duplicate findings merge without losing sources
# ─────────────────────────────────────────────────────────────
def _finding(fid, title, agent, provider, url, competitor="OpenAI", signals=("launch",)):
    return FindingRecord(
        id=fid, title=title, source="news", summary=title, url=url,
        published_date="2026-08-20", provider=provider, competitor=competitor,
        signals=list(signals), relevance=0.6, discovered_by=agent,
    )


def test_same_development_from_two_agents_is_detected():
    a = _finding("a", "OpenAI launches new AI agent platform for developers",
                 RESEARCH_AGENT.key, "arxiv", "https://x.test/a")
    b = _finding("b", "OpenAI launches AI agent platform aimed at developers",
                 COMPETITIVE_AGENT.key, "tavily", "https://y.test/b")
    assert _describes_same_development(a, b) is True


def test_unrelated_findings_are_not_merged():
    a = _finding("a", "OpenAI launches new AI agent platform",
                 RESEARCH_AGENT.key, "arxiv", "https://x.test/a")
    b = _finding("b", "Quantum error correction milestone reached",
                 COMPETITIVE_AGENT.key, "tavily", "https://y.test/b",
                 competitor="", signals=("benchmark",))
    assert _describes_same_development(a, b) is False


def test_merge_raises_confidence_and_keeps_both_sources():
    """The core merge guarantee: corroboration boosts, and no source is lost."""
    from app.agents.state import AgentState

    class Host:
        def __init__(self):
            self.state = AgentState(user_goal="Track OpenAI AI agents")
            self.state.competitors = ["OpenAI"]

            class _L:
                def speaking_as(self, *_a, **_k): pass
                def orchestration(self, *_a, **_k): pass
                def collaboration(self, *_a, **_k): pass
                def warning(self, *_a, **_k): pass
            self.logger = _L()

    host = Host()
    a = _finding("a", "OpenAI launches new AI agent platform for developers",
                 RESEARCH_AGENT.key, "arxiv", "https://x.test/a")
    b = _finding("b", "OpenAI launches AI agent platform aimed at developers",
                 COMPETITIVE_AGENT.key, "tavily", "https://y.test/b")
    host.state.findings.extend([a, b])

    before = (a.relevance, b.relevance)
    IntelligenceOrchestrator()._merge(host)  # noqa: SLF001

    assert a.relevance > before[0] and b.relevance > before[1], "corroboration must raise both"
    assert COMPETITIVE_AGENT.key in a.corroborated_by
    assert RESEARCH_AGENT.key in b.corroborated_by
    assert host.state.corroborated_finding_ids == {"a", "b"}

    events = [e for e in host.state.collaboration_events if e["kind"] == "corroboration"]
    assert events, "a corroboration event must be recorded"
    # both original source links survive the merge
    assert set(events[0]["evidence"]) == {"https://x.test/a", "https://y.test/b"}
    assert events[0]["confidence_delta"] > 0


# ─────────────────────────────────────────────────────────────
# 5. Missing provider key → degraded, not crashed
# ─────────────────────────────────────────────────────────────
def test_missing_tavily_key_degrades_without_crashing(monkeypatch):
    from app.sources.web import TavilyConnector

    monkeypatch.setattr(TavilyConnector, "available", lambda self: False)

    r = go("Track recent announcements from OpenAI", keywords=["AI agents"],
           competitors=["OpenAI"])
    assert r.status in {"completed", "completed_partial"}

    card = agent_card(r, COMPETITIVE_AGENT.key)
    assert card is not None
    assert card["coverage"] in {"live", "partial", "simulated", "unavailable"}
    # simulated data must be labelled, never presented as live
    if r.metrics["simulated_data_used"]:
        assert any(f["simulated"] for f in r.findings)


def test_coverage_labels_are_constrained():
    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    for card in r.agents:
        assert card["coverage"] in {"live", "partial", "simulated", "unavailable"}


# ─────────────────────────────────────────────────────────────
# 6. One agent fails → the other still reaches the report
# ─────────────────────────────────────────────────────────────
def test_one_agent_failure_does_not_block_the_other(monkeypatch):
    from app.tools.research_tool import ResearchTool

    async def boom(self, tool_input, ctx, result):
        raise RuntimeError("simulated research outage")

    monkeypatch.setattr(ResearchTool, "_execute", boom)

    r = go("Track AI agent technology, OpenAI, Anthropic, research and competitor developments",
           keywords=["AI agents"], competitors=["OpenAI", "Anthropic"])

    assert r.status in {"completed", "completed_partial"}
    competitive = agent_card(r, COMPETITIVE_AGENT.key)
    assert competitive is not None
    assert competitive["findings_count"] > 0, "the healthy agent must still deliver"
    assert r.insights, "the report must still be produced"


def test_orchestrator_survives_a_specialist_exception(monkeypatch):
    from app.agents.specialists import SpecialistAgent

    original = SpecialistAgent.execute

    async def flaky(self, host, task, ctx):
        if self.profile.key == RESEARCH_AGENT.key:
            raise RuntimeError("specialist blew up")
        return await original(self, host, task, ctx)

    monkeypatch.setattr(SpecialistAgent, "execute", flaky)

    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    # orchestration errors are contained; the run still finishes and reports
    assert r.status in {"completed", "completed_partial", "failed"}
    assert isinstance(r.agents, list)


# ─────────────────────────────────────────────────────────────
# Report + API integration
# ─────────────────────────────────────────────────────────────
def test_report_contains_an_agent_contributions_section():
    from app.reports.builder import build_report

    r = go("Track AI agent technology, OpenAI, Anthropic, research and competitor developments",
           keywords=["AI agents"], competitors=["OpenAI", "Anthropic"])
    report = build_report(r.to_dict())
    ac = report["agent_contributions"]

    assert ac["architecture"]
    assert len(ac["specialists"]) >= 2
    assert ac["orchestrator"]["bullets"]
    assert isinstance(ac["collaboration_count"], int)
    for card in ac["specialists"]:
        assert card["name"] and card["responsibility"]


def test_report_renderers_include_agent_contributions():
    from app.reports.builder import build_report
    from app.reports.html import render_html
    from app.reports.markdown import render_markdown

    r = go("Track AI agents research and OpenAI announcements",
           keywords=["AI agents"], competitors=["OpenAI"])
    report = build_report(r.to_dict())

    html = render_html(report)
    assert "Agent Contributions" in html
    assert "Research Intelligence Agent" in html or "Competitive Intelligence Agent" in html

    md = render_markdown(report)
    assert "## 03 · Agent Contributions" in md


def test_api_result_shape_is_backward_compatible():
    r = go("Track AI agents research", keywords=["AI agents"])
    payload = r.to_dict()
    # every pre-existing key still present
    for key in ("status", "run_id", "goal", "activity_log", "tools_used", "findings",
                "insights", "summary", "state", "metrics"):
        assert key in payload, f"breaking change: {key!r} missing"
    # plus the additive multi-agent surface
    for key in ("execution_plan", "agents", "collaboration_events"):
        assert key in payload
    assert "agents_used" in payload["metrics"]
    assert "collaboration_events" in payload["metrics"]


def test_message_dataclasses_serialise():
    task = AgentTask(run_id="r", from_agent="orchestrator", to_agent="research_agent",
                     task="do the thing", allowed_tools=["research_search"])
    report = AgentReport(run_id="r", from_agent="research_agent")
    event = CollaborationEvent(run_id="r", kind="corroboration", initiator="orchestrator")
    for obj in (task, report, event):
        d = obj.to_dict()
        assert isinstance(d, dict) and d["run_id"] == "r"
    assert report.public()["name"] == RESEARCH_AGENT.name


@pytest.mark.parametrize("goal,expect_selected,expect_skipped", [
    ("Track recent research on multi-agent reinforcement learning",
     RESEARCH_AGENT.key, COMPETITIVE_AGENT.key),
    ("Monitor generative AI patents", RESEARCH_AGENT.key, COMPETITIVE_AGENT.key),
    ("Monitor industry news about solid-state batteries",
     COMPETITIVE_AGENT.key, RESEARCH_AGENT.key),
])
def test_agent_selection_is_goal_driven(goal, expect_selected, expect_skipped):
    r = go(goal)
    assert expect_selected in selected(r), f"{goal!r} → {r.execution_plan}"
    assert expect_skipped in skipped(r), f"{goal!r} → {r.execution_plan}"
