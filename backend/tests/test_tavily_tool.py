"""Task 2 — Tavily web-intelligence tool: registration, gating and resilience.

These assert the property that matters for the hackathon requirement: Tavily is
*chosen*, not always called, and the choice depends on the goal and on what
earlier observations returned.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.decision_engine import NEED_TO_TOOL, DecisionEngine
from app.agents.planner import Planner
from app.agents.state import AgentState, Observation, Plan, ToolCallRecord
from app.tools.base import FindingRecord
from app.sources.base import SourceQuery
from app.sources.web import TavilyConnector, _parse_web_date
from app.tools.registry import ToolRegistry, tool_registry


# ── registration ────────────────────────────────────────────
def test_web_tool_is_registered_and_exposed():
    assert "web_search" in tool_registry.names()
    entry = next(e for e in tool_registry.catalog() if e["name"] == "web_search")
    assert entry["available"] is True
    assert "tavily" in entry["providers_live"] + entry["providers_simulated"]
    assert NEED_TO_TOOL["web"] == "web_search"


def test_tavily_connector_declares_web_source_type():
    c = TavilyConnector(credentials={})
    assert c.source_type == "web"
    assert c.requires_key is True
    # No key configured -> not available, so the runner serves simulated items.
    assert c.available() is False


# ── planning: goal decides whether Tavily is required ───────
def _plan(goal: str, **kw) -> tuple[Plan, dict]:
    planner = Planner(llm=None)
    plan = asyncio.run(planner.build(goal, **kw))
    return plan, planner.derived()


@pytest.mark.parametrize(
    "goal,competitors,expect_required",
    [
        ("Track competitor activity and recent announcements from OpenAI", ["OpenAI"], True),
        ("Monitor recent industry developments in AI voice assistants", [], True),
        ("Track scientific research on multi-agent reinforcement learning", [], False),
        ("Monitor patents related to generative AI", [], False),
    ],
)
def test_web_need_required_only_for_current_activity_goals(goal, competitors, expect_required):
    plan, _ = _plan(goal, competitors=competitors)
    web = plan.need("web")
    assert web is not None, "the web need should always be declared"
    assert web.required is expect_required, f"web.required for {goal!r}"


# ── gating: conditional web need must earn its call ─────────
def _state(goal: str, *, competitors: list[str] | None = None) -> AgentState:
    plan, derived = _plan(goal, competitors=competitors or [])
    st = AgentState(user_goal=goal)
    st.plan = plan
    st.keywords = derived["keywords"]
    st.competitors = derived["competitors"]
    st.available_tools = tool_registry.usable_names()
    return st


def _engine() -> DecisionEngine:
    return DecisionEngine(ToolRegistry(), llm=None)


def _add_findings(st: AgentState, n: int, source: str = "research") -> None:
    """Register n relevant findings so coverage checks see a realistic state."""
    for i in range(n):
        st.register_finding(
            FindingRecord(
                id=f"{source}-{i}", title=f"{source} finding {i}", source=source,
                summary="…", url=f"https://example.com/{source}/{i}",
                published_date="2026-08-20", provider=source, relevance=0.7,
            )
        )


def test_conditional_web_is_not_called_without_a_reason():
    """A research goal with a healthy first result must not reach for the web."""
    st = _state("Track scientific research on multi-agent reinforcement learning")
    assert st.plan.need("web").required is False

    # research came back strong, satisfied its need, and produced real coverage
    st.iteration_count = 1
    st.plan.need("research").satisfied = True
    st.plan.need("research").attempts = 1
    _add_findings(st, 8, "research")
    st.tool_calls.append(
        ToolCallRecord(iteration=1, tool="research_search", tool_input={}, items_returned=10)
    )
    st.observations.append(
        Observation(iteration=1, tool="research_search", items_returned=10,
                    relevant_items=8, yield_quality="good")
    )

    tools = [c.tool for c in _engine().candidates(st)]
    assert "web_search" not in tools, f"web_search should not be a candidate, got {tools}"


def test_low_overall_yield_widens_to_web_search():
    """Distinct from the thin-observation path: too little total evidence also
    justifies widening to the open web."""
    st = _state("Track scientific research on multi-agent reinforcement learning")
    st.iteration_count = 1
    st.plan.need("research").satisfied = True
    st.plan.need("research").attempts = 1
    _add_findings(st, 1, "research")  # below MIN_TOTAL_RELEVANT
    st.tool_calls.append(
        ToolCallRecord(iteration=1, tool="research_search", tool_input={}, items_returned=2)
    )
    st.observations.append(
        Observation(iteration=1, tool="research_search", items_returned=2,
                    relevant_items=1, yield_quality="good")
    )
    web = next((c for c in _engine().candidates(st) if c.tool == "web_search"), None)
    assert web is not None, "insufficient total evidence should widen to the web"
    assert web.kind == "fill_gap"


def test_thin_observation_promotes_web_search():
    """Requirement 8: the previous observation changes the next decision."""
    st = _state("Track scientific research on multi-agent reinforcement learning")
    st.iteration_count = 1
    st.plan.need("research").attempts = 1
    st.tool_calls.append(
        ToolCallRecord(iteration=1, tool="research_search", tool_input={}, items_returned=1)
    )
    st.observations.append(
        Observation(iteration=1, tool="research_search", items_returned=1,
                    relevant_items=0, yield_quality="thin")
    )

    candidates = _engine().candidates(st)
    web = next((c for c in candidates if c.tool == "web_search"), None)
    assert web is not None, "a thin research yield should promote web_search"
    assert web.kind == "follow_signal"
    assert "thin" in web.reason.lower()


def test_market_signal_promotes_web_search():
    st = _state("Track scientific research on multi-agent reinforcement learning")
    st.iteration_count = 1
    st.plan.need("research").attempts = 1
    st.plan.need("research").satisfied = True
    st.tool_calls.append(
        ToolCallRecord(iteration=1, tool="research_search", tool_input={}, items_returned=9)
    )
    st.observations.append(
        Observation(iteration=1, tool="research_search", items_returned=9,
                    relevant_items=7, yield_quality="good", signals=["funding"])
    )
    st.detected_signals.add("funding")

    web = next((c for c in _engine().candidates(st) if c.tool == "web_search"), None)
    assert web is not None, "a market signal should promote web_search for corroboration"
    assert web.kind == "follow_signal"


def test_required_web_need_is_a_top_candidate():
    st = _state("Track competitor announcements from OpenAI", competitors=["OpenAI"])
    assert st.plan.need("web").required is True
    web = next((c for c in _engine().candidates(st) if c.tool == "web_search"), None)
    assert web is not None
    assert web.score >= 9.0, "a required need should outrank opportunistic candidates"


# ── input construction ──────────────────────────────────────
def test_web_input_targets_uncovered_competitors_and_news_topic():
    st = _state("Track competitor announcements from OpenAI and Anthropic",
                competitors=["OpenAI", "Anthropic"])
    engine = _engine()
    ti = engine._build_input(st, st.plan.need("web"))  # noqa: SLF001
    assert ti.extra.get("topic") == "news"
    assert ti.competitors == ["OpenAI", "Anthropic"]
    assert 0 < ti.since_days <= 60


def test_tavily_query_is_natural_language_not_boolean_soup():
    c = TavilyConnector(credentials={"tavily": "x"})
    q = SourceQuery(source="tavily", source_type="web", query="AI agents",
                    keywords=["AI agents"], competitors=["OpenAI"], limit=5)
    built = c._build_query(q)  # noqa: SLF001
    assert "OpenAI" in built
    assert "AI agents" in built
    assert '"' not in built, "Tavily ranks on prose; quoted boolean syntax hurts it"


def test_explicit_web_query_is_used_verbatim():
    c = TavilyConnector(credentials={"tavily": "x"})
    q = SourceQuery(source="tavily", source_type="web", query="x",
                    extra={"web_query": "exact phrase here"})
    assert c._build_query(q) == "exact phrase here"  # noqa: SLF001


# ── resilience ──────────────────────────────────────────────
def test_tavily_failure_yields_simulated_items_not_an_exception():
    """A missing key must degrade to labelled simulated data, never crash."""
    from app.sources.resilience import collect_from_source

    connector = TavilyConnector(credentials={})  # no key
    q = SourceQuery(source="tavily", source_type="web", query="AI agents",
                    keywords=["AI agents"], limit=4)

    outcome = asyncio.run(collect_from_source(connector, None, q))
    assert outcome.simulated is True
    assert outcome.items, "simulated fallback should still produce items"
    assert all(i.is_simulated for i in outcome.items), "every fallback item must be labelled"
    assert all(i.source_type == "web" for i in outcome.items)


def test_date_parsing_handles_tavily_rfc1123_and_junk():
    assert _parse_web_date("Wed, 19 Aug 2026 11:19:12 GMT") is not None
    assert _parse_web_date("2026-08-19T11:19:12Z") is not None
    assert _parse_web_date("garbage") is None
    assert _parse_web_date(None) is None


def test_unknown_need_key_does_not_break_reason_label():
    engine = _engine()
    assert "coverage" in engine._reason_label("web")  # noqa: SLF001
    assert "coverage" in engine._reason_label("mystery")  # noqa: SLF001
