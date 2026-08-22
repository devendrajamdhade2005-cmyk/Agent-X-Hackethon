"""Inter-agent message contracts.

The orchestrator and the specialists only ever communicate through these
structures — never by reaching into each other's internals. That constraint is
what makes the collaboration real and inspectable: every delegation and every
report is a serialisable object that ends up in the API response and the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

AgentKey = Literal["orchestrator", "research_agent", "competitive_agent"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────
# Agent identity
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AgentProfile:
    """Static description of a specialist — its scope, tools and presentation."""

    key: str
    name: str
    icon: str
    accent: str
    responsibility: str
    # The information needs this agent owns, and the only tools it may call.
    need_keys: tuple[str, ...]
    tool_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ORCHESTRATOR = AgentProfile(
    key="orchestrator",
    name="Intelligence Orchestrator",
    icon="🧠",
    accent="purple",
    responsibility=(
        "Interprets the goal, decides which specialists are needed, delegates scoped "
        "tasks, reviews what each returns, requests cross-agent validation, then merges "
        "and prioritizes the combined evidence."
    ),
    need_keys=(),
    tool_names=(),
)

RESEARCH_AGENT = AgentProfile(
    key="research_agent",
    name="Research Intelligence Agent",
    icon="🔬",
    accent="blue",
    responsibility=(
        "Monitors academic and technological research: papers, preprints, emerging "
        "methods, benchmarks and patent filings that show where the technology is going."
    ),
    need_keys=("research", "patent"),
    tool_names=("research_search", "patent_search"),
)

COMPETITIVE_AGENT = AgentProfile(
    key="competitive_agent",
    name="Competitive Intelligence Agent",
    icon="🏢",
    accent="orange",
    responsibility=(
        "Tracks companies and the market: announcements, launches, funding, "
        "partnerships, acquisitions, shipped code and community reaction, using curated "
        "news and live open-web search."
    ),
    need_keys=("competitor", "news", "web"),
    tool_names=("competitor_search", "news_search", "web_search"),
)

SPECIALISTS: tuple[AgentProfile, ...] = (RESEARCH_AGENT, COMPETITIVE_AGENT)
PROFILES: dict[str, AgentProfile] = {
    p.key: p for p in (ORCHESTRATOR, RESEARCH_AGENT, COMPETITIVE_AGENT)
}


def profile(key: str) -> AgentProfile:
    return PROFILES.get(key, ORCHESTRATOR)


# ─────────────────────────────────────────────────────────────
# Orchestrator → specialist
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentTask:
    """A scoped assignment. The specialist may only act inside `allowed_tools`."""

    run_id: str
    from_agent: str
    to_agent: str
    task: str
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    need_keys: list[str] = field(default_factory=list)
    max_iterations: int = 3
    kind: Literal["primary", "follow_up"] = "primary"
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "task": self.task,
            "reason": self.reason,
            "context": self.context,
            "allowed_tools": self.allowed_tools,
            "need_keys": self.need_keys,
            "max_iterations": self.max_iterations,
            "kind": self.kind,
            "at": self.at,
        }


# ─────────────────────────────────────────────────────────────
# Specialist → orchestrator
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentReport:
    """What a specialist hands back. The orchestrator reasons over this only."""

    run_id: str
    from_agent: str
    to_agent: str = "orchestrator"
    status: Literal["completed", "partial", "degraded", "failed", "skipped"] = "completed"
    task: str = ""
    sources_checked: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    findings_count: int = 0
    relevant_count: int = 0
    observations: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    # Research-specific
    research_trends: list[str] = field(default_factory=list)
    key_developments: list[str] = field(default_factory=list)
    # Competitive-specific
    competitors_analyzed: list[str] = field(default_factory=list)
    market_signals: list[str] = field(default_factory=list)
    # Shared
    coverage: Literal["live", "partial", "simulated", "unavailable"] = "live"
    degraded_providers: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_summary: str = ""
    recommended_next_step: str = ""
    duration_ms: int = 0
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "status": self.status,
            "task": self.task,
            "sources_checked": self.sources_checked,
            "tools_used": self.tools_used,
            "findings_count": self.findings_count,
            "relevant_count": self.relevant_count,
            "observations": self.observations,
            "signals": self.signals,
            "research_trends": self.research_trends,
            "key_developments": self.key_developments,
            "competitors_analyzed": self.competitors_analyzed,
            "market_signals": self.market_signals,
            "coverage": self.coverage,
            "degraded_providers": self.degraded_providers,
            "errors": self.errors,
            "confidence": round(self.confidence, 3),
            "reasoning_summary": self.reasoning_summary,
            "recommended_next_step": self.recommended_next_step,
            "duration_ms": self.duration_ms,
            "at": self.at,
        }

    def public(self) -> dict[str, Any]:
        """The per-agent card the UI and the report render."""
        p = profile(self.from_agent)
        payload = {
            "agent": self.from_agent,
            "name": p.name,
            "icon": p.icon,
            "accent": p.accent,
            "responsibility": p.responsibility,
            "status": self.status,
            "task": self.task,
            "tools_used": self.tools_used,
            "sources_checked": self.sources_checked,
            "findings_count": self.findings_count,
            "relevant_count": self.relevant_count,
            "coverage": self.coverage,
            "confidence": round(self.confidence, 3),
            "summary": self.reasoning_summary,
            "observations": self.observations,
            "degraded_providers": self.degraded_providers,
            "errors": self.errors,
        }
        if self.from_agent == RESEARCH_AGENT.key:
            payload["research_trends"] = self.research_trends
            payload["key_developments"] = self.key_developments
        if self.from_agent == COMPETITIVE_AGENT.key:
            payload["competitors_analyzed"] = self.competitors_analyzed
            payload["market_signals"] = self.market_signals
        return payload


# ─────────────────────────────────────────────────────────────
# Collaboration
# ─────────────────────────────────────────────────────────────
@dataclass
class CollaborationEvent:
    """A point where one agent's output changed what another agent did."""

    run_id: str
    kind: Literal["follow_up", "corroboration", "merge", "gap_fill", "handoff"]
    initiator: str
    participants: list[str] = field(default_factory=list)
    summary: str = ""
    detail: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "initiator": self.initiator,
            "participants": self.participants,
            "summary": self.summary,
            "detail": self.detail,
            "evidence": self.evidence,
            "confidence_delta": round(self.confidence_delta, 3),
            "at": self.at,
        }


@dataclass
class ExecutionPlanEntry:
    """One line of the orchestrator's plan, with the reason it was decided."""

    agent: str
    selected: bool
    reason: str
    order: int = 0
    need_keys: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        p = profile(self.agent)
        return {
            "agent": self.agent,
            "name": p.name,
            "icon": p.icon,
            "accent": p.accent,
            "selected": self.selected,
            "reason": self.reason,
            "order": self.order,
            "need_keys": self.need_keys,
            "allowed_tools": self.allowed_tools,
        }
