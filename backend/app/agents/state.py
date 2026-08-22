"""Structured agent state.

Everything the agent knows lives here. Decisions are a pure-ish function of this
object, which is what makes the loop inspectable and testable: given a state you
can predict (and assert on) the next action.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..tools.base import FindingRecord, ToolInput, ToolResult

MAX_ITERATIONS = 10

AgentStatus = Literal[
    "initialized",
    "planning",
    "running",
    "finalizing",
    "completed",
    "completed_partial",
    "failed",
]

Step = Literal[
    "goal",
    "plan",
    "decide",
    "act",
    "observe",
    "analyze",
    "finalize",
    "done",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _title_fingerprint(title: str) -> str:
    """Normalized title used for near-duplicate suppression."""
    if not title:
        return ""
    letters = "".join(c if c.isalnum() else " " for c in title.lower())
    return " ".join(letters.split())[:120]


# ─────────────────────────────────────────────────────────────
# Plan
# ─────────────────────────────────────────────────────────────
@dataclass
class InformationNeed:
    """One thing the agent has decided it must find out to satisfy the goal."""

    key: str                      # research | news | competitor | patent
    reason: str = ""
    required: bool = True
    satisfied: bool = False
    min_items: int = 2
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "reason": self.reason,
            "required": self.required,
            "satisfied": self.satisfied,
            "min_items": self.min_items,
            "attempts": self.attempts,
        }


@dataclass
class Plan:
    objective: str = ""
    interpretation: str = ""
    needs: list[InformationNeed] = field(default_factory=list)
    opening_move: str = ""
    success_criteria: str = ""
    revisions: list[str] = field(default_factory=list)
    author: str = "heuristic-planner"

    def need(self, key: str) -> InformationNeed | None:
        return next((n for n in self.needs if n.key == key), None)

    def required_keys(self) -> list[str]:
        return [n.key for n in self.needs if n.required]

    def unsatisfied_required(self) -> list[InformationNeed]:
        return [n for n in self.needs if n.required and not n.satisfied]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "interpretation": self.interpretation,
            "needs": [n.to_dict() for n in self.needs],
            "opening_move": self.opening_move,
            "success_criteria": self.success_criteria,
            "revisions": self.revisions,
            "author": self.author,
        }


# ─────────────────────────────────────────────────────────────
# Decisions, calls, observations
# ─────────────────────────────────────────────────────────────
@dataclass
class Decision:
    action: Literal["call_tool", "finalize", "abort"]
    reasoning: str = ""
    tool: str | None = None
    tool_input: ToolInput | None = None
    expected_gain: str = ""
    confidence: float = 0.6
    author: str = "heuristic-policy"
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "tool": self.tool,
            "tool_input": self.tool_input.describe() if self.tool_input else None,
            "expected_gain": self.expected_gain,
            "confidence": round(self.confidence, 2),
            "author": self.author,
            "iteration": self.iteration,
        }


@dataclass
class ToolCallRecord:
    iteration: int
    tool: str
    tool_input: dict[str, Any]
    reasoning: str = ""
    ok: bool = True
    items_returned: int = 0
    new_items: int = 0
    duplicates: int = 0
    latency_ms: int = 0
    providers_used: list[str] = field(default_factory=list)
    providers_failed: list[dict[str, str]] = field(default_factory=list)
    simulated: bool = False
    error: str = ""
    note: str = ""
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "reasoning": self.reasoning,
            "ok": self.ok,
            "items_returned": self.items_returned,
            "new_items": self.new_items,
            "duplicates": self.duplicates,
            "latency_ms": self.latency_ms,
            "providers_used": self.providers_used,
            "providers_failed": self.providers_failed,
            "simulated": self.simulated,
            "error": self.error,
            "note": self.note,
            "at": self.at,
        }


@dataclass
class Observation:
    """The analyzed result of one tool call — this is what drives the next decision."""

    iteration: int
    tool: str
    summary: str = ""
    items_returned: int = 0
    new_items: int = 0
    duplicates: int = 0
    relevant_items: int = 0
    top_titles: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    competitors_seen: list[str] = field(default_factory=list)
    yield_quality: Literal["good", "thin", "empty", "failed"] = "good"
    ok: bool = True
    error: str = ""
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "tool": self.tool,
            "summary": self.summary,
            "items_returned": self.items_returned,
            "new_items": self.new_items,
            "duplicates": self.duplicates,
            "relevant_items": self.relevant_items,
            "top_titles": self.top_titles,
            "signals": self.signals,
            "competitors_seen": self.competitors_seen,
            "yield_quality": self.yield_quality,
            "ok": self.ok,
            "error": self.error,
            "at": self.at,
        }


@dataclass
class AgentError:
    iteration: int
    where: str
    message: str
    recovered: bool = True
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "where": self.where,
            "message": self.message,
            "recovered": self.recovered,
            "at": self.at,
        }


# ─────────────────────────────────────────────────────────────
# The state object
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    # ── goal ────────────────────────────────────────────────
    user_goal: str = ""
    tracking_topics: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # ── control ─────────────────────────────────────────────
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    current_step: Step = "goal"
    status: AgentStatus = "initialized"
    iteration_count: int = 0
    max_iterations: int = MAX_ITERATIONS
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None

    # ── working memory ──────────────────────────────────────
    plan: Plan = field(default_factory=Plan)
    available_tools: list[str] = field(default_factory=list)
    unavailable_tools: dict[str, str] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    findings: list[FindingRecord] = field(default_factory=list)
    errors: list[AgentError] = field(default_factory=list)

    # ── output ──────────────────────────────────────────────
    final_decision: str = ""
    final_insights: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    # ── bookkeeping ─────────────────────────────────────────
    seen_finding_ids: set[str] = field(default_factory=set)
    seen_title_keys: set[str] = field(default_factory=set)
    call_signatures: set[str] = field(default_factory=set)
    detected_signals: set[str] = field(default_factory=set)
    mentioned_companies: set[str] = field(default_factory=set)
    simulated_data_used: bool = False
    reasoner: str = "heuristic"
    llm_calls: int = 0
    stop_reason: str = ""

    # ── derived views used by the decision engine ───────────
    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for call in self.tool_calls:
            if call.tool not in seen:
                seen.append(call.tool)
        return seen

    def call_count(self, tool: str) -> int:
        return sum(1 for c in self.tool_calls if c.tool == tool)

    def findings_by_source(self, source: str) -> list[FindingRecord]:
        return [f for f in self.findings if f.source == source]

    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.source] = counts.get(f.source, 0) + 1
        return counts

    def relevant_findings(self, threshold: float = 0.35) -> list[FindingRecord]:
        return [f for f in self.findings if f.relevance >= threshold]

    def competitor_coverage(self) -> dict[str, int]:
        counts = {c: 0 for c in self.competitors}
        for f in self.findings:
            if f.competitor and f.competitor in counts:
                counts[f.competitor] += 1
        return counts

    def uncovered_competitors(self) -> list[str]:
        return [c for c, n in self.competitor_coverage().items() if n == 0]

    def last_observation(self) -> Observation | None:
        return self.observations[-1] if self.observations else None

    def has_signal(self, signal: str) -> bool:
        return signal in self.detected_signals

    def budget_left(self) -> int:
        return max(0, self.max_iterations - self.iteration_count)

    # ── mutation helpers ────────────────────────────────────
    def register_finding(self, finding: FindingRecord) -> bool:
        """Returns True when the finding is new to this run.

        Two-stage: URL identity, then a normalized-title fingerprint. The second
        stage matters because the same work legitimately arrives under different
        URLs — OpenAlex returns the preprint and the published version with
        different DOIs, and news wires get syndicated across outlets.
        """
        if finding.id in self.seen_finding_ids:
            return False
        title_key = _title_fingerprint(finding.title)
        if title_key and title_key in self.seen_title_keys:
            return False
        self.seen_finding_ids.add(finding.id)
        if title_key:
            self.seen_title_keys.add(title_key)
        self.findings.append(finding)
        return True

    def record_error(self, where: str, message: str, *, recovered: bool = True) -> AgentError:
        err = AgentError(
            iteration=self.iteration_count, where=where, message=message, recovered=recovered
        )
        self.errors.append(err)
        return err

    def absorb_result(self, result: ToolResult) -> None:
        if result.simulated:
            self.simulated_data_used = True

    # ── serialization ───────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """Full inspectable state — returned by the API so judges can see it."""
        return {
            "run_id": self.run_id,
            "user_goal": self.user_goal,
            "tracking_topics": self.tracking_topics,
            "competitors": self.competitors,
            "keywords": self.keywords,
            "current_step": self.current_step,
            "status": self.status,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "plan": self.plan.to_dict(),
            "available_tools": self.available_tools,
            "unavailable_tools": self.unavailable_tools,
            "decisions": [d.to_dict() for d in self.decisions],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "observations": [o.to_dict() for o in self.observations],
            "coverage": self.coverage(),
            "competitor_coverage": self.competitor_coverage(),
            "detected_signals": sorted(self.detected_signals),
            "errors": [e.to_dict() for e in self.errors],
            "final_decision": self.final_decision,
            "stop_reason": self.stop_reason,
            "reasoner": self.reasoner,
            "llm_calls": self.llm_calls,
            "simulated_data_used": self.simulated_data_used,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
