"""Trace, span and diagnosis contracts.

Plain dataclasses with `to_dict()`, matching the project's style. The span model is
deliberately OpenTelemetry-shaped (trace_id / span_id / parent_span_id / kind /
status / attributes / events) so an external exporter is a mapping exercise rather
than a rewrite — that is what keeps the project free of third-party lock-in.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# ── span taxonomy ────────────────────────────────────────────
SPAN_KINDS: tuple[str, ...] = (
    "run",            # the whole execution
    "orchestrator",   # graph-level coordination
    "node",           # one LangGraph node
    "agent",          # a specialist agent
    "decision",       # a routing/planning decision (safe summary only)
    "llm",            # a model call (metadata + tokens, never content)
    "tool",           # a tool invocation
    "provider",       # one source/provider attempt inside a tool
    "retry",          # a retry attempt
    "fallback",       # a fallback to an alternate source
    "memory",         # memory retrieval/consolidation
    "evaluation",     # Task 6 measurement
    "verification",   # evidence verification
    "synthesis",      # final synthesis
)

SpanStatus = Literal["running", "ok", "error", "degraded", "skipped"]

# ── error taxonomy (section 13) ──────────────────────────────
ERROR_CATEGORIES: tuple[str, ...] = (
    "RATE_LIMIT",
    "TIMEOUT",
    "NETWORK_ERROR",
    "HTTP_ERROR",
    "BAD_RESPONSE",
    "VALIDATION_ERROR",
    "MODEL_ERROR",
    "TOOL_ERROR",
    "ROUTING_ERROR",
    "MEMORY_ERROR",
    "CHECKPOINT_ERROR",
    "RESOURCE_LIMIT",
    "UNKNOWN",
)

# ── root-cause taxonomy (section 19) ─────────────────────────
ROOT_CAUSES: tuple[str, ...] = (
    "RATE_LIMIT",
    "TIMEOUT",
    "NETWORK_ERROR",
    "BAD_PROVIDER_RESPONSE",
    "EXCESSIVE_RETRY",
    "UNNECESSARY_TOOL_CALL",
    "REDUNDANT_TOOL_CALL",
    "POOR_ROUTING",
    "OVER_DECOMPOSITION",
    "UNDER_DECOMPOSITION",
    "PROMPT_OVERHEAD",
    "LOW_EVIDENCE",
    "RESOURCE_LIMIT",
    "MEMORY_RETRIEVAL",
    "CHECKPOINT",
    "MODEL_ERROR",
    "MULTIPLE_POSSIBLE_CAUSES",
    "UNKNOWN",
)


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# ─────────────────────────────────────────────────────────────
@dataclass
class SpanEvent:
    """A point-in-time occurrence inside a span (retry, fallback, error, note)."""

    name: str
    at: str = field(default_factory=now_iso)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    kind: str
    parent_span_id: str | None = None
    start_time: str = field(default_factory=now_iso)
    end_time: str = ""
    duration_ms: int = 0
    status: SpanStatus = "running"
    # Safe, redacted key/values. Never prompt text, never secrets.
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    # Correlation with the rest of the system.
    run_id: str = ""
    agent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
            "run_id": self.run_id,
            "agent": self.agent,
        }


@dataclass
class ErrorRecord:
    error_id: str
    trace_id: str
    span_id: str
    component: str
    error_type: str                 # one of ERROR_CATEGORIES
    safe_message: str
    agent: str = ""
    tool: str = ""
    provider: str = ""
    http_status: int | None = None
    retryable: bool = False
    retry_count: int = 0
    fallback_attempted: bool = False
    recovery_status: str = "unrecovered"   # recovered | unrecovered | degraded
    injected: bool = False
    at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenUsage:
    """Token accounting. `status` is explicit so 0 never masquerades as measured."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float | None = None
    status: str = "unavailable"      # measured | unavailable
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    trace_id: str
    run_id: str
    root_operation: str
    thread_id: str = ""
    start_time: str = field(default_factory=now_iso)
    end_time: str = ""
    duration_ms: int = 0
    status: SpanStatus = "running"
    goal: str = ""
    environment: str = "development"
    framework: str = "langgraph"
    framework_version: str = ""
    spans: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    # What this execution was for: baseline | controlled_failure | after_improvement
    scenario: str = "normal"
    scenario_config: dict[str, Any] = field(default_factory=dict)
    optimization_version: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ── derived views ───────────────────────────────────────
    def spans_of(self, kind: str) -> list[dict[str, Any]]:
        return [s for s in self.spans if s.get("kind") == kind]

    def children_of(self, span_id: str) -> list[dict[str, Any]]:
        return [s for s in self.spans if s.get("parent_span_id") == span_id]

    def root_spans(self) -> list[dict[str, Any]]:
        return [s for s in self.spans if not s.get("parent_span_id")]

    def summary(self) -> dict[str, Any]:
        tools = self.spans_of("tool")
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "scenario": self.scenario,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "agent_count": len(self.spans_of("agent")),
            "tool_call_count": len(tools),
            "llm_call_count": len(self.spans_of("llm")),
            "error_count": len(self.errors),
            "retry_count": sum(
                1 for s in self.spans if s.get("kind") == "retry"
            ),
            "fallback_count": sum(
                1 for s in self.spans if s.get("kind") == "fallback"
            ),
            "token_usage": self.token_usage,
            "optimization_version": self.optimization_version,
            "started_at": self.start_time,
            "export": self.export,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "root_operation": self.root_operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "goal": self.goal,
            "environment": self.environment,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "spans": self.spans,
            "errors": self.errors,
            "token_usage": self.token_usage,
            "scenario": self.scenario,
            "scenario_config": self.scenario_config,
            "optimization_version": self.optimization_version,
            "metrics": self.metrics,
            "export": self.export,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────
# Diagnosis + improvement contracts
# ─────────────────────────────────────────────────────────────
@dataclass
class Diagnosis:
    diagnosis_id: str
    trace_id: str
    root_cause_type: str
    affected_component: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    impact: dict[str, Any] = field(default_factory=dict)
    recommended_improvement: str = ""
    improvement_type: str = ""
    uncertain: bool = False
    alternatives: list[str] = field(default_factory=list)
    at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImprovementPlan:
    improvement_id: str
    root_cause_type: str
    improvement_type: str
    target_component: str
    current_configuration: dict[str, Any] = field(default_factory=dict)
    proposed_configuration: dict[str, Any] = field(default_factory=dict)
    expected_benefit: str = ""
    risk: str = "low"
    validation_required: bool = True
    status: str = "proposed"      # proposed | applied | rejected | reverted
    optimization_version: int = 0
    previous_version: int = 0
    changed_parameter: str = ""
    reason: str = ""
    created_at: str = field(default_factory=now_iso)
    applied_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
