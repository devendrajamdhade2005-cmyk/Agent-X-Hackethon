"""Instrumentation wrappers — where the running system meets the tracer.

The design goal is that instrumentation is *additive*: none of the existing
behaviour changes, and if any of it throws, the run continues untraced rather
than failing. That is why every wrapper here falls back to calling the original
function when there is no active tracer, and why the tracer's own methods already
swallow their exceptions.

Three seams are covered:

  * `traced_node` wraps a graph node at registration time in `builder.py`, so all
    fourteen nodes get spans from one change instead of fourteen edits. The node's
    returned state update supplies the span attributes, which means the span
    records what the node actually decided rather than what we hoped it would.
  * `traced_tool_call` wraps `GraphHost._call_tool` so each tool execution becomes
    a span parented to its agent, with provider fan-out recorded underneath.
  * `traced_llm_call` wraps `LLMClient.complete_json` to produce an `llm` span and,
    critically, to record *measured* token usage only. Prompt text never enters a
    span; only its length does.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from .tracer import current_trace

# Each graph node's span kind. The graph's semantics are already known statically,
# so classifying here keeps the node functions themselves untouched.
NODE_KINDS: dict[str, str] = {
    "understand": "node",
    "plan": "node",
    "decompose": "node",
    "resource_check": "decision",
    "dispatch": "decision",
    "research_agent": "agent",
    "competitive_agent": "agent",
    "observer": "node",
    "conflict_resolution": "node",
    "self_evaluator": "decision",
    "verify": "verification",
    "replan": "decision",
    "finalize": "synthesis",
    "memory_update": "memory",
}

# State keys worth lifting onto a span as scalar attributes. Counted when the value
# is a list, copied when scalar. Nothing here can contain prompt text or a secret.
_COUNT_KEYS = (
    "findings", "evidence_items", "agent_reports", "tool_executions", "tool_errors",
    "fallback_history", "insights", "final_insights", "current_tasks",
    "completed_agents", "selected_agents", "conflicting_evidence",
    "uncertainty_flags", "verification_findings", "hypotheses", "injected_events",
)
_SCALAR_KEYS = (
    "graph_step_count", "plan_version", "replan_count", "verify_count",
    "overall_confidence", "verification_status", "status", "next_route",
    "deadlock_detected", "termination_reason", "task_completion",
)

# `end_span` owns these keyword names, so a state key of the same name is renamed
# rather than silently dropped. `finalize` really does return a `status` key.
_RESERVED = {"status": "node_status", "error": "node_error"}


def _node_attributes(update: Any) -> dict[str, Any]:
    """Turn a node's state update into span attributes."""
    if not isinstance(update, dict):
        return {}
    attrs: dict[str, Any] = {}
    for key in _COUNT_KEYS:
        value = update.get(key)
        if isinstance(value, list) and value:
            attrs[f"{_RESERVED.get(key, key)}_count"] = len(value)
    for key in _SCALAR_KEYS:
        if key in update:
            value = update[key]
            if isinstance(value, (str, int, float, bool)):
                attrs[_RESERVED.get(key, key)] = value
    agents = update.get("completed_agents")
    if isinstance(agents, list) and agents:
        attrs["agents"] = ",".join(str(a) for a in agents[:6])
    return attrs


def traced_node(name: str, fn: Callable) -> Callable:
    """Wrap a graph node so it emits one span per execution."""
    kind = NODE_KINDS.get(name, "node")
    agent = name if kind == "agent" else ""

    @functools.wraps(fn)
    async def wrapped(state: dict, config: Any) -> Any:
        tracer = current_trace()
        if tracer is None or not tracer.enabled:
            return await fn(state, config)

        span_id = tracer.start_span(name, kind, agent=agent, node=name)
        try:
            update = await fn(state, config)
        except Exception as exc:  # noqa: BLE001 — record, then let it propagate
            tracer.record_error(
                component=f"node:{name}",
                error_type="GRAPH_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                agent=agent,
                span_id=span_id,
            )
            tracer.end_span(span_id, status="error", error=type(exc).__name__)
            raise

        # Everything from here is instrumentation. A failure in it must cost the
        # run nothing, so it is contained rather than propagated.
        try:
            attrs = _node_attributes(update)
            # A decision node's whole purpose is the branch it picked, so make
            # that explicit as an event instead of leaving it implicit.
            if kind == "decision":
                route = ""
                if isinstance(update, dict):
                    route = str(update.get("next_route") or "")
                    decisions = update.get("route_decisions")
                    if not route and isinstance(decisions, list) and decisions:
                        last = decisions[-1]
                        if isinstance(last, dict):
                            route = str(last.get("route") or last.get("decision") or "")
                if route:
                    attrs["decision"] = route
                    tracer.add_event(span_id, "decision_made", route=route)
            status = "ok"
            if isinstance(update, dict) and update.get("status") == "failed":
                status = "error"
            tracer.end_span(span_id, status=status, **attrs)
        except Exception:  # noqa: BLE001
            tracer.instrumentation_failures += 1
            try:
                tracer.end_span(span_id, status="ok")
            except Exception:  # noqa: BLE001
                pass
        return update

    return wrapped


def traced_tool_call(fn: Callable) -> Callable:
    """Wrap `GraphHost._call_tool` so every tool execution becomes a span."""

    @functools.wraps(fn)
    async def wrapped(self: Any, decision: Any, ctx: Any, iteration: int) -> Any:
        tracer = current_trace()
        tool_name = getattr(decision, "tool", "") or ""
        if tracer is None or not tracer.enabled:
            return await fn(self, decision, ctx, iteration)

        tool_input = getattr(decision, "tool_input", None)
        query = getattr(tool_input, "query", "") if tool_input else ""
        agent = getattr(self, "agent_key", "") or ""
        span_id = tracer.start_span(
            tool_name or "tool", "tool", agent=agent,
            tool=tool_name, query=query, iteration=iteration,
        )
        try:
            result = await fn(self, decision, ctx, iteration)
        except Exception as exc:  # noqa: BLE001
            tracer.record_error(
                component=f"tool:{tool_name}",
                error_type="TOOL_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                agent=agent,
                tool=tool_name,
                span_id=span_id,
            )
            tracer.end_span(span_id, status="error", error=type(exc).__name__)
            raise

        try:
            used = list(getattr(result, "providers_used", []) or [])
            failed = [
                str(p.get("provider", ""))
                for p in (getattr(result, "providers_failed", []) or [])
                if isinstance(p, dict)
            ]
            count = int(getattr(result, "count", 0) or 0)
            tracer.end_span(
                span_id, status=("ok" if count else "degraded"),
                result_count=count,
                latency_ms=int(getattr(result, "latency_ms", 0) or 0),
                providers_used=",".join(used),
                providers_failed=",".join(p for p in failed if p),
                simulated=bool(getattr(result, "simulated", False)),
            )
            # A tool that lost a provider but still returned data recovered by
            # fallback. Recording it keeps each error's lifecycle honest.
            if failed and count:
                tracer.add_event(
                    span_id, "fallback_recovered",
                    lost=",".join(p for p in failed if p),
                    served_by=",".join(used),
                )
        except Exception:  # noqa: BLE001
            tracer.instrumentation_failures += 1
            try:
                tracer.end_span(span_id, status="ok")
            except Exception:  # noqa: BLE001
                pass
        return result

    return wrapped


def traced_llm_call(fn: Callable) -> Callable:
    """Wrap `LLMClient.complete_json` for prompt/LLM spans and token accounting."""

    @functools.wraps(fn)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = current_trace()
        if tracer is None or not tracer.enabled:
            return await fn(self, *args, **kwargs)

        purpose = str(kwargs.get("purpose") or "unknown")
        system = kwargs.get("system") or ""
        user = kwargs.get("user") or ""
        before_in = int(getattr(getattr(self, "usage", None), "tokens_in", 0) or 0)
        before_out = int(getattr(getattr(self, "usage", None), "tokens_out", 0) or 0)

        # Only lengths are recorded — never the prompt itself (section 44).
        span_id = tracer.start_span(
            f"llm:{purpose}", "llm",
            prompt_type=purpose,
            system_prompt_chars=len(str(system)),
            user_prompt_chars=len(str(user)),
            model=str(getattr(self, "model", "") or ""),
            provider=str(getattr(getattr(self, "provider", None), "name", "") or ""),
        )
        try:
            parsed = await fn(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            tracer.record_error(
                component="llm", error_type="MODEL_ERROR",
                message=f"{type(exc).__name__}: {exc}", span_id=span_id,
            )
            tracer.end_span(span_id, status="error", error=type(exc).__name__)
            raise

        try:
            usage = getattr(self, "usage", None)
            delta_in = int(getattr(usage, "tokens_in", 0) or 0) - before_in
            delta_out = int(getattr(usage, "tokens_out", 0) or 0) - before_out
            if delta_in > 0 or delta_out > 0:
                tracer.record_tokens(
                    input_tokens=delta_in, output_tokens=delta_out,
                    model=str(getattr(self, "model", "") or ""),
                    span_id=span_id, prompt_type=purpose,
                )
            if parsed is None:
                # A refused or failed model call is an error the run *does* recover
                # from: every caller has a deterministic heuristic path. Recording it
                # as recovered keeps the error lifecycle accurate — the run continued
                # and still produced findings.
                error_id = tracer.record_error(
                    component="llm", error_type="MODEL_ERROR",
                    message=str(getattr(self, "last_error", "") or "model returned no result"),
                    span_id=span_id, retryable=True, fallback_attempted=True,
                )
                tracer.mark_recovered(error_id, via="heuristic reasoner")
                tracer.end_span(span_id, status="degraded", parsed=False,
                                fallback="heuristic reasoner")
            else:
                tracer.end_span(span_id, status="ok", parsed=True)
        except Exception:  # noqa: BLE001
            tracer.instrumentation_failures += 1
            try:
                tracer.end_span(span_id, status="ok")
            except Exception:  # noqa: BLE001
                pass
        return parsed

    return wrapped
