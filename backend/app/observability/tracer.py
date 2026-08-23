"""The tracer — span creation, nesting and correlation.

One `Tracer` per execution. Nesting is handled with a `ContextVar` stack so a span
opened inside an `async` call automatically becomes a child of whatever span is
active on that task, which is what gives real parent/child traceability through
LangGraph nodes, agents and tool calls without threading ids through every signature.

Two properties are load-bearing:

  * **Tracing never breaks the run.** Every public method is wrapped; a bug in
    instrumentation must not take down the agent, so failures are counted and
    swallowed.
  * **Parallel branches nest correctly.** LangGraph runs the two specialist agents in
    one superstep. `ContextVar` is copied per task, so each branch keeps its own
    parent chain instead of interleaving.
"""

from __future__ import annotations

import contextlib
import time
from contextvars import ContextVar
from typing import Any

from .providers import ExternalTraceProvider, LocalTraceProvider, local_provider
from .redaction import safe_error_message, scrub_attributes, scrub_text
from .schemas import (
    ERROR_CATEGORIES,
    ErrorRecord,
    Span,
    SpanEvent,
    TokenUsage,
    Trace,
    new_id,
    now_iso,
)

# The active span stack for the current task.
_SPAN_STACK: ContextVar[tuple[str, ...]] = ContextVar("_ip_span_stack", default=())
# The tracer serving the current execution.
_CURRENT: ContextVar["Tracer | None"] = ContextVar("_ip_tracer", default=None)


def current_trace() -> "Tracer | None":
    """The tracer for this execution, or None when tracing is off."""
    return _CURRENT.get()


class Tracer:
    """Collects one trace. Safe to construct even when observability is disabled."""

    def __init__(
        self,
        *,
        run_id: str,
        goal: str = "",
        thread_id: str = "",
        root_operation: str = "agent_run",
        scenario: str = "normal",
        scenario_config: dict[str, Any] | None = None,
        optimization_version: int = 0,
        environment: str = "development",
        framework_version: str = "",
        enabled: bool = True,
        local: LocalTraceProvider | None = None,
        external: ExternalTraceProvider | None = None,
    ) -> None:
        self.enabled = enabled
        self.trace = Trace(
            trace_id=new_id("tr-"),
            run_id=run_id,
            thread_id=thread_id,
            root_operation=root_operation,
            goal=scrub_text(goal, limit=300),
            scenario=scenario,
            scenario_config=scrub_attributes(scenario_config or {}),
            optimization_version=optimization_version,
            environment=environment,
            framework_version=framework_version,
        )
        self._spans: dict[str, Span] = {}
        self._t0 = time.perf_counter()
        self._starts: dict[str, float] = {}
        self.local = local if local is not None else local_provider
        self.external = external
        self.instrumentation_failures = 0
        # Aggregated token accounting across llm spans.
        self._tokens = TokenUsage()
        self._token_measured = False

    # ── identity ────────────────────────────────────────────
    @property
    def trace_id(self) -> str:
        return self.trace.trace_id

    def activate(self) -> None:
        _CURRENT.set(self)

    # ── span lifecycle ──────────────────────────────────────
    def start_span(
        self,
        name: str,
        kind: str,
        *,
        parent_span_id: str | None = None,
        agent: str = "",
        **attributes: Any,
    ) -> str:
        """Open a span and return its id. Returns "" when tracing is disabled."""
        if not self.enabled:
            return ""
        try:
            stack = _SPAN_STACK.get()
            parent = parent_span_id if parent_span_id is not None else (stack[-1] if stack else None)
            span = Span(
                span_id=new_id("sp-"),
                trace_id=self.trace.trace_id,
                name=str(name)[:120],
                kind=kind if kind in _KINDS else "node",
                parent_span_id=parent,
                run_id=self.trace.run_id,
                agent=agent,
                attributes=scrub_attributes(attributes),
            )
            self._spans[span.span_id] = span
            self._starts[span.span_id] = time.perf_counter()
            _SPAN_STACK.set((*stack, span.span_id))
            return span.span_id
        except Exception:  # noqa: BLE001 — instrumentation must not break the run
            self.instrumentation_failures += 1
            return ""

    def end_span(
        self,
        span_id: str,
        *,
        status: str = "ok",
        error: BaseException | str | None = None,
        **attributes: Any,
    ) -> None:
        if not self.enabled or not span_id:
            return
        try:
            span = self._spans.get(span_id)
            if span is None:
                return
            started = self._starts.pop(span_id, None)
            span.duration_ms = int(((time.perf_counter() - started) * 1000)) if started else 0
            span.end_time = now_iso()
            span.status = status if status in {"ok", "error", "degraded", "skipped"} else "ok"
            if attributes:
                span.attributes.update(scrub_attributes(attributes))
            if error is not None:
                span.status = "error"
                span.error = {"message": safe_error_message(error)}
            # Pop this span off the stack (and anything left above it, which would
            # only happen if a child was never closed).
            stack = _SPAN_STACK.get()
            if span_id in stack:
                idx = stack.index(span_id)
                _SPAN_STACK.set(stack[:idx])
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1

    @contextlib.contextmanager
    def span(self, name: str, kind: str, *, agent: str = "", **attributes: Any):
        """Context-managed span. Records an error status if the body raises."""
        span_id = self.start_span(name, kind, agent=agent, **attributes)
        try:
            yield span_id
        except BaseException as exc:
            self.end_span(span_id, status="error", error=exc)
            raise
        else:
            self.end_span(span_id, status="ok")

    def add_event(self, span_id: str, name: str, **attributes: Any) -> None:
        if not self.enabled or not span_id:
            return
        try:
            span = self._spans.get(span_id)
            if span is None:
                return
            span.events.append(
                SpanEvent(name=name, attributes=scrub_attributes(attributes)).to_dict()
            )
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1

    def set_attributes(self, span_id: str, **attributes: Any) -> None:
        if not self.enabled or not span_id:
            return
        try:
            span = self._spans.get(span_id)
            if span is not None:
                span.attributes.update(scrub_attributes(attributes))
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1

    # ── errors ──────────────────────────────────────────────
    def record_error(
        self,
        *,
        component: str,
        error_type: str,
        message: BaseException | str,
        span_id: str = "",
        agent: str = "",
        tool: str = "",
        provider: str = "",
        http_status: int | None = None,
        retryable: bool = False,
        retry_count: int = 0,
        fallback_attempted: bool = False,
        recovery_status: str = "unrecovered",
        injected: bool = False,
    ) -> str:
        """Record an error against the trace. Returns the error id."""
        if not self.enabled:
            return ""
        try:
            record = ErrorRecord(
                error_id=new_id("er-"),
                trace_id=self.trace.trace_id,
                span_id=span_id or (_SPAN_STACK.get()[-1] if _SPAN_STACK.get() else ""),
                component=component,
                error_type=error_type if error_type in ERROR_CATEGORIES else "UNKNOWN",
                safe_message=safe_error_message(message),
                agent=agent,
                tool=tool,
                provider=provider,
                http_status=http_status,
                retryable=retryable,
                retry_count=retry_count,
                fallback_attempted=fallback_attempted,
                recovery_status=recovery_status,
                injected=injected,
            )
            self.trace.errors.append(record.to_dict())
            if record.span_id:
                self.add_event(
                    record.span_id, "error_recorded",
                    error_type=record.error_type, http_status=http_status,
                    retryable=retryable, injected=injected,
                )
            return record.error_id
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1
            return ""

    def mark_recovered(self, error_id: str, *, via: str = "") -> None:
        """Flip an error to recovered once a fallback succeeded."""
        if not self.enabled or not error_id:
            return
        for err in self.trace.errors:
            if err.get("error_id") == error_id:
                err["recovery_status"] = "recovered"
                err["fallback_attempted"] = True
                if via:
                    err["recovered_via"] = via
                return

    # ── tokens ──────────────────────────────────────────────
    def record_tokens(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost_usd: float | None = None,
        span_id: str = "",
        prompt_type: str = "",
    ) -> None:
        """Fold a model call's token usage into the run total.

        Zero-token calls are *not* treated as measured: a provider that rejected the
        request reports nothing, and counting that as "0 tokens used" would be a
        fabricated measurement.
        """
        if not self.enabled:
            return
        try:
            measured = bool(input_tokens or output_tokens)
            if measured:
                self._token_measured = True
                self._tokens.input_tokens += int(input_tokens)
                self._tokens.output_tokens += int(output_tokens)
                self._tokens.total_tokens = (
                    self._tokens.input_tokens + self._tokens.output_tokens
                )
                self._tokens.model = model or self._tokens.model
                if cost_usd is not None:
                    self._tokens.estimated_cost_usd = round(
                        (self._tokens.estimated_cost_usd or 0.0) + float(cost_usd), 6
                    )
                self._tokens.status = "measured"
            if span_id:
                self.set_attributes(
                    span_id,
                    prompt_type=prompt_type,
                    model=model,
                    input_token_count=int(input_tokens) if measured else None,
                    output_token_count=int(output_tokens) if measured else None,
                    total_token_count=(
                        int(input_tokens) + int(output_tokens) if measured else None
                    ),
                    token_usage_status="measured" if measured else "unavailable",
                )
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1

    def note(self, text: str) -> None:
        if self.enabled and text:
            self.trace.notes.append(scrub_text(text, limit=240))

    # ── finish ──────────────────────────────────────────────
    def finish(
        self, *, status: str = "ok", metrics: dict[str, Any] | None = None,
        token_reason: str = "",
    ) -> Trace:
        """Close the trace, persist it locally and (optionally) export it."""
        if not self.enabled:
            return self.trace
        try:
            # Close any span left open so the tree is always well-formed.
            for span_id, span in self._spans.items():
                if not span.end_time:
                    self.end_span(span_id, status="degraded")
            self.trace.spans = [
                s.to_dict() for s in sorted(self._spans.values(), key=lambda x: x.start_time)
            ]
            self.trace.duration_ms = int((time.perf_counter() - self._t0) * 1000)
            self.trace.end_time = now_iso()
            self.trace.status = status if status in {"ok", "error", "degraded"} else "ok"
            if not self._token_measured:
                self._tokens.status = "unavailable"
                self._tokens.reason = token_reason or (
                    "the configured model provider did not report token usage for this run"
                )
            self.trace.token_usage = self._tokens.to_dict()
            self.trace.metrics = scrub_attributes(metrics or {})
            if self.instrumentation_failures:
                self.trace.notes.append(
                    f"{self.instrumentation_failures} instrumentation error(s) were "
                    f"swallowed so they could not affect the run"
                )
        except Exception:  # noqa: BLE001
            self.instrumentation_failures += 1

        # Persist locally first — the local trace is the source of truth.
        try:
            self.local.record(self.trace.to_dict())
        except Exception:  # noqa: BLE001
            pass
        export_status: dict[str, Any] = {"external": "not configured"}
        if self.external is not None and self.external.enabled:
            self.external.record(self.trace.to_dict())
            export_status = self.external.status()
        elif self.external is not None:
            export_status = self.external.status()
        self.trace.export = export_status
        try:
            # Re-record so the stored copy carries the export status too.
            self.local.record(self.trace.to_dict())
        except Exception:  # noqa: BLE001
            pass
        return self.trace


_KINDS = set(
    (
        "run", "orchestrator", "node", "agent", "decision", "llm", "tool", "provider",
        "retry", "fallback", "memory", "evaluation", "verification", "synthesis",
    )
)


# ─────────────────────────────────────────────────────────────
def get_tracer(
    *,
    run_id: str,
    goal: str = "",
    thread_id: str = "",
    scenario: str = "normal",
    scenario_config: dict[str, Any] | None = None,
    optimization_version: int = 0,
    root_operation: str = "agent_run",
) -> Tracer:
    """Build a tracer using the project's configuration."""
    from ..config import settings

    external = None
    if getattr(settings, "trace_export_enabled", False):
        external = ExternalTraceProvider(
            endpoint=getattr(settings, "trace_export_endpoint", ""),
            api_key=getattr(settings, "trace_export_api_key", ""),
            project=getattr(settings, "trace_project", "insightpulse"),
        )
    tracer = Tracer(
        run_id=run_id,
        goal=goal,
        thread_id=thread_id,
        root_operation=root_operation,
        scenario=scenario,
        scenario_config=scenario_config,
        optimization_version=optimization_version,
        environment=getattr(settings, "app_env", "development"),
        framework_version=getattr(settings, "app_version", ""),
        enabled=bool(getattr(settings, "observability_enabled", True)),
        external=external,
    )
    tracer.activate()
    return tracer


def reset_span_stack() -> None:
    """Clear the nesting stack. Used between runs and by tests."""
    _SPAN_STACK.set(())
    _CURRENT.set(None)
