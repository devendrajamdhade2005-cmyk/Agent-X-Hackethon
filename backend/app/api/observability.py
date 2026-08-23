"""Observability API — traces, diagnosis, and the self-improvement cycle.

    GET  /api/observability/status               provider + policy + injection state
    GET  /api/observability/traces               recent trace summaries
    GET  /api/observability/traces/{trace_id}    one full trace (spans + errors)
    GET  /api/observability/traces/{id}/tree     the same trace as a nested tree
    GET  /api/observability/runs/{run_id}        traces belonging to one run
    GET  /api/observability/errors               errors across recent traces
    GET  /api/observability/analysis/{trace_id}  measured observations
    GET  /api/observability/root-cause/{id}      diagnosis for one trace
    GET  /api/observability/policy               active runtime policy + versions
    POST /api/observability/policy/revert        roll back one policy version
    POST /api/observability/policy/reset         return to the shipped defaults
    GET  /api/observability/failure-targets      what a controlled failure may target
    POST /api/observability/controlled-failure   arm / disarm an injection
    POST /api/observability/improve              run the whole cycle, return report
    POST /api/observability/improve/stream       same, streamed stage by stage (SSE)
    GET  /api/observability/cycles               previous cycle reports

Nothing here fabricates a number. Every value returned was measured during a real
run; where a value could not be measured the response says so and why.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..observability import controlled_failure as cf
from ..observability.analyzer import RootCauseAnalyzer, TraceAnalyzer
from ..observability.improvement import METRIC_DIRECTION
from ..observability.loop import (
    DEFAULT_CASE_ID,
    DEFAULT_FAILURE,
    DEFAULT_FAILURE_COUNT,
    DEFAULT_REPEATS,
    DEFAULT_TARGET,
    MAX_REPEATS,
    loop,
)
from ..observability.policy import BOUNDS, IMPROVEMENT_TYPES, registry as policy_registry
from ..observability.providers import local_provider
from ..observability.schemas import ERROR_CATEGORIES, ROOT_CAUSES, SPAN_KINDS
from .agent import require_token
from .guard import limit_heavy

router = APIRouter(prefix="/api/observability", tags=["observability"])

# Completed cycle reports, newest last. Kept in memory like the run history: a
# cycle report is a demo/diagnostic artefact, not durable business data.
_CYCLES: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_MAX_CYCLES = 10

_analyzer = TraceAnalyzer()
_root_cause = RootCauseAnalyzer()


def _remember_cycle(report: dict[str, Any]) -> None:
    cycle_id = str(report.get("cycle_id") or "")
    if not cycle_id:
        return
    _CYCLES[cycle_id] = report
    while len(_CYCLES) > _MAX_CYCLES:
        _CYCLES.popitem(last=False)


def _trace_or_404(trace_id: str) -> dict[str, Any]:
    trace = local_provider.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class ControlledFailureRequest(BaseModel):
    """Arm (or disarm) a deterministic failure for one upcoming run."""

    run_id: str = Field(min_length=1, max_length=64)
    target_source: str = Field(default=DEFAULT_TARGET, max_length=64)
    failure_type: str = Field(default=DEFAULT_FAILURE)
    failure_count: int = Field(default=DEFAULT_FAILURE_COUNT, ge=1,
                               le=cf.MAX_FAILURE_COUNT)
    enabled: bool = True

    @field_validator("failure_type")
    @classmethod
    def _known_failure(cls, v: str) -> str:
        if v not in cf.FAILURE_TYPES:
            raise ValueError(f"failure_type must be one of {', '.join(cf.FAILURE_TYPES)}")
        return v

    @field_validator("target_source")
    @classmethod
    def _known_target(cls, v: str) -> str:
        targets = cf.available_targets()
        # Only a registered source may be targeted, so an injection can never
        # describe a provider the system does not actually call.
        if targets and v not in targets:
            raise ValueError(f"target_source must be a registered source: {', '.join(targets)}")
        return v


class ImproveRequest(BaseModel):
    """Run the trace → diagnose → improve → re-run → verify cycle."""

    target_source: str = Field(default=DEFAULT_TARGET, max_length=64)
    failure_type: str = Field(default=DEFAULT_FAILURE)
    failure_count: int = Field(default=DEFAULT_FAILURE_COUNT, ge=1,
                               le=cf.MAX_FAILURE_COUNT)
    case_id: str = Field(default=DEFAULT_CASE_ID, max_length=32)
    primary_metric: str = Field(default="duration_ms")
    simulation_mode: bool = True
    validate_with_evaluation: bool = True
    # Runs per side. More repeats measure the workload's own variance more tightly,
    # so the acceptance floor is better grounded.
    repeats: int = Field(default=DEFAULT_REPEATS, ge=1, le=MAX_REPEATS)

    @field_validator("failure_type")
    @classmethod
    def _known_failure(cls, v: str) -> str:
        if v not in cf.FAILURE_TYPES:
            raise ValueError(f"failure_type must be one of {', '.join(cf.FAILURE_TYPES)}")
        return v

    @field_validator("target_source")
    @classmethod
    def _known_target(cls, v: str) -> str:
        targets = cf.available_targets()
        if targets and v not in targets:
            raise ValueError(f"target_source must be a registered source: {', '.join(targets)}")
        return v

    @field_validator("primary_metric")
    @classmethod
    def _known_metric(cls, v: str) -> str:
        if v not in METRIC_DIRECTION:
            raise ValueError(
                f"primary_metric must be one of {', '.join(sorted(METRIC_DIRECTION))}"
            )
        return v


# ─────────────────────────────────────────────────────────────
# Read: status, traces, errors
# ─────────────────────────────────────────────────────────────
@router.get("/status")
async def observability_status() -> dict[str, Any]:
    """What the observability layer is doing right now."""
    from ..config import settings

    return {
        "enabled": bool(getattr(settings, "observability_enabled", True)),
        "trace_provider": local_provider.status(),
        "external_export": {
            "enabled": bool(getattr(settings, "trace_export_enabled", False)),
            "endpoint_configured": bool(getattr(settings, "trace_export_endpoint", "")),
            "project": getattr(settings, "trace_project", "insightpulse"),
            "note": (
                "traces are always retained locally; external export is an optional "
                "mirror and its failure never affects a run"
            ),
        },
        "policy": policy_registry.active.to_dict(),
        "policy_versions": policy_registry.history(),
        "armed_failures": cf.controller.active_plans(),
        "vocabulary": {
            "span_kinds": list(SPAN_KINDS),
            "error_categories": list(ERROR_CATEGORIES),
            "root_causes": list(ROOT_CAUSES),
            "improvement_types": list(IMPROVEMENT_TYPES),
            "failure_types": list(cf.FAILURE_TYPES),
            "bounds": {k: list(v) for k, v in BOUNDS.items()},
        },
    }


def _summarise(trace: dict[str, Any]) -> dict[str, Any]:
    """Project a stored trace to the fields a list view needs.

    Full traces carry every span, which is far more than a list needs to render —
    so the list endpoint returns counts and totals and leaves the spans to
    `GET /traces/{id}`.
    """
    spans = trace.get("spans") or []
    errors = trace.get("errors") or []
    return {
        "trace_id": trace.get("trace_id", ""),
        "run_id": trace.get("run_id", ""),
        "goal": trace.get("goal", ""),
        "scenario": trace.get("scenario", "normal"),
        "status": trace.get("status", "ok"),
        "duration_ms": trace.get("duration_ms", 0),
        "span_count": trace.get("span_count", len(spans)),
        "error_count": trace.get("error_count", len(errors)),
        "optimization_version": trace.get("optimization_version", 0),
        "token_usage": trace.get("token_usage") or {},
        "start_time": trace.get("start_time", ""),
        "end_time": trace.get("end_time", ""),
        # A trace restored from the disk mirror keeps its summary but not its
        # spans, so say so rather than letting the UI imply the spans are gone.
        "partial": bool(trace.get("partial")),
    }


@router.get("/traces")
async def list_traces(limit: int = Query(default=20, ge=1, le=40)) -> dict[str, Any]:
    """Recent trace summaries, newest first."""
    traces = [_summarise(t) for t in local_provider.list()[:limit]]
    return {"traces": traces, "count": len(traces), "store": local_provider.status()}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """One full trace: every span, every error, token usage, export status."""
    return _trace_or_404(trace_id)


@router.get("/traces/{trace_id}/tree")
async def get_trace_tree(trace_id: str) -> dict[str, Any]:
    """The same trace shaped as a nested tree, ready to render as a timeline."""
    trace = _trace_or_404(trace_id)
    spans = trace.get("spans") or []
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        by_parent.setdefault(str(span.get("parent_span_id") or ""), []).append(span)

    def build(parent_id: str, depth: int = 0) -> list[dict[str, Any]]:
        out = []
        for span in by_parent.get(parent_id, []):
            out.append({
                "span_id": span.get("span_id"),
                "name": span.get("name"),
                "kind": span.get("kind"),
                "agent": span.get("agent", ""),
                "status": span.get("status"),
                "duration_ms": span.get("duration_ms", 0),
                "start_time": span.get("start_time"),
                "attributes": span.get("attributes") or {},
                "events": span.get("events") or [],
                "error": span.get("error"),
                "depth": depth,
                "children": build(str(span.get("span_id") or ""), depth + 1),
            })
        return out

    total = int(trace.get("duration_ms") or 0)
    # A trace restored from the disk mirror keeps its summary counts but not its
    # spans. Report both numbers so an empty timeline is explained rather than
    # looking like a trace with no spans.
    recorded = int(trace.get("span_count") or len(spans))
    partial = bool(trace.get("partial")) or (recorded > 0 and not spans)
    return {
        "trace_id": trace.get("trace_id"),
        "run_id": trace.get("run_id"),
        "scenario": trace.get("scenario"),
        "status": trace.get("status"),
        "duration_ms": total,
        "span_count": len(spans),
        "recorded_span_count": recorded,
        "partial": partial,
        "partial_reason": (
            "This trace was restored from the on-disk mirror, which stores summaries "
            "only. Span detail is kept in memory for the most recent runs."
            if partial else ""
        ),
        "optimization_version": trace.get("optimization_version", 0),
        "token_usage": trace.get("token_usage") or {},
        "tree": build(""),
        # An orphaned span would mean the parent/child chain broke; surfacing the
        # count makes trace integrity checkable from the UI rather than assumed.
        "orphan_count": sum(
            1 for s in spans
            if s.get("parent_span_id")
            and s.get("parent_span_id") not in {x.get("span_id") for x in spans}
        ),
    }


@router.get("/runs/{run_id}")
async def traces_for_run(run_id: str) -> dict[str, Any]:
    """The trace recorded for one agent run."""
    trace = local_provider.by_run(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="no trace for that run")
    return {"run_id": run_id, "trace": trace, "summary": _summarise(trace)}


@router.get("/errors")
async def list_errors(limit: int = Query(default=10, ge=1, le=40)) -> dict[str, Any]:
    """Errors across recent traces, grouped by category and by component."""
    errors: list[dict[str, Any]] = []
    for trace in local_provider.list()[:limit]:
        for err in trace.get("errors") or []:
            errors.append({**err, "scenario": trace.get("scenario", "")})

    by_category: dict[str, int] = {}
    by_component: dict[str, int] = {}
    for err in errors:
        cat = str(err.get("error_type") or "UNKNOWN")
        by_category[cat] = by_category.get(cat, 0) + 1
        comp = str(err.get("provider") or err.get("component") or "unknown")
        by_component[comp] = by_component.get(comp, 0) + 1

    return {
        "errors": errors,
        "count": len(errors),
        "by_category": by_category,
        "by_component": by_component,
        "recovered": sum(1 for e in errors if e.get("recovery_status") == "recovered"),
        "injected": sum(1 for e in errors if e.get("injected")),
    }


@router.get("/analysis/{trace_id}")
async def analyze_trace(trace_id: str) -> dict[str, Any]:
    """Measured observations for one trace: no interpretation, only facts."""
    return _analyzer.analyze(_trace_or_404(trace_id))


@router.get("/root-cause/{trace_id}")
async def root_cause(trace_id: str) -> dict[str, Any]:
    """Diagnose one trace and describe the evidence behind the conclusion."""
    trace = _trace_or_404(trace_id)
    analysis = _analyzer.analyze(trace)
    diagnosis = _root_cause.diagnose(trace, analysis)
    return {
        "trace_id": trace_id,
        "diagnosis": diagnosis.to_dict(),
        "analysis": analysis,
    }


# ─────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────
@router.get("/policy")
async def get_policy() -> dict[str, Any]:
    """The runtime policy currently in force, and how it got there."""
    return {
        "active": policy_registry.active.to_dict(),
        "version": policy_registry.version,
        "history": policy_registry.history(),
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "note": (
            "Improvements change these bounded runtime values only. No source file "
            "is ever written by the improvement engine."
        ),
    }


@router.post("/policy/revert", dependencies=[Depends(require_token)])
async def revert_policy() -> dict[str, Any]:
    """Roll back one policy version."""
    before = policy_registry.version
    policy_registry.revert()
    return {
        "reverted_from": before,
        "version": policy_registry.version,
        "active": policy_registry.active.to_dict(),
    }


@router.post("/policy/reset", dependencies=[Depends(require_token)])
async def reset_policy() -> dict[str, Any]:
    """Return to the shipped defaults, so a demo can be repeated cleanly."""
    policy_registry.reset()
    return {
        "version": policy_registry.version,
        "active": policy_registry.active.to_dict(),
        "note": "runtime policy is back to the values the application ships with",
    }


# ─────────────────────────────────────────────────────────────
# Controlled failure
# ─────────────────────────────────────────────────────────────
@router.get("/failure-targets")
async def failure_targets() -> dict[str, Any]:
    """Which providers may be targeted, and with what failure shapes."""
    return {
        "targets": cf.available_targets(),
        "failure_types": list(cf.FAILURE_TYPES),
        "max_failure_count": cf.MAX_FAILURE_COUNT,
        "default": {
            "target_source": DEFAULT_TARGET,
            "failure_type": DEFAULT_FAILURE,
            "failure_count": DEFAULT_FAILURE_COUNT,
        },
        "armed": cf.controller.active_plans(),
        "note": (
            "An injection is keyed to a single run id, so arming one cannot affect "
            "any other run in this process."
        ),
    }


@router.post("/controlled-failure", dependencies=[Depends(require_token)])
async def controlled_failure(payload: ControlledFailureRequest) -> dict[str, Any]:
    """Arm or disarm a deterministic failure for a specific run id."""
    if not payload.enabled:
        removed = cf.controller.disarm(payload.run_id)
        return {
            "armed": False,
            "run_id": payload.run_id,
            "was_armed": removed is not None,
            "active": cf.controller.active_plans(),
        }
    plan = cf.controller.arm(
        run_id=payload.run_id,
        target_source=payload.target_source,
        failure_type=payload.failure_type,
        failure_count=payload.failure_count,
    )
    return {
        "armed": True,
        "plan": plan.to_dict(),
        "shape": plan.shape(),
        "active": cf.controller.active_plans(),
        "note": (
            "The next run started with this run_id will see a real error raised "
            "inside the production retry loop."
        ),
    }


# ─────────────────────────────────────────────────────────────
# The cycle
# ─────────────────────────────────────────────────────────────
@router.post("/improve", dependencies=[Depends(require_token), Depends(limit_heavy)])
async def improve(payload: ImproveRequest) -> dict[str, Any]:
    """Run the full cycle and return every stage's measured result."""
    report = await loop.execute(
        target_source=payload.target_source,
        failure_type=payload.failure_type,
        failure_count=payload.failure_count,
        case_id=payload.case_id,
        primary_metric=payload.primary_metric,
        simulation_mode=payload.simulation_mode,
        validate_with_evaluation=payload.validate_with_evaluation,
        repeats=payload.repeats,
    )
    _remember_cycle(report)
    return report


@router.post("/improve/stream", dependencies=[Depends(require_token), Depends(limit_heavy)])
async def improve_stream(payload: ImproveRequest) -> StreamingResponse:
    """Stream the cycle stage by stage, then the full report (SSE).

    The stages are slow enough (two real agent runs plus evaluation) that
    streaming is what makes the loop legible instead of a long blank wait.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def emit(name: str, data: dict[str, Any]) -> None:
        # Called from the loop's own coroutine, so a non-blocking put is correct;
        # dropping an event on a full queue is better than stalling the cycle.
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait({"type": name, **data})

    async def producer() -> dict[str, Any]:
        try:
            return await loop.execute(
                target_source=payload.target_source,
                failure_type=payload.failure_type,
                failure_count=payload.failure_count,
                case_id=payload.case_id,
                primary_metric=payload.primary_metric,
                simulation_mode=payload.simulation_mode,
                validate_with_evaluation=payload.validate_with_evaluation,
                repeats=payload.repeats,
                emit=emit,
            )
        finally:
            await queue.put({"type": "__eof__"})

    async def event_stream():
        task = asyncio.create_task(producer())
        yield _sse({
            "type": "stream_started",
            "target_source": payload.target_source,
            "failure_type": payload.failure_type,
            "failure_count": payload.failure_count,
            "primary_metric": payload.primary_metric,
        })
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    if task.done():
                        break
                    continue
                if event.get("type") == "__eof__":
                    break
                yield _sse(event)

            report = await task
            _remember_cycle(report)
            yield _sse({"type": "report", "report": report})
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        except Exception as exc:  # noqa: BLE001 — surface it, do not hide it
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cycles")
async def list_cycles() -> dict[str, Any]:
    """Previous cycle reports, newest first."""
    return {
        "cycles": [
            {
                "cycle_id": c.get("cycle_id"),
                "scenario": c.get("scenario"),
                "verdict": c.get("verdict"),
                "improvement_verified": c.get("improvement_verified"),
                "before_trace_id": c.get("before_trace_id"),
                "after_trace_id": c.get("after_trace_id"),
                "root_cause": (c.get("diagnosis") or {}).get("root_cause_type"),
                "confidence": (c.get("diagnosis") or {}).get("confidence"),
                "changed_parameter": (c.get("plan") or {}).get("changed_parameter"),
                "reverted": bool(c.get("reverted")),
                "started_at": c.get("started_at"),
                "completed_at": c.get("completed_at"),
            }
            for c in reversed(_CYCLES.values())
        ],
        "count": len(_CYCLES),
    }


@router.get("/cycles/{cycle_id}")
async def get_cycle(cycle_id: str) -> dict[str, Any]:
    report = _CYCLES.get(cycle_id)
    if report is None:
        raise HTTPException(status_code=404, detail="cycle not found")
    return report


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
