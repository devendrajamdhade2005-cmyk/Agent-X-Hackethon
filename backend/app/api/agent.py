"""Agent API.

    POST /api/agent/run          run the loop, return the full result
    POST /api/agent/run/stream   same, but stream the activity log live (SSE)
    GET  /api/agent/tools        the agent's action space and provider health
    GET  /api/agent/runs         recent runs
    GET  /api/agent/runs/{id}    one run, replayable
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..agents.agent import AgentRunRequest, AgentRunResult, InsightPulseAgent
from ..agents.state import MAX_ITERATIONS
from ..config import settings
from ..security import clean_terms, clean_text
from ..sources.resilience import registry as resilience_registry
from ..tools.registry import tool_registry

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Recent runs, newest last. In-memory on purpose: Task 1 is the agent core, and
# durable run history is a separate concern.
_RUNS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_MAX_RUNS = 25


def _remember(result: AgentRunResult) -> None:
    _RUNS[result.run_id] = result.to_dict()
    while len(_RUNS) > _MAX_RUNS:
        _RUNS.popitem(last=False)


def get_stored_run(run_id: str) -> dict[str, Any] | None:
    """Read a completed run. Used by the report builder.

    Reports are always built from an already-finished run — no tool is ever
    re-called to produce a document.
    """
    return _RUNS.get(run_id)


def latest_stored_run() -> dict[str, Any] | None:
    for run in reversed(_RUNS.values()):
        return run
    return None


# ─────────────────────────────────────────────────────────────
# Optional shared-secret gate
# ─────────────────────────────────────────────────────────────
async def require_token(x_api_token: str | None = Header(default=None)) -> None:
    """No-op unless AGENT_API_TOKEN is set.

    The agent endpoints are open by default so the local demo works with zero
    setup. Set AGENT_API_TOKEN before exposing this service on a network — the
    run endpoint spends API quota and makes outbound requests on demand.
    """
    expected = settings.agent_api_token.strip()
    if not expected:
        return
    if not x_api_token or x_api_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Token header",
        )


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=600)
    keywords: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=MAX_ITERATIONS, ge=1, le=25)
    simulation_mode: bool | None = None

    @field_validator("goal")
    @classmethod
    def _clean_goal(cls, v: str) -> str:
        cleaned = clean_text(v, max_len=600)
        if len(cleaned) < 3:
            raise ValueError("goal is too short")
        return cleaned

    @field_validator("keywords", "competitors", "topics")
    @classmethod
    def _clean_lists(cls, v: list[str]) -> list[str]:
        return clean_terms(v, max_items=10, max_len=120)

    def to_agent_request(self) -> AgentRunRequest:
        return AgentRunRequest(
            goal=self.goal,
            keywords=self.keywords,
            competitors=self.competitors,
            topics=self.topics,
            max_iterations=self.max_iterations,
            simulation_mode=self.simulation_mode,
        )


class RunResponse(BaseModel):
    status: str
    run_id: str
    goal: str
    activity_log: list[dict[str, Any]]
    tools_used: list[str]
    findings: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    summary: str
    state: dict[str, Any]
    metrics: dict[str, Any]
    # Multi-agent surface. Additive: every field above is unchanged, so existing
    # consumers keep working.
    execution_plan: list[dict[str, Any]] = Field(default_factory=list)
    agents: list[dict[str, Any]] = Field(default_factory=list)
    collaboration_events: list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@router.post("/run", response_model=RunResponse, dependencies=[Depends(require_token)])
async def run_agent_endpoint(payload: RunRequest) -> dict[str, Any]:
    """Run the full reason → decide → act → observe loop and return everything."""
    agent = InsightPulseAgent(simulation_mode=payload.simulation_mode)
    result = await agent.run(payload.to_agent_request())
    _remember(result)
    return result.to_dict()


@router.post("/run/stream", dependencies=[Depends(require_token)])
async def run_agent_stream(payload: RunRequest) -> StreamingResponse:
    """Same run, but each activity entry is pushed as it happens (SSE).

    This is what makes the loop legible in the UI: you watch the agent decide,
    call a tool, observe, and decide again — rather than waiting for a blob.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    agent = InsightPulseAgent(simulation_mode=payload.simulation_mode, queue=queue)

    async def producer() -> AgentRunResult:
        try:
            return await agent.run(payload.to_agent_request())
        finally:
            await queue.put({"type": "__eof__"})

    async def event_stream():
        task = asyncio.create_task(producer())
        yield _sse({"type": "run_started", "run_id": agent.state.run_id, "goal": payload.goal})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"  # stop proxies from closing the stream
                    if task.done():
                        break
                    continue
                if event.get("type") == "__eof__":
                    break
                yield _sse(event)

            result = await task
            _remember(result)
            yield _sse({"type": "result", "result": result.to_dict()})
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        except Exception as exc:  # noqa: BLE001 — surface the failure to the client
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


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """The agent's action space, plus which providers are live vs. simulated."""
    from ..agents.messages import ORCHESTRATOR, SPECIALISTS

    return {
        "tools": tool_registry.catalog(),
        "usable": tool_registry.usable_names(),
        "agents": [
            {
                **p.to_dict(),
                "tools_available": [
                    t for t in p.tool_names if t in tool_registry.usable_names()
                ],
            }
            for p in (ORCHESTRATOR, *SPECIALISTS)
        ],
        "provider_health": resilience_registry.snapshots(),
        "capabilities": settings.capability_report(),
        "simulation_mode": settings.simulation_mode,
        "max_iterations_default": MAX_ITERATIONS,
    }


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    return {
        "runs": [
            {
                "run_id": run["run_id"],
                "goal": run["goal"],
                "status": run["status"],
                "tools_used": run["tools_used"],
                "iterations": run["metrics"].get("iterations"),
                "insights": run["metrics"].get("insights"),
                "priority_counts": run["metrics"].get("priority_counts"),
                "started_at": run["state"].get("started_at"),
            }
            for run in reversed(_RUNS.values())
        ]
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
