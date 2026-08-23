"""LangGraph agent framework API (Task 5).

    POST /api/agent/graph/run            run the graph, return the full result
    POST /api/agent/graph/run/stream     same, streaming framework events (SSE)
    POST /api/agent/graph/adversarial    run a named adversarial scenario (SSE)
    GET  /api/agent/graph/info           the graph topology, for visualisation

These are additive: the classic `/api/agent/*` routes are untouched, so every
existing client keeps working. Runs are stored in the same in-memory registry as
classic runs, so the report builder and run history work unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..graph.adversarial import AdversarialConfig
from ..graph.runner import run_graph
from ..security import clean_terms, clean_text
from .agent import remember_dict, require_token
from .guard import ConcurrencyGate, limit_run

router = APIRouter(prefix="/api/agent/graph", tags=["agent-framework"])


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class GraphRunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=600)
    keywords: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=10, ge=1, le=25)
    simulation_mode: bool | None = None
    # Adversarial demo: either a named scenario or an explicit config.
    adversarial: bool = False
    scenario: str = "full"
    adversarial_config: dict[str, Any] | None = None

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

    def resolve_adversarial(self) -> AdversarialConfig | None:
        if self.adversarial_config:
            cfg = AdversarialConfig.from_dict(self.adversarial_config)
            cfg.enabled = True
            return cfg
        if self.adversarial:
            comp = self.competitors[0] if self.competitors else "OpenAI"
            return AdversarialConfig.named(self.scenario, comp)
        return None


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@router.post("/run", dependencies=[Depends(require_token), Depends(limit_run)])
async def graph_run(payload: GraphRunRequest) -> dict[str, Any]:
    """Run the full LangGraph orchestration and return everything."""
    async with ConcurrencyGate("graph run"):
        result = await run_graph(
            payload.goal,
            keywords=payload.keywords,
            competitors=payload.competitors,
            topics=payload.topics,
            simulation_mode=payload.simulation_mode,
            max_iterations=payload.max_iterations,
            adversarial=payload.resolve_adversarial(),
        )
    remember_dict(result)
    return result


@router.post("/run/stream", dependencies=[Depends(require_token), Depends(limit_run)])
async def graph_run_stream(payload: GraphRunRequest) -> StreamingResponse:
    """Same run, streaming each framework event as it happens (SSE)."""
    return _stream(payload)


@router.post("/adversarial", dependencies=[Depends(require_token), Depends(limit_run)])
async def graph_adversarial(payload: GraphRunRequest) -> StreamingResponse:
    """Run a named adversarial scenario with streaming events."""
    payload.adversarial = True
    return _stream(payload)


@router.get("/info")
async def graph_info() -> dict[str, Any]:
    """The static graph topology, for the frontend visualisation."""
    return {
        "runtime": "langgraph",
        "nodes": [
            {"id": "understand", "label": "Understand", "kind": "understand"},
            {"id": "plan", "label": "Dynamic Planner", "kind": "plan"},
            {"id": "decompose", "label": "Task Decomposer", "kind": "plan"},
            {"id": "resource_check", "label": "Resource / Policy", "kind": "govern"},
            {"id": "dispatch", "label": "Dynamic Router", "kind": "router"},
            {"id": "research_agent", "label": "Research Agent", "kind": "agent"},
            {"id": "competitive_agent", "label": "Competitive Agent", "kind": "agent"},
            {"id": "observer", "label": "Observer", "kind": "join"},
            {"id": "conflict_resolution", "label": "Conflict Resolution", "kind": "verify"},
            {"id": "self_evaluator", "label": "Self-Evaluator", "kind": "evaluate"},
            {"id": "verify", "label": "Verification", "kind": "verify"},
            {"id": "replan", "label": "Replanner", "kind": "plan"},
            {"id": "finalize", "label": "Final Synthesis", "kind": "final"},
            {"id": "memory_update", "label": "Memory Update", "kind": "memory"},
        ],
        "edges": [
            {"from": "understand", "to": "plan"},
            {"from": "plan", "to": "decompose"},
            {"from": "decompose", "to": "resource_check"},
            {"from": "resource_check", "to": "dispatch"},
            {"from": "dispatch", "to": "research_agent", "conditional": True},
            {"from": "dispatch", "to": "competitive_agent", "conditional": True},
            {"from": "research_agent", "to": "observer"},
            {"from": "competitive_agent", "to": "observer"},
            {"from": "observer", "to": "conflict_resolution"},
            {"from": "conflict_resolution", "to": "self_evaluator"},
            {"from": "self_evaluator", "to": "verify", "conditional": True},
            {"from": "self_evaluator", "to": "replan", "conditional": True},
            {"from": "self_evaluator", "to": "finalize", "conditional": True},
            {"from": "verify", "to": "conflict_resolution"},
            {"from": "replan", "to": "dispatch", "conditional": True},
            {"from": "finalize", "to": "memory_update"},
        ],
        "scenarios": ["full", "tool_failure", "conflict", "budget"],
    }


# ─────────────────────────────────────────────────────────────
# SSE plumbing (mirrors the classic /api/agent/run/stream contract)
# ─────────────────────────────────────────────────────────────
def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _stream(payload: GraphRunRequest) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    adversarial = payload.resolve_adversarial()

    async def producer() -> dict[str, Any]:
        try:
            # Held for the run only, so a slow SSE reader does not keep a slot.
            async with ConcurrencyGate("graph run"):
                return await run_graph(
                    payload.goal,
                    keywords=payload.keywords,
                    competitors=payload.competitors,
                    topics=payload.topics,
                    simulation_mode=payload.simulation_mode,
                    max_iterations=payload.max_iterations,
                    adversarial=adversarial,
                    queue=queue,
                )
        finally:
            await queue.put({"type": "__eof__"})

    async def event_stream():
        task = asyncio.create_task(producer())
        yield _sse({"type": "run_started", "goal": payload.goal,
                    "runtime": "langgraph",
                    "adversarial": bool(adversarial and adversarial.enabled)})
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

            result = await task
            remember_dict(result)
            yield _sse({"type": "result", "result": result})
        except asyncio.CancelledError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        except Exception as exc:  # noqa: BLE001
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
