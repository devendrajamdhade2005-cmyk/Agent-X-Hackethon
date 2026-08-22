"""Evaluation API (Task 6).

    GET  /api/evaluation/cases          benchmark dataset + scenario coverage
    GET  /api/evaluation/metrics        metric catalogue (methodology) + latest values
    POST /api/evaluation/run            run a suite and return the result
    POST /api/evaluation/run/stream     same, streaming live progress (SSE)
    POST /api/evaluation/repeat         repeated-run mode for one case
    GET  /api/evaluation/runs           evaluation runs from the latest suite
    GET  /api/evaluation/runs/{id}      one evaluation run + human review comparison
    GET  /api/evaluation/baseline       latest baseline comparison
    GET  /api/evaluation/history        suite history + regression comparison
    GET  /api/evaluation/human          pending / completed human reviews
    POST /api/evaluation/human-review   submit a reviewer's scores

Additive only — nothing about the existing agent, graph or report routes changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..evaluation import dataset, human, metrics as M, store
from ..evaluation.reports import (
    build_evaluation_report,
    render_evaluation_html,
    render_evaluation_markdown,
)
from ..evaluation.runner import MAX_REPEATS, SuiteRunner
from ..evaluation.schemas import HumanEvaluation, Thresholds
from ..security import clean_text
from .agent import require_token

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

VALID_MODES = {"demo", "full", "single", "repeated", "adversarial", "scenario"}


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    mode: str = "demo"
    case_ids: list[str] = Field(default_factory=list)
    scenario: str = ""
    repeats: int | None = Field(default=None, ge=1, le=MAX_REPEATS)
    include_baseline: bool = True
    simulation_mode: bool = True
    thresholds: dict[str, Any] | None = None

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = (v or "demo").strip().lower()
        if v not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        return v

    @field_validator("scenario")
    @classmethod
    def _scenario(cls, v: str) -> str:
        return (v or "").strip().upper()

    @field_validator("case_ids")
    @classmethod
    def _cases(cls, v: list[str]) -> list[str]:
        return [clean_text(str(c), max_len=32) for c in (v or [])][:20]


class RepeatRequest(BaseModel):
    case_id: str = Field(min_length=3, max_length=32)
    repeats: int = Field(default=3, ge=2, le=MAX_REPEATS)
    simulation_mode: bool = True


class HumanReviewRequest(BaseModel):
    evaluation_run_id: str = Field(min_length=3, max_length=64)
    reviewer_id: str = Field(default="reviewer-1", max_length=64)
    accuracy_score: int = Field(default=3, ge=1, le=5)
    completion_score: int = Field(default=3, ge=1, le=5)
    evidence_score: int = Field(default=3, ge=1, le=5)
    groundedness_score: int = Field(default=3, ge=1, le=5)
    uncertainty_score: int = Field(default=3, ge=1, le=5)
    actionability_score: int = Field(default=3, ge=1, le=5)
    overall_score: int = Field(default=3, ge=1, le=5)
    decision: str = "PARTIAL"
    comment: str = ""

    @field_validator("decision")
    @classmethod
    def _decision(cls, v: str) -> str:
        v = (v or "PARTIAL").strip().upper()
        if v not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError("decision must be PASS, PARTIAL or FAIL")
        return v

    @field_validator("comment")
    @classmethod
    def _comment(cls, v: str) -> str:
        return clean_text(v or "", max_len=1200)

    @field_validator("reviewer_id")
    @classmethod
    def _reviewer(cls, v: str) -> str:
        return clean_text(v or "reviewer-1", max_len=64) or "reviewer-1"


# ─────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────
@router.get("/cases")
async def list_cases() -> dict[str, Any]:
    """The benchmark dataset and its scenario coverage."""
    return {
        "cases": [c.to_dict() for c in dataset.all_cases()],
        "demo_suite": [c.case_id for c in dataset.demo_suite()],
        "coverage": dataset.coverage(),
        "scenario_types": list(dataset.SCENARIO_TYPES),
        "max_repeats": MAX_REPEATS,
    }


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Metric methodology, plus the latest measured values when a suite has run."""
    latest = store.latest_suite()
    return {
        "methodology": M.catalogue_dicts(),
        "thresholds": Thresholds().to_dict(),
        "latest": (latest or {}).get("aggregate") or {},
        "latest_suite_id": (latest or {}).get("suite_id"),
        "scenario_matrix": (latest or {}).get("scenario_matrix") or {},
        "counts": (latest or {}).get("counts") or {},
        "has_data": bool(latest),
        "empty_state": None if latest else "No evaluation has been run yet. Run the evaluation suite.",
    }


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    latest = store.latest_suite()
    if not latest:
        return {"runs": [], "suite_id": None,
                "empty_state": "No evaluation has been run yet. Run the evaluation suite."}
    return {
        "suite_id": latest.get("suite_id"),
        "runs": [
            {
                "evaluation_run_id": r.get("evaluation_run_id"),
                "case_id": r.get("case_id"),
                "case_name": r.get("case_name"),
                "scenario_type": r.get("scenario_type"),
                "system": r.get("system"),
                "repeat_index": r.get("repeat_index"),
                "outcome": r.get("outcome"),
                "status": r.get("status"),
                "agent_run_id": r.get("agent_run_id"),
                "gate_failures": r.get("gate_failures"),
                "metrics": {
                    name: (r.get("metrics") or {}).get(name)
                    for name in (M.ACCURACY, M.TASK_COMPLETION, M.GROUNDEDNESS,
                                 M.HALLUCINATION_RATE, M.RECOVERY_RATE,
                                 M.EVIDENCE_QUALITY, M.LATENCY)
                },
                "reviewer_count": len(store.human_reviews(r.get("evaluation_run_id") or "")),
            }
            for r in (latest.get("runs") or [])
        ],
    }


@router.get("/runs/{evaluation_run_id}")
async def get_run(evaluation_run_id: str) -> dict[str, Any]:
    run = store.get_run(evaluation_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"evaluation run '{evaluation_run_id}' not found")
    return {
        "run": run,
        "human": human.aggregate(evaluation_run_id),
        "human_vs_automated": human.compare_with_automated(
            evaluation_run_id, run.get("metrics") or {}
        ),
    }


@router.get("/baseline")
async def get_baseline() -> dict[str, Any]:
    latest = store.latest_suite()
    if not latest:
        return {"comparison": {}, "empty_state": "No evaluation has been run yet."}
    return {
        "suite_id": latest.get("suite_id"),
        "comparison": latest.get("baseline_comparison") or {},
        "note": (
            "Baselines run the same cases as InsightPulse wherever that is fair. "
            "Metrics that cannot apply to a baseline are reported unavailable with a "
            "reason rather than scored."
        ),
    }


@router.get("/history")
async def get_history() -> dict[str, Any]:
    latest = store.latest_suite()
    return {
        "history": store.history(),
        "regression": (latest or {}).get("regression") or {},
        "storage": store.status(),
    }


@router.get("/report")
async def evaluation_report(suite_id: str = "", format: str = "json") -> Any:
    """Evaluation report export: json (default), md or html."""
    suite = store.get_suite(suite_id) if suite_id else store.latest_suite()
    if suite is None:
        raise HTTPException(
            status_code=404,
            detail="no evaluation suite available — run an evaluation first",
        )
    report = build_evaluation_report(suite)
    fmt = (format or "json").strip().lower()
    if fmt == "md":
        return Response(
            content=render_evaluation_markdown(report),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition":
                    f'attachment; filename="InsightPulse-Evaluation-{report["suite_id"]}.md"'
            },
        )
    if fmt == "html":
        return HTMLResponse(render_evaluation_html(report))
    return {"report": report}


@router.get("/human")
async def human_queue() -> dict[str, Any]:
    latest = store.latest_suite()
    queue = human.pending_and_completed(latest)
    return {
        "suite_id": (latest or {}).get("suite_id"),
        **queue,
        "review_count": store.human_review_count(),
    }


# ─────────────────────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────────────────────
@router.post("/run", dependencies=[Depends(require_token)])
async def run_evaluation(payload: RunRequest) -> dict[str, Any]:
    """Execute an evaluation suite against the real agent and return the result."""
    runner = _runner(payload)
    if payload.mode == "adversarial":
        return await runner.run_adversarial()
    return await runner.run_suite(
        mode=payload.mode,
        case_ids=payload.case_ids or None,
        scenario=payload.scenario,
        repeats=payload.repeats,
        include_baseline=payload.include_baseline,
    )


@router.post("/repeat", dependencies=[Depends(require_token)])
async def repeat_case(payload: RepeatRequest) -> dict[str, Any]:
    """Repeated-run mode: reliability and consistency for one case."""
    if dataset.get_case(payload.case_id) is None:
        raise HTTPException(status_code=404, detail=f"case '{payload.case_id}' not found")
    runner = SuiteRunner(simulation_mode=payload.simulation_mode)
    return await runner.run_repeated(payload.case_id, repeats=payload.repeats)


@router.post("/human-review", dependencies=[Depends(require_token)])
async def submit_human_review(payload: HumanReviewRequest) -> dict[str, Any]:
    """Store one reviewer's scores for a completed evaluation run."""
    run = store.get_run(payload.evaluation_run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run '{payload.evaluation_run_id}' not found — run an evaluation first",
        )
    review = HumanEvaluation(
        evaluation_run_id=payload.evaluation_run_id,
        reviewer_id=payload.reviewer_id,
        accuracy_score=payload.accuracy_score,
        completion_score=payload.completion_score,
        evidence_score=payload.evidence_score,
        groundedness_score=payload.groundedness_score,
        uncertainty_score=payload.uncertainty_score,
        actionability_score=payload.actionability_score,
        overall_score=payload.overall_score,
        decision=payload.decision,  # type: ignore[arg-type]
        comment=payload.comment,
    )
    result = human.submit(review)
    result["human_vs_automated"] = human.compare_with_automated(
        payload.evaluation_run_id, run.get("metrics") or {}
    )
    return result


# ─────────────────────────────────────────────────────────────
# SSE (live suite progress)
# ─────────────────────────────────────────────────────────────
@router.post("/run/stream", dependencies=[Depends(require_token)])
async def run_evaluation_stream(payload: RunRequest) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    runner = _runner(payload, queue=queue)

    async def producer() -> dict[str, Any]:
        try:
            if payload.mode == "adversarial":
                return await runner.run_adversarial()
            return await runner.run_suite(
                mode=payload.mode,
                case_ids=payload.case_ids or None,
                scenario=payload.scenario,
                repeats=payload.repeats,
                include_baseline=payload.include_baseline,
            )
        finally:
            await queue.put({"type": "__eof__"})

    async def event_stream():
        task = asyncio.create_task(producer())
        yield _sse({"type": "run_started", "mode": payload.mode})
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


def _runner(payload: RunRequest, *, queue: Any = None) -> SuiteRunner:
    return SuiteRunner(
        thresholds=Thresholds.from_dict(payload.thresholds),
        simulation_mode=payload.simulation_mode,
        queue=queue,
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
