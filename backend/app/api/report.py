"""Intelligence Report API.

    POST /api/report/generate              build a report from a finished run
    GET  /api/report                       list generated reports
    GET  /api/report/{id}                  report JSON
    GET  /api/report/{id}/preview          standalone print-ready HTML
    GET  /api/report/{id}/download/pdf     PDF file
    GET  /api/report/{id}/download/md      Markdown file
    GET  /api/report/{id}/download/json    JSON file

Reports are projections of an already-completed run. No tool is ever re-called and
no search is repeated to produce a document.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..reports import store
from ..reports.builder import build_report
from ..reports.html import render_html
from ..reports.markdown import render_markdown
from .agent import get_stored_run, latest_stored_run, require_token
from .guard import limit_run

router = APIRouter(prefix="/api/report", tags=["report"])


class GenerateRequest(BaseModel):
    # Omit to use the most recent completed run.
    run_id: str | None = Field(default=None, max_length=64)
    # "Generate again" bypasses the cache.
    force: bool = False


def _resolve_run(run_id: str | None) -> dict[str, Any]:
    run = get_stored_run(run_id) if run_id else latest_stored_run()
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No stored run found for id '{run_id}'."
                if run_id
                else "No completed agent run available yet. Run a scan first."
            ),
        )
    if run.get("status") == "failed":
        raise HTTPException(
            status_code=409, detail="That run failed, so there is nothing to report on."
        )
    return run


def _require_report(report_id: str) -> dict[str, Any]:
    report = store.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return report


def _filename(report: dict[str, Any], ext: str) -> str:
    return f"InsightPulse-Report-{report.get('report_id', 'report')}.{ext}"


# ─────────────────────────────────────────────────────────────
@router.post("/generate", dependencies=[Depends(require_token), Depends(limit_run)])
async def generate(payload: GenerateRequest) -> dict[str, Any]:
    run = _resolve_run(payload.run_id)
    run_id = run.get("run_id", "")

    if not payload.force:
        cached = store.get_by_run(run_id)
        if cached is not None:
            return {"report": cached, "cached": True}

    store.drop_run(run_id)
    report = store.save(build_report(run))
    return {"report": report, "cached": False}


@router.get("")
async def list_reports() -> dict[str, Any]:
    return {"reports": store.listing()}


@router.get("/{report_id}")
async def get_report(report_id: str) -> dict[str, Any]:
    return {"report": _require_report(report_id)}


@router.get("/{report_id}/preview", response_class=HTMLResponse)
async def preview(
    report_id: str,
    embedded: bool = Query(default=False, description="Hide the print bar (iframe use)"),
) -> HTMLResponse:
    report = _require_report(report_id)
    return HTMLResponse(render_html(report, embedded=embedded))


@router.get("/{report_id}/download/pdf")
async def download_pdf(report_id: str) -> Response:
    report = _require_report(report_id)
    try:
        from ..reports.pdf import render_pdf

        data = render_pdf(report)
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HTTPException(
            status_code=501,
            detail=(
                "PDF generation needs the 'reportlab' package. Install it, or use the "
                "Preview and print to PDF from the browser."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - never 500 on a formatting edge case
        raise HTTPException(
            status_code=500, detail=f"Could not render the PDF: {type(exc).__name__}: {exc}"
        ) from exc

    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{_filename(report, "pdf")}"',
            "Content-Length": str(len(data)),
        },
    )


@router.get("/{report_id}/download/md")
async def download_markdown(report_id: str) -> Response:
    report = _require_report(report_id)
    return Response(
        content=render_markdown(report),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{_filename(report, "md")}"'},
    )


@router.get("/{report_id}/download/json")
async def download_json(report_id: str) -> Response:
    report = _require_report(report_id)
    return Response(
        content=json.dumps(report, indent=2, ensure_ascii=False, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{_filename(report, "json")}"'},
    )


@router.delete("/{report_id}")
async def delete_report(report_id: str) -> JSONResponse:
    report = _require_report(report_id)
    store.drop_run(report.get("run_id", ""))
    return JSONResponse({"deleted": report_id})
