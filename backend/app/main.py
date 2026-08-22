"""FastAPI application for InsightPulse AI.

Task 1 scope: serve the agent core and a test interface.
    GET  /               → the test UI
    GET  /health         → liveness + capability report
    GET  /docs           → OpenAPI
    POST /api/agent/run  → run the agent
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.agent import router as agent_router
from .config import settings
from .tools.registry import tool_registry

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="InsightPulse AI",
        version=settings.app_version,
        description=(
            "Autonomous research & competitor intelligence agent. Implements an "
            "explicit ReAct loop: goal → plan → decide → select tool → call → "
            "observe → analyze → decide … → prioritized insights."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(agent_router)

    @app.get("/health", tags=["ops"])
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "app": settings.app_name,
                "version": settings.app_version,
                "reasoner": settings.active_llm_provider
                if settings.llm_enabled
                else "heuristic-fallback",
                "simulation_mode": settings.simulation_mode,
                "tools": tool_registry.usable_names(),
                "capabilities": settings.capability_report(),
                "auth": "token required" if settings.agent_api_token else "open (local demo)",
            }
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
