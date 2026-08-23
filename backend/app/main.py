"""FastAPI application for InsightPulse AI.

Task 1 scope: serve the agent core and a test interface.
    GET  /               → the test UI
    GET  /health         → liveness + capability report
    GET  /docs           → OpenAPI
    POST /api/agent/run  → run the agent
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import guard
from .api.agent import router as agent_router
from .api.evaluation import router as evaluation_router
from .api.graph import router as graph_router
from .api.observability import router as observability_router
from .api.report import router as report_router
from .config import settings
from .tools.registry import tool_registry

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Responses that must never be cached by a CDN or browser, because they reflect
# live run state. Vercel proxies the frontend, not these, but a stale 200 from an
# intermediary would misreport the system.
_NO_STORE_PREFIXES = ("/api/", "/health")


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

    @app.middleware("http")
    async def _guard_and_headers(request: Request, call_next):
        """Reject oversized bodies early, then add hardening response headers.

        The body cap is enforced from `Content-Length` before the route parses
        anything, so a large payload cannot be buffered just to fail validation.
        """
        cap = int(getattr(settings, "max_request_bytes", 0) or 0)
        if cap and request.method in {"POST", "PUT", "PATCH"}:
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > cap:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {cap} bytes."},
                )

        response = await call_next(request)

        # This service returns JSON and generated documents; it never needs to be
        # framed, sniffed, or to leak a referrer to a third-party provider.
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        if request.url.path.startswith(_NO_STORE_PREFIXES):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    app.include_router(agent_router)
    app.include_router(graph_router)
    app.include_router(evaluation_router)
    app.include_router(report_router)
    app.include_router(observability_router)

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
                # What protects the run endpoints when no token is configured.
                "limits": guard.status_report(),
            }
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
