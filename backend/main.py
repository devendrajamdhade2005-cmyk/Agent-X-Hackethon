"""Entry point.

    python main.py                        # http://localhost:8000
    uvicorn main:app --reload --port 8000

On a platform host (Railway, Render, Fly, Heroku) the port is assigned at runtime
and injected as $PORT, so it is read from the environment rather than hardcoded.
Binding 0.0.0.0 is required for the platform's router to reach the container;
localhost-only binding makes the service look dead to its health check.
"""

from __future__ import annotations

import os

from app.main import app  # re-exported for `uvicorn main:app`


def _port() -> int:
    """Platform-assigned port, falling back to 8000 for local development."""
    raw = (os.getenv("PORT") or "").strip()
    if raw.isdigit():
        value = int(raw)
        if 1 <= value <= 65535:
            return value
    return 8000


if __name__ == "__main__":
    import uvicorn

    from app.config import settings

    port = _port()
    reasoner = (
        f"{settings.active_llm_provider} ({settings.gemini_model})"
        if settings.llm_enabled
        else "heuristic fallback (no LLM key set)"
    )
    print(f"InsightPulse AI {settings.app_version}")
    print(f"  reasoner : {reasoner}")
    print(f"  binding  : 0.0.0.0:{port}")
    print(f"  UI       : http://localhost:{port}")
    print(f"  API docs : http://localhost:{port}/docs")
    print(f"  health   : http://localhost:{port}/health")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
