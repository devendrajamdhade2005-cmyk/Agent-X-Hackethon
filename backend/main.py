"""Entry point.

    python main.py                        # http://localhost:8000
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from app.main import app  # re-exported for `uvicorn main:app`

if __name__ == "__main__":
    import uvicorn

    from app.config import settings

    print(f"InsightPulse AI {settings.app_version}")
    print(f"  reasoner : {'claude (' + settings.anthropic_model + ')' if settings.llm_enabled else 'heuristic fallback (no ANTHROPIC_API_KEY set)'}")
    print(f"  UI       : http://localhost:8000")
    print(f"  API docs : http://localhost:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
