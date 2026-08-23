"""Trace sinks — local always, external optionally.

`TraceProvider` is the seam that keeps the project independent of any observability
vendor. `LocalTraceProvider` is the always-on implementation: it keeps traces in
process and mirrors them to disk, which is what the built-in Observability dashboard
reads. `ExternalTraceProvider` is a thin, best-effort exporter (OTLP/LangSmith/
Langfuse-shaped) that is disabled unless explicitly configured.

The rule that matters: **an export failure is never allowed to affect execution.**
Every external call is wrapped, timed out, and on failure the provider records the
reason and marks itself degraded. The run continues on the local trace.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Protocol

from ..config import DATA_DIR

MAX_TRACES = 40
_TRACE_FILE = DATA_DIR / "traces.json"


class TraceProvider(Protocol):
    name: str
    degraded: str

    def record(self, trace: dict[str, Any]) -> None: ...
    def status(self) -> dict[str, Any]: ...


# ─────────────────────────────────────────────────────────────
class LocalTraceProvider:
    """In-process trace store with a best-effort disk mirror.

    Mirrors the project's existing persistence pattern (capped OrderedDict + JSON
    under DATA_DIR). Disk is a convenience, not a dependency: on a filesystem
    without durable storage the provider degrades to memory and says so.
    """

    name = "local"

    def __init__(self) -> None:
        self._traces: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self.degraded = ""
        self._loaded = False

    # ── writes ──────────────────────────────────────────────
    def record(self, trace: dict[str, Any]) -> None:
        self._load()
        trace_id = trace.get("trace_id")
        if not trace_id:
            return
        self._traces[trace_id] = trace
        while len(self._traces) > MAX_TRACES:
            self._traces.popitem(last=False)
        self._persist()

    def _persist(self) -> None:
        """Persist trace summaries only. Full spans stay in memory: a complete
        trace is large, and the dashboard's history view needs the headline, not
        every span of every historical run."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = [
                {
                    **{k: v for k, v in t.items() if k not in ("spans", "errors")},
                    "span_count": len(t.get("spans") or []),
                    "error_count": len(t.get("errors") or []),
                }
                for t in self._traces.values()
            ]
            _TRACE_FILE.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            self.degraded = ""
        except Exception as exc:  # noqa: BLE001 — storage must never break a run
            self.degraded = f"traces not persisted to disk ({type(exc).__name__})"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if _TRACE_FILE.is_file():
                data = json.loads(_TRACE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data:
                        tid = entry.get("trace_id")
                        if tid and tid not in self._traces:
                            entry.setdefault("spans", [])
                            entry.setdefault("errors", [])
                            entry["partial"] = True   # summary-only, restored from disk
                            self._traces[tid] = entry
        except Exception as exc:  # noqa: BLE001
            self.degraded = f"traces could not be read ({type(exc).__name__})"

    # ── reads ───────────────────────────────────────────────
    def get(self, trace_id: str) -> dict[str, Any] | None:
        self._load()
        return self._traces.get(trace_id)

    def by_run(self, run_id: str) -> dict[str, Any] | None:
        self._load()
        for trace in reversed(self._traces.values()):
            if trace.get("run_id") == run_id:
                return trace
        return None

    def list(self) -> list[dict[str, Any]]:
        self._load()
        return list(reversed(self._traces.values()))

    def latest(self, scenario: str = "") -> dict[str, Any] | None:
        self._load()
        for trace in reversed(self._traces.values()):
            if not scenario or trace.get("scenario") == scenario:
                return trace
        return None

    def reset(self) -> None:
        self._traces.clear()
        self._loaded = True
        self.degraded = ""

    def status(self) -> dict[str, Any]:
        self._load()
        return {
            "provider": self.name,
            "traces_stored": len(self._traces),
            "file": str(_TRACE_FILE),
            "degraded": self.degraded,
        }


# ─────────────────────────────────────────────────────────────
class ExternalTraceProvider:
    """Best-effort export to an external observability backend.

    Deliberately generic: the payload is OTLP-shaped (trace/span/attributes), so the
    same exporter serves an OTLP collector, LangSmith or Langfuse depending on the
    configured endpoint. It is inert unless an endpoint is configured, and any
    failure degrades silently rather than propagating.
    """

    name = "external"

    def __init__(self, endpoint: str = "", api_key: str = "", project: str = "") -> None:
        self.endpoint = (endpoint or "").strip()
        self._api_key = (api_key or "").strip()      # never exposed in status()
        self.project = (project or "").strip()
        self.degraded = "" if self.endpoint else "no external endpoint configured"
        self.exported = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def record(self, trace: dict[str, Any]) -> None:
        """Export without blocking the caller's critical path.

        The export is fire-and-forget: it is scheduled on the running loop when one
        exists, so a slow or unavailable collector cannot add latency to the agent.
        """
        if not self.enabled:
            return
        try:
            import asyncio

            payload = self._payload(trace)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(self._send(payload))
                # Consume the result so a failed export never raises unretrieved.
                task.add_done_callback(lambda t: t.exception())
            else:
                self.degraded = "no event loop available for export; trace kept locally"
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.degraded = f"export scheduling failed ({type(exc).__name__})"

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            import asyncio

            import httpx

            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await asyncio.wait_for(
                    client.post(self.endpoint, json=payload, headers=headers),
                    timeout=6.0,
                )
            if res.status_code >= 400:
                self.failures += 1
                self.degraded = (
                    f"external telemetry rejected the export (HTTP {res.status_code}); "
                    f"local trace retained"
                )
                return
            self.exported += 1
            self.degraded = ""
        except Exception as exc:  # noqa: BLE001 — export failure is not a run failure
            self.failures += 1
            self.degraded = (
                f"external telemetry unavailable ({type(exc).__name__}); "
                f"local trace retained"
            )

    def _payload(self, trace: dict[str, Any]) -> dict[str, Any]:
        """OTLP-ish projection. Attributes are already redacted by the tracer."""
        return {
            "resource": {
                "service.name": "insightpulse",
                "deployment.environment": trace.get("environment"),
                "project": self.project,
            },
            "trace_id": trace.get("trace_id"),
            "spans": [
                {
                    "traceId": s.get("trace_id"),
                    "spanId": s.get("span_id"),
                    "parentSpanId": s.get("parent_span_id"),
                    "name": s.get("name"),
                    "kind": s.get("kind"),
                    "startTime": s.get("start_time"),
                    "endTime": s.get("end_time"),
                    "durationMs": s.get("duration_ms"),
                    "status": s.get("status"),
                    "attributes": s.get("attributes") or {},
                    "events": s.get("events") or [],
                }
                for s in (trace.get("spans") or [])
            ],
        }

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "enabled": self.enabled,
            # The endpoint host is useful for debugging; the key never appears.
            "endpoint_configured": bool(self.endpoint),
            "project": self.project,
            "exported": self.exported,
            "failures": self.failures,
            "degraded": self.degraded,
        }


# Process-wide local provider (the dashboard reads this).
local_provider = LocalTraceProvider()
