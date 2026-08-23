"""Inbound abuse protection for the public deployment.

The agent endpoints spend real money. One `POST /api/agent/run` makes live calls to
Gemini and several keyed providers; one `POST /api/observability/improve` with
`repeats=5` runs the graph ten times. Those endpoints are reachable without a
credential whenever `AGENT_API_TOKEN` is unset, which is how the hosted demo runs so
that a reviewer can try it with zero setup.

That combination — public URL, no credential, metered third-party keys — is the one
place where "open for the demo" turns into "anyone can drain the quota". This module
closes that without taking the demo away:

  * a **per-IP sliding window** bounds how much work one caller can start,
  * a **global concurrency cap** stops a burst from exhausting the container
    (the free Render tier has 512 MB and every concurrent run builds its own HTTP
    client, LLM client and memory manager),
  * a **body size cap** rejects oversized payloads before they are parsed.

Reads stay unmetered: they are cheap and the dashboard polls them. `/health` is
never limited because the platform uses it for liveness checks.

Deliberately dependency-free and in-process. A shared store like Redis would be the
right answer behind multiple replicas, and `Retry-After` is set so a client can back
off correctly either way. Every limit is configurable, and setting a limit to 0
disables it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict, deque
from typing import Any

from fastapi import HTTPException, Request, status

from ..config import settings

# Distinct cost tiers. `standard` is one agent run; `heavy` is an endpoint that can
# fan out into many runs, so it gets a much smaller allowance.
TIER_STANDARD = "standard"
TIER_HEAVY = "heavy"

# Keep at most this many client windows so the limiter cannot itself become a
# memory leak under a spray of forged addresses.
_MAX_CLIENTS = 2048


class SlidingWindowLimiter:
    """Per-key sliding window. Thread-safe; O(1) amortised per check."""

    def __init__(self) -> None:
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, *, limit: int, window_seconds: float) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, retry_after_seconds)."""
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            self._hits.move_to_end(key)
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= limit:
                # Oldest hit decides when a slot frees up.
                retry = max(1, int(hits[0] + window_seconds - now) + 1)
                return False, retry
            hits.append(now)
            while len(self._hits) > _MAX_CLIENTS:
                self._hits.popitem(last=False)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._hits.items()}


limiter = SlidingWindowLimiter()

# One global gate for expensive work. Built lazily because the value is read from
# settings and tests may change it.
_gate: asyncio.Semaphore | None = None
_gate_size: int | None = None
_active = 0


def _semaphore() -> asyncio.Semaphore | None:
    global _gate, _gate_size
    size = int(getattr(settings, "max_concurrent_runs", 4) or 0)
    if size <= 0:
        return None
    if _gate is None or _gate_size != size:
        _gate = asyncio.Semaphore(size)
        _gate_size = size
    return _gate


def client_key(request: Request) -> str:
    """Best-effort caller identity.

    Behind Render/Vercel the peer address is the proxy, so the forwarded chain is
    used when present. It is client-supplied and therefore spoofable — which is why
    the global concurrency cap exists as a backstop that no header can bypass.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return first[:64]
    real = request.headers.get("x-real-ip", "").strip()
    if real:
        return real[:64]
    return (request.client.host if request.client else "unknown")[:64]


def _limits_for(tier: str) -> tuple[int, float]:
    if tier == TIER_HEAVY:
        return (int(getattr(settings, "rate_limit_heavy_requests", 6) or 0),
                float(getattr(settings, "rate_limit_heavy_window_seconds", 900) or 900))
    return (int(getattr(settings, "rate_limit_run_requests", 20) or 0),
            float(getattr(settings, "rate_limit_run_window_seconds", 300) or 300))


async def _enforce(request: Request, tier: str) -> None:
    if not bool(getattr(settings, "rate_limit_enabled", True)):
        return
    # A caller presenting the configured token is the operator, not a stranger.
    expected = (settings.agent_api_token or "").strip()
    if expected and request.headers.get("x-api-token") == expected:
        return

    limit, window = _limits_for(tier)
    allowed, retry = limiter.check(f"{tier}:{client_key(request)}",
                                   limit=limit, window_seconds=window)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit reached for this endpoint ({limit} request(s) per "
                f"{int(window / 60)} min). These endpoints run the real agent against "
                f"metered third-party APIs, so they are capped on the public demo. "
                f"Retry in {retry}s, or set AGENT_API_TOKEN and send X-API-Token to "
                f"lift the cap on your own deployment."
            ),
            headers={"Retry-After": str(retry)},
        )


async def limit_run(request: Request) -> None:
    """Dependency for endpoints that execute one agent/graph run."""
    await _enforce(request, TIER_STANDARD)


async def limit_heavy(request: Request) -> None:
    """Dependency for endpoints that can fan out into many runs."""
    await _enforce(request, TIER_HEAVY)


class ConcurrencyGate:
    """Async context manager bounding how many runs execute at once.

    Refuses rather than queues indefinitely: a caller waiting behind a long queue
    on a small container is worse served than one told to retry.
    """

    def __init__(self, label: str = "run") -> None:
        self.label = label
        self._sem: asyncio.Semaphore | None = None

    async def __aenter__(self) -> "ConcurrencyGate":
        global _active
        self._sem = _semaphore()
        if self._sem is None:
            return self
        wait = float(getattr(settings, "max_concurrent_wait_seconds", 20) or 0)
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=wait)
        except (TimeoutError, asyncio.TimeoutError):
            self._sem = None
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"The agent is at capacity ({_gate_size} concurrent "
                    f"{self.label}(s)). This keeps the hosted container within its "
                    f"memory budget. Please retry shortly."
                ),
                headers={"Retry-After": "15"},
            ) from None
        _active += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        global _active
        if self._sem is not None:
            _active -= 1
            self._sem.release()


def status_report() -> dict[str, Any]:
    """What the guard is currently enforcing. Safe to expose: no secrets."""
    std_limit, std_window = _limits_for(TIER_STANDARD)
    heavy_limit, heavy_window = _limits_for(TIER_HEAVY)
    return {
        "rate_limit_enabled": bool(getattr(settings, "rate_limit_enabled", True)),
        "per_ip_run_limit": f"{std_limit} / {int(std_window / 60)} min",
        "per_ip_heavy_limit": f"{heavy_limit} / {int(heavy_window / 60)} min",
        "max_concurrent_runs": int(getattr(settings, "max_concurrent_runs", 4) or 0),
        "active_runs": _active,
        "max_request_bytes": int(getattr(settings, "max_request_bytes", 262_144) or 0),
        "token_required": bool((settings.agent_api_token or "").strip()),
        "tracked_clients": len(limiter.snapshot()),
    }


def reset() -> None:
    """Clear limiter state. Used between tests."""
    global _gate, _gate_size, _active
    limiter.reset()
    _gate = None
    _gate_size = None
    _active = 0
