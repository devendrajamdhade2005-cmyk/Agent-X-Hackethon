"""In-process pub/sub + WebSocket hub.

The agent runs in a background task; the dashboard subscribes over /ws.
Every node transition, insight, alert and source failure is published here,
which is what makes the Activity Log feel live instead of polled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from datetime import UTC, datetime
from typing import Any

MAX_REPLAY = 200


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class EventBus:
    """Fan-out bus with a small replay buffer so late subscribers see recent history."""

    def __init__(self, max_queue: int = 500) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=MAX_REPLAY)
        self._max_queue = max_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seq = 0

    # ── lifecycle ───────────────────────────────────────────
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._recent)
        return items[-limit:]

    # ── publishing ──────────────────────────────────────────
    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Thread-safe: callable from sync agent code running in a worker thread."""
        self._seq += 1
        event = {
            "seq": self._seq,
            "type": event_type,
            "ts": _now_iso(),
            "payload": payload or {},
        }
        self._recent.append(event)

        if not self._subscribers:
            return event

        loop = self._loop
        if loop is None or loop.is_closed():
            return event

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is loop:
            self._dispatch(event)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._dispatch, event)
        return event

    def _dispatch(self, event: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow client: drop the oldest item rather than blocking the agent.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)


bus = EventBus()


def encode(event: dict[str, Any]) -> str:
    return json.dumps(event, default=str)
