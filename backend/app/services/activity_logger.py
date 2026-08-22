"""Agent Activity Log.

The log is a product surface, not debug output. It is what makes the agent's loop
visible: goal → plan → decision → action → observation → decision → … → insights.

Two important constraints:
  * Entries are structured (phase, title, detail, data) so a UI can render them
    and a test can assert on them.
  * Entries carry concise, user-facing decision summaries — what the agent is
    doing and why. Private model chain-of-thought is never surfaced here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Phase = Literal[
    "start",
    "goal",
    "plan",
    "orchestration",
    "delegation",
    "collaboration",
    "thought",
    "action",
    "observation",
    "decision",
    "warning",
    "error",
    "insight",
    "final",
    "done",
]

# Judged event taxonomy (§ activity log): ORCHESTRATION, DELEGATION, TOOL_CALL,
# OBSERVATION, COLLABORATION, RESULT, ERROR.
EVENT_TYPES: dict[str, str] = {
    "start": "ORCHESTRATION",
    "goal": "ORCHESTRATION",
    "plan": "ORCHESTRATION",
    "orchestration": "ORCHESTRATION",
    "delegation": "DELEGATION",
    "decision": "ORCHESTRATION",
    "action": "TOOL_CALL",
    "observation": "OBSERVATION",
    "thought": "OBSERVATION",
    "collaboration": "COLLABORATION",
    "insight": "RESULT",
    "final": "RESULT",
    "done": "RESULT",
    "warning": "ERROR",
    "error": "ERROR",
}

ICONS: dict[str, str] = {
    "start": "🤖",
    "orchestration": "🧭",
    "delegation": "📤",
    "collaboration": "🔄",
    "goal": "🎯",
    "plan": "🧠",
    "thought": "🧠",
    "action": "🔧",
    "observation": "👁",
    "decision": "🧭",
    "warning": "⚠️",
    "error": "❌",
    "insight": "📊",
    "final": "🧠",
    "done": "✅",
}

LABELS: dict[str, str] = {
    "start": "Agent started",
    "orchestration": "Orchestration",
    "delegation": "Delegation",
    "collaboration": "Collaboration",
    "goal": "Goal understood",
    "plan": "Planning",
    "thought": "Reasoning",
    "action": "Action",
    "observation": "Observation",
    "decision": "Decision",
    "warning": "Warning",
    "error": "Error",
    "insight": "Insights",
    "final": "Final decision",
    "done": "Task completed",
}


@dataclass
class ActivityEntry:
    seq: int
    phase: Phase
    title: str
    detail: str = ""
    iteration: int | None = None
    elapsed_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = ""
    agent: str = ""

    @property
    def icon(self) -> str:
        return ICONS.get(self.phase, "•")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "phase": self.phase,
            "icon": self.icon,
            "label": LABELS.get(self.phase, self.phase.title()),
            "event_type": EVENT_TYPES.get(self.phase, "RESULT"),
            "agent": self.agent,
            "title": self.title,
            "detail": self.detail,
            "iteration": self.iteration,
            "elapsed_ms": self.elapsed_ms,
            "data": self.data,
            "ts": self.ts,
        }

    def render(self) -> str:
        """Human-readable single entry, as shown in the terminal demo."""
        label = LABELS.get(self.phase, self.phase)
        head = f"{self.icon} {label}"
        if self.iteration:
            head += f" [step {self.iteration}]"
        # Avoid "Agent started: Agent started" when the title restates the label.
        line = head if not self.title or self.title == label else f"{head}: {self.title}"
        return f"{line}\n   {self.detail}" if self.detail else line


class ActivityLogger:
    """Collects entries, optionally pushing each one to live subscribers."""

    def __init__(
        self,
        run_id: str,
        *,
        sink: Callable[[ActivityEntry], None] | None = None,
        queue: asyncio.Queue | None = None,
        echo: bool = False,
    ) -> None:
        self.run_id = run_id
        self.entries: list[ActivityEntry] = []
        self._seq = 0
        self._sink = sink
        self._queue = queue
        self._echo = echo
        self._t0 = time.perf_counter()
        self._agent = ""

    def speaking_as(self, agent: str) -> None:
        """Attribute subsequent entries to this agent (orchestrator or specialist)."""
        self._agent = agent or ""

    # ── core ────────────────────────────────────────────────
    def log(
        self,
        phase: Phase,
        title: str,
        detail: str = "",
        *,
        iteration: int | None = None,
        **data: Any,
    ) -> ActivityEntry:
        self._seq += 1
        entry = ActivityEntry(
            seq=self._seq,
            phase=phase,
            title=title.strip(),
            detail=detail.strip(),
            iteration=iteration,
            elapsed_ms=int((time.perf_counter() - self._t0) * 1000),
            data=data,
            ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
            agent=str(data.pop("agent", "") or self._agent),
        )
        self.entries.append(entry)

        if self._echo:
            print(entry.render(), flush=True)
        if self._sink is not None:
            try:
                self._sink(entry)
            except Exception:  # noqa: BLE001 — logging must never break the agent
                pass
        if self._queue is not None:
            try:
                self._queue.put_nowait({"type": "activity", "entry": entry.to_dict()})
            except asyncio.QueueFull:
                pass
        return entry

    # ── phase shorthands (keeps agent code readable) ─────────
    def start(self, goal: str, **data: Any) -> ActivityEntry:
        return self.log("start", "Agent started", f"Goal: {goal}", **data)

    def goal(self, title: str, detail: str = "", **data: Any) -> ActivityEntry:
        return self.log("goal", title, detail, **data)

    def plan(self, title: str, detail: str = "", **data: Any) -> ActivityEntry:
        return self.log("plan", title, detail, **data)

    def thought(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("thought", title, detail, **kw)

    def action(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("action", title, detail, **kw)

    def observation(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("observation", title, detail, **kw)

    def decision(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("decision", title, detail, **kw)

    def warning(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("warning", title, detail, **kw)

    def error(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("error", title, detail, **kw)

    def insight(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("insight", title, detail, **kw)

    def final(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("final", title, detail, **kw)

    def orchestration(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("orchestration", title, detail, **kw)

    def delegation(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("delegation", title, detail, **kw)

    def collaboration(self, title: str, detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("collaboration", title, detail, **kw)

    def done(self, title: str = "Task completed", detail: str = "", **kw: Any) -> ActivityEntry:
        return self.log("done", title, detail, **kw)

    # ── output ──────────────────────────────────────────────
    def as_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def render(self) -> str:
        return "\n\n".join(e.render() for e in self.entries)

    def count(self, phase: Phase) -> int:
        return sum(1 for e in self.entries if e.phase == phase)
