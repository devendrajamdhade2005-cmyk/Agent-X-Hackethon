"""Runtime optimization policy — the only thing an improvement is allowed to change.

The improvement engine may not edit source code. It edits *this*: a small, typed,
versioned set of runtime parameters that the existing resilience and tool layers
consult. That makes every improvement traceable, reversible and testable, and keeps
the blast radius of an automated change to a handful of numbers.

`retry_attempts_by_source` is the parameter the rate-limit scenario tunes. The
resilience layer already computed its attempt ceiling as
`getattr(connector, "max_attempts", None) or MAX_ATTEMPTS`, so overriding it per
source is a configuration decision at exactly the point the code already allowed for
one — no behavioural rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# What the improvement engine is permitted to touch (section 22).
IMPROVEMENT_TYPES: tuple[str, ...] = (
    "RETRY_POLICY",
    "FALLBACK_ORDER",
    "TOOL_ROUTING",
    "TOOL_SELECTION_THRESHOLD",
    "EVIDENCE_THRESHOLD",
    "CACHE_DEDUP",
    "TIMEOUT",
    "PARALLELISM",
    "PROMPT_VERSION",
    "RESOURCE_POLICY",
)

# Hard bounds. An automated change can never move a parameter outside these, so a
# bad diagnosis cannot disable retries entirely or set an absurd timeout.
BOUNDS: dict[str, tuple[float, float]] = {
    "retry_attempts": (1, 5),
    "timeout_seconds": (3.0, 40.0),
    "dedup_window": (0, 50),
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class OptimizationPolicy:
    """The live policy. Version 0 is the project's original behaviour."""

    version: int = 0
    # Per-source retry ceiling. Empty = use the resilience default (MAX_ATTEMPTS).
    retry_attempts_by_source: dict[str, int] = field(default_factory=dict)
    # Per-source timeout override in seconds.
    timeout_by_source: dict[str, float] = field(default_factory=dict)
    # Suppress a repeat provider call for an identical query within one run.
    dedup_identical_tool_calls: bool = False
    # Free-form (still bounded) extras for future improvement types.
    extras: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now)
    reason: str = "initial default policy"

    # ── reads used by the runtime ───────────────────────────
    def retry_attempts_for(self, source: str, default: int) -> int:
        value = self.retry_attempts_by_source.get(source)
        if value is None:
            return default
        low, high = BOUNDS["retry_attempts"]
        return int(max(low, min(float(value), high)))

    def timeout_for(self, source: str, default: float) -> float:
        value = self.timeout_by_source.get(source)
        if value is None:
            return default
        low, high = BOUNDS["timeout_seconds"]
        return float(max(low, min(float(value), high)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "retry_attempts_by_source": dict(self.retry_attempts_by_source),
            "timeout_by_source": dict(self.timeout_by_source),
            "dedup_identical_tool_calls": self.dedup_identical_tool_calls,
            "extras": dict(self.extras),
            "updated_at": self.updated_at,
            "reason": self.reason,
        }

    def clone(self) -> "OptimizationPolicy":
        return OptimizationPolicy(
            version=self.version,
            retry_attempts_by_source=dict(self.retry_attempts_by_source),
            timeout_by_source=dict(self.timeout_by_source),
            dedup_identical_tool_calls=self.dedup_identical_tool_calls,
            extras=dict(self.extras),
            updated_at=self.updated_at,
            reason=self.reason,
        )


class PolicyRegistry:
    """Holds the active policy and its history, so any change can be reverted."""

    def __init__(self) -> None:
        self._active = OptimizationPolicy()
        self._history: list[dict[str, Any]] = [self._active.to_dict()]

    @property
    def active(self) -> OptimizationPolicy:
        return self._active

    @property
    def version(self) -> int:
        return self._active.version

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def apply(
        self,
        *,
        retry_attempts_by_source: dict[str, int] | None = None,
        timeout_by_source: dict[str, float] | None = None,
        dedup_identical_tool_calls: bool | None = None,
        extras: dict[str, Any] | None = None,
        reason: str = "",
    ) -> OptimizationPolicy:
        """Apply a bounded change as a new version. Returns the new policy."""
        nxt = self._active.clone()
        nxt.version = self._active.version + 1
        nxt.reason = reason or "automated improvement"
        nxt.updated_at = _now()

        low, high = BOUNDS["retry_attempts"]
        for source, attempts in (retry_attempts_by_source or {}).items():
            nxt.retry_attempts_by_source[str(source)] = int(
                max(low, min(float(attempts), high))
            )
        t_low, t_high = BOUNDS["timeout_seconds"]
        for source, seconds in (timeout_by_source or {}).items():
            nxt.timeout_by_source[str(source)] = float(
                max(t_low, min(float(seconds), t_high))
            )
        if dedup_identical_tool_calls is not None:
            nxt.dedup_identical_tool_calls = bool(dedup_identical_tool_calls)
        if extras:
            nxt.extras.update(extras)

        self._active = nxt
        self._history.append(nxt.to_dict())
        return nxt

    def revert(self) -> OptimizationPolicy:
        """Roll back to the previous version. Reversibility is a requirement."""
        if len(self._history) <= 1:
            return self._active
        self._history.pop()
        previous = self._history[-1]
        restored = OptimizationPolicy(
            version=int(previous.get("version", 0)),
            retry_attempts_by_source=dict(previous.get("retry_attempts_by_source") or {}),
            timeout_by_source=dict(previous.get("timeout_by_source") or {}),
            dedup_identical_tool_calls=bool(previous.get("dedup_identical_tool_calls")),
            extras=dict(previous.get("extras") or {}),
            reason=f"reverted to v{previous.get('version', 0)}",
        )
        self._active = restored
        return restored

    def reset(self) -> OptimizationPolicy:
        """Back to stock behaviour. Used between benchmark runs and by tests."""
        self._active = OptimizationPolicy()
        self._history = [self._active.to_dict()]
        return self._active


# Process-wide registry the runtime consults.
registry = PolicyRegistry()
