"""Controlled failure injection — deterministic, bounded, and safe.

The requirement is a repeatable failure whose recovery is genuine. That rules out
waiting for a real provider to rate-limit us, and it also rules out the project's
existing `set_forced_failure` hook: that hook returns immediately, so it never
exercises the retry path we want to observe and then improve.

So this injector sits at the top of `collect_from_source` and hands the *real* retry
loop a fetch function that raises a real `SourceError(status=429)`. Everything after
that — the attempt ceiling, the jittered backoff, the circuit breaker, the
provider-level failure bookkeeping, the tool's fallback to its other providers — is
the production code path. Only the trigger is synthetic.

Safety properties:
  * Failure types are an enum, target sources come from the registered source list;
    no free-form callable or code path can be injected.
  * A plan is scoped to one run id, so a controlled failure in one execution cannot
    leak into a normal one.
  * `failure_count` bounds how many attempts fail, after which the injector steps
    aside and the source behaves normally again.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# Only these failure modes can be injected (section 40/50).
FAILURE_TYPES: tuple[str, ...] = ("rate_limit", "timeout", "server_error", "bad_response")

# HTTP status and error category per failure type.
_FAILURE_SHAPE: dict[str, dict[str, Any]] = {
    "rate_limit": {"status": 429, "category": "RATE_LIMIT", "retryable": True,
                   "message": "HTTP 429: rate limit exceeded (controlled failure)"},
    "timeout": {"status": None, "category": "TIMEOUT", "retryable": True,
                "message": "request timed out (controlled failure)"},
    "server_error": {"status": 503, "category": "HTTP_ERROR", "retryable": True,
                     "message": "HTTP 503: service unavailable (controlled failure)"},
    "bad_response": {"status": 200, "category": "BAD_RESPONSE", "retryable": False,
                     "message": "malformed provider payload (controlled failure)"},
}

MAX_FAILURE_COUNT = 5


@dataclass
class FailurePlan:
    """A scoped, deterministic injection plan."""

    run_id: str
    target_source: str
    failure_type: str = "rate_limit"
    failure_count: int = 2
    enabled: bool = True
    # How many attempts have actually been failed, per source.
    _fired: dict[str, int] = field(default_factory=dict)

    def shape(self) -> dict[str, Any]:
        return _FAILURE_SHAPE.get(self.failure_type, _FAILURE_SHAPE["rate_limit"])

    def should_fail(self, source: str) -> bool:
        if not self.enabled or source != self.target_source:
            return False
        return self._fired.get(source, 0) < self.failure_count

    def record_fired(self, source: str) -> int:
        n = self._fired.get(source, 0) + 1
        self._fired[source] = n
        return n

    def fired_count(self, source: str = "") -> int:
        if source:
            return self._fired.get(source, 0)
        return sum(self._fired.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_source": self.target_source,
            "failure_type": self.failure_type,
            "failure_count": self.failure_count,
            "enabled": self.enabled,
            "fired": dict(self._fired),
            "http_status": self.shape()["status"],
            "error_category": self.shape()["category"],
        }


class InjectionController:
    """Process-wide registry of active plans, keyed by run id.

    Keyed by run id rather than global on purpose: a controlled-failure run and a
    normal run can be in flight at the same time, and the normal one must be
    untouched (§28 of the brief).
    """

    def __init__(self) -> None:
        self._plans: dict[str, FailurePlan] = {}
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────
    def arm(
        self,
        *,
        run_id: str,
        target_source: str,
        failure_type: str = "rate_limit",
        failure_count: int = 2,
    ) -> FailurePlan:
        """Register a plan for one run. Inputs are validated by the caller/API."""
        plan = FailurePlan(
            run_id=run_id,
            target_source=target_source,
            failure_type=(
                failure_type if failure_type in FAILURE_TYPES else "rate_limit"
            ),
            failure_count=max(1, min(int(failure_count), MAX_FAILURE_COUNT)),
        )
        with self._lock:
            self._plans[run_id] = plan
        return plan

    def disarm(self, run_id: str) -> FailurePlan | None:
        with self._lock:
            return self._plans.pop(run_id, None)

    def plan_for(self, run_id: str) -> FailurePlan | None:
        return self._plans.get(run_id) if run_id else None

    def active_plans(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._plans.values()]

    def reset(self) -> None:
        with self._lock:
            self._plans.clear()


controller = InjectionController()


# ─────────────────────────────────────────────────────────────
# Run-scoped current run id
# ─────────────────────────────────────────────────────────────
from contextvars import ContextVar  # noqa: E402 — kept next to its use

_CURRENT_RUN: ContextVar[str] = ContextVar("_ip_obs_run_id", default="")


def set_current_run(run_id: str) -> None:
    """Bind the executing run so the resilience layer can find its plan."""
    _CURRENT_RUN.set(run_id or "")


def current_run() -> str:
    return _CURRENT_RUN.get()


def plan_for_current_run() -> FailurePlan | None:
    return controller.plan_for(_CURRENT_RUN.get())


def build_injected_error(plan: FailurePlan) -> Exception:
    """The exception the real retry loop will see.

    A genuine `SourceError` with a real HTTP status, so the existing retry/breaker
    logic classifies and handles it exactly as it would a live rate limit.
    """
    from ..sources.base import SourceError

    shape = plan.shape()
    if plan.failure_type == "timeout":
        return TimeoutError(shape["message"])
    return SourceError(
        shape["message"],
        retryable=bool(shape["retryable"]),
        status=shape["status"],
        retry_after=None,
    )


def available_targets() -> list[str]:
    """Registered source names that may be targeted. No arbitrary strings."""
    try:
        from ..sources.registry import registry as source_registry

        return sorted(c.name for c in source_registry.all())
    except Exception:  # noqa: BLE001
        return []
