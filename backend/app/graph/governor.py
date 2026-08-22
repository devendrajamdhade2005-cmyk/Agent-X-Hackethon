"""Resource governor + loop / deadlock detection.

Two safety systems that the routing logic consults so the graph adapts under
constraint and can never spin forever:

  * `ResourceGovernor` — tracks tool calls, LLM calls, graph steps, wall-clock and
    an estimated USD cost against a budget, and answers "can we still afford X?".
    Under pressure the router uses it to drop low-value work and keep only the
    highest-impact verification.

  * `ProgressMonitor` — an explicit progress check independent of LangGraph's
    recursion limit. It looks for repeated actions that produce no new state and
    repeated plan versions, and flags a deadlock so the graph can break out and
    terminate gracefully with an explanation.

Counts are derived from the state's executed-work lists (`tool_executions`,
`route_decisions`) rather than from a scalar counter, because parallel branches
update state concurrently and a raced integer would be wrong. Deriving from the
reduced lists is race-free.
"""

from __future__ import annotations

import time
from typing import Any

# Rough per-call cost estimate for the resource picture (USD). Deterministic and
# transparent — this is a governance signal, not a billing system.
COST_PER_TOOL_CALL = 0.002
COST_PER_LLM_CALL = 0.010


class ResourceGovernor:
    def __init__(self, budget: dict[str, Any], *, started_at: float | None = None) -> None:
        self.budget = budget
        self.started_at = started_at if started_at is not None else time.monotonic()

    # ── measurement ─────────────────────────────────────────
    @staticmethod
    def tool_calls(state: dict) -> int:
        return len(state.get("tool_executions") or [])

    @staticmethod
    def llm_calls(state: dict) -> int:
        return int(state.get("llm_call_count") or 0)

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    def estimated_cost(self, state: dict) -> float:
        return round(
            self.tool_calls(state) * COST_PER_TOOL_CALL
            + self.llm_calls(state) * COST_PER_LLM_CALL,
            4,
        )

    def snapshot(self, state: dict) -> dict[str, Any]:
        tools = self.tool_calls(state)
        return {
            "tool_calls": tools,
            "max_tool_calls": self.budget.get("max_tool_calls"),
            "tool_calls_remaining": max(0, self.budget.get("max_tool_calls", 0) - tools),
            "llm_calls": self.llm_calls(state),
            "graph_steps": int(state.get("graph_step_count") or 0),
            "max_graph_steps": self.budget.get("max_graph_steps"),
            "replans": int(state.get("replan_count") or 0),
            "max_replans": self.budget.get("max_replans"),
            "elapsed_ms": self.elapsed_ms(),
            "estimated_cost": self.estimated_cost(state),
            "usd_ceiling": self.budget.get("usd_ceiling"),
        }

    # ── affordability ───────────────────────────────────────
    def can_afford_tools(self, state: dict, n: int = 1) -> bool:
        return self.tool_calls(state) + n <= self.budget.get("max_tool_calls", 10)

    def budget_pressure(self, state: dict) -> float:
        """0.0 = plenty left, 1.0 = exhausted. Drives triage decisions."""
        limit = max(1, self.budget.get("max_tool_calls", 10))
        return min(1.0, self.tool_calls(state) / limit)

    def hard_limit_hit(self, state: dict) -> tuple[bool, str]:
        """Non-negotiable stops. Returns (hit, reason)."""
        if int(state.get("graph_step_count") or 0) >= self.budget.get("max_graph_steps", 60):
            return True, "graph step limit reached"
        if self.elapsed_ms() >= self.budget.get("max_runtime_seconds", 120) * 1000:
            return True, "runtime budget exhausted"
        if self.estimated_cost(state) >= self.budget.get("usd_ceiling", 0.5):
            return True, "cost ceiling reached"
        return False, ""

    def can_replan(self, state: dict) -> bool:
        return int(state.get("replan_count") or 0) < self.budget.get("max_replans", 2)

    def can_verify(self, state: dict) -> bool:
        return int(state.get("verify_count") or 0) < self.budget.get("max_verifications", 2)


class ProgressMonitor:
    """Explicit deadlock detection, independent of the recursion limit."""

    REPEAT_THRESHOLD = 3   # same signature this many times with no new findings

    @staticmethod
    def signature(stage: str, state: dict) -> str:
        """A fingerprint of 'where we are and what we know'. If it repeats without
        findings growing, we are not making progress."""
        return (
            f"{stage}|plan={state.get('plan_version', 0)}"
            f"|tasks={len(state.get('current_tasks') or [])}"
            f"|findings={len(state.get('findings') or [])}"
            f"|conflicts={len(state.get('conflicting_evidence') or [])}"
        )

    @classmethod
    def is_deadlocked(cls, state: dict, next_stage: str) -> tuple[bool, str]:
        sig = cls.signature(next_stage, state)
        history = [p.get("signature") for p in (state.get("progress_history") or [])]
        repeats = sum(1 for s in history if s == sig)
        if repeats >= cls.REPEAT_THRESHOLD:
            return True, (
                f"no measurable progress after {repeats} passes through '{next_stage}' "
                f"at plan v{state.get('plan_version', 0)} with "
                f"{len(state.get('findings') or [])} finding(s)"
            )
        # Repeated identical actions (e.g. the same tool+query) is the other classic
        # loop. The specialists suppress duplicate tool signatures themselves, so
        # this is a backstop at the graph level.
        actions = state.get("action_history") or []
        if len(actions) >= cls.REPEAT_THRESHOLD:
            tail = actions[-cls.REPEAT_THRESHOLD:]
            if len(set(tail)) == 1:
                return True, f"same action repeated {cls.REPEAT_THRESHOLD}×: {tail[-1]}"
        return False, ""

    @classmethod
    def progress_entry(cls, stage: str, state: dict) -> dict[str, Any]:
        return {
            "stage": stage,
            "signature": cls.signature(stage, state),
            "findings": len(state.get("findings") or []),
            "plan_version": state.get("plan_version", 0),
        }
