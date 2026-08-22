"""LangGraph agent framework (Task 5).

InsightPulse's orchestration runtime. A `StateGraph` owns the shared state,
conditional routing, parallel fan-out, checkpointing and the observe → evaluate →
decide → act → replan loop. The specialist agents, tools, source resilience and
Task 4 memory are reused unchanged — LangGraph is the runtime that coordinates
them, not a rewrite of them.

Public surface:
  * `GraphState`         — the typed, checkpointable shared state.
  * `build_graph`        — compile the StateGraph with a checkpointer.
  * `run_graph`          — high-level entry point (used by the API and tests).
  * `AdversarialConfig`  — controlled fault injection for the adversarial demo.
"""

from __future__ import annotations

from .adversarial import AdversarialConfig
from .state import GraphState, new_state

__all__ = ["AdversarialConfig", "GraphState", "new_state"]
