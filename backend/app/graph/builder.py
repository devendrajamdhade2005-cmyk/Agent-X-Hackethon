"""Graph assembly — where the StateGraph is wired and compiled.

The topology encodes the dynamic behaviour:

    understand → plan → decompose → resource_check → dispatch
                                                       │ (conditional fan-out)
                        ┌──────────────────────────────┼───────────────┐
                        ▼                               ▼               ▼
                 research_agent                 competitive_agent   (observer)
                        └───────────────┬───────────────┘
                                        ▼
                                    observer → conflict_resolution → self_evaluator
                                                                          │ (conditional)
                                      ┌───────────────────────────────────┼──────────┐
                                      ▼                                    ▼          ▼
                                   verify                               replan     finalize
                                      │                                    │          │
                             conflict_resolution                    (conditional)     ▼
                                      ▲                             dispatch │      memory_update → END
                                      └──────── self_evaluator ◀────────────┘

`dispatch` fans out (a list return from the router) to whichever agents have
pending tasks; `self_evaluator` routes to verify / replan / finalize from the
observed state; `verify` loops back through conflict resolution; `replan` loops
back through the router. Every loop is bounded by the resource governor, the
progress monitor and the compiled `recursion_limit`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState

# Hard step ceiling in addition to the governor and progress monitor (section 24).
RECURSION_LIMIT = 60

# Node name → node function. Registering from a table lets the observability layer
# wrap every node once, instead of each node having to know it is being traced.
NODE_FUNCTIONS: dict[str, Any] = {
    "understand": nodes.understand_node,
    "plan": nodes.plan_node,
    "decompose": nodes.decompose_node,
    "resource_check": nodes.resource_check_node,
    "dispatch": nodes.dispatch_node,
    "research_agent": nodes.research_agent_node,
    "competitive_agent": nodes.competitive_agent_node,
    "observer": nodes.observer_node,
    "conflict_resolution": nodes.conflict_resolution_node,
    "self_evaluator": nodes.self_evaluator_node,
    "verify": nodes.verify_node,
    "replan": nodes.replan_node,
    "finalize": nodes.finalize_node,
    "memory_update": nodes.memory_update_node,
}


def _register(g: StateGraph) -> None:
    """Add every node, traced when observability is importable.

    Tracing is a wrapper, not a rewrite: if the observability package is missing or
    raises at import, the graph is built with the bare node functions and the run
    proceeds exactly as before.
    """
    try:
        from ..observability.instrument import traced_node
    except Exception:  # noqa: BLE001 — never let instrumentation block the graph
        traced_node = None  # type: ignore[assignment]

    for name, fn in NODE_FUNCTIONS.items():
        g.add_node(name, traced_node(name, fn) if traced_node else fn)


def build_graph(checkpointer: Any | None = None, *, interrupt_before: list[str] | None = None):
    """Construct and compile the InsightPulse agent graph.

    `interrupt_before` pauses the graph before the named nodes (used to demonstrate
    and test checkpoint interruption + resume)."""
    g = StateGraph(GraphState)

    _register(g)

    g.add_edge(START, "understand")
    g.add_edge("understand", "plan")
    g.add_edge("plan", "decompose")
    g.add_edge("decompose", "resource_check")
    g.add_edge("resource_check", "dispatch")

    # Dynamic fan-out to the specialists (parallel) or straight to the observer.
    g.add_conditional_edges(
        "dispatch",
        nodes.route_after_dispatch,
        ["research_agent", "competitive_agent", "observer"],
    )
    g.add_edge("research_agent", "observer")
    g.add_edge("competitive_agent", "observer")

    g.add_edge("observer", "conflict_resolution")
    g.add_edge("conflict_resolution", "self_evaluator")

    # The dynamic decision: verify, replan, or finalize.
    g.add_conditional_edges(
        "self_evaluator",
        nodes.route_after_eval,
        {"verify": "verify", "replan": "replan", "finalize": "finalize"},
    )
    # Verify loops back through conflict resolution (re-check with new evidence).
    g.add_edge("verify", "conflict_resolution")
    # Replan loops back through the router, or gives up and finalises.
    g.add_conditional_edges(
        "replan",
        nodes.route_after_replan,
        {"dispatch": "dispatch", "finalize": "finalize"},
    )

    g.add_edge("finalize", "memory_update")
    g.add_edge("memory_update", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before or [])
