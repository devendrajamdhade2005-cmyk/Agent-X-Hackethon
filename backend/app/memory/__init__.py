"""Context and memory management.

Three layers, deliberately separated:

  * `TaskContext`   — the structured reading of *this* goal (the UNDERSTAND stage).
  * `WorkingMemory` — short-term memory for one run: goal, plan state, the findings
                      that mattered, decisions, gaps. Updated after every step.
  * `LongTermStore` — what survives the run, selected on importance and retrieved
                      by relevance when a later run looks related.

`ContextBuilder` sits between working memory and the specialists: it decides what
a *particular* agent needs to know for a *particular* objective, instead of handing
every agent the whole history. `MemoryManager` owns the lifecycle and is the only
thing the orchestrator talks to.
"""

from __future__ import annotations

from .context_builder import AgentContextPacket, ContextBuilder
from .long_term import (
    IMPORTANCE_ORDER,
    LongTermMemoryItem,
    LongTermStore,
    long_term_store,
)
from .manager import MemoryManager
from .task_context import TaskContext, TaskContextExtractor
from .working import MemoryFact, PlanStepState, WorkingMemory

__all__ = [
    "AgentContextPacket",
    "ContextBuilder",
    "IMPORTANCE_ORDER",
    "LongTermMemoryItem",
    "LongTermStore",
    "MemoryFact",
    "MemoryManager",
    "PlanStepState",
    "TaskContext",
    "TaskContextExtractor",
    "WorkingMemory",
    "long_term_store",
]
