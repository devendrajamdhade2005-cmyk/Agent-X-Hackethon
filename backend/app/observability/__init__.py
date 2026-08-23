"""Advanced tracing & observability (Task 7).

The point of this layer is not to collect traces — it is to close a loop:

    TRACE → CONTROLLED FAILURE → ROOT CAUSE → IMPROVEMENT → RE-RUN → MEASURE → VERIFY

Everything is instrumentation over the existing runtime. LangGraph, the specialist
agents, the tool registry, the source resilience layer and the Task 6 evaluation
engine are all reused; this package watches them, diagnoses what went wrong, applies
a *bounded* configuration change from a controlled policy registry, re-runs the same
scenario and lets Task 6 decide whether quality held.

Design constraints that matter:
  * No third-party lock-in. `LocalTraceProvider` always works; an external exporter
    is optional and its failure never touches execution.
  * No source-code rewriting. Improvements are versioned, reversible runtime policy.
  * No secrets, no chain-of-thought. Redaction runs before anything is persisted.
"""

from __future__ import annotations

from .schemas import (
    ERROR_CATEGORIES,
    ROOT_CAUSES,
    SPAN_KINDS,
    ErrorRecord,
    Span,
    Trace,
)
from .tracer import Tracer, current_trace, get_tracer

__all__ = [
    "ERROR_CATEGORIES",
    "ROOT_CAUSES",
    "SPAN_KINDS",
    "ErrorRecord",
    "Span",
    "Trace",
    "Tracer",
    "current_trace",
    "get_tracer",
]
