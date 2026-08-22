"""Baselines — what InsightPulse is compared against.

Two baselines, both producing the *same result shape* as a real run so the identical
evaluators can score them. Where a metric genuinely cannot apply to a baseline (a
single LLM call has no tools to recover from), it is reported unavailable with a
reason rather than given a flattering or punitive number.

  Baseline A — single-pass LLM: one model call, no tools, no evidence retrieval.
               This is the honest "just ask the model" comparison. Falls back to a
               deterministic no-evidence stub when no credential is configured, so
               the benchmark still runs offline.

  Baseline B — fixed pipeline: the project's own pre-LangGraph classic agent
               (`run_agent`), which uses real tools in a fixed order with no
               dynamic replanning, verification or conflict resolution.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..agents.agent import run_agent
from ..agents.llm import LLMClient
from ..agents.sanitize import sanitize

BASELINE_LLM = "baseline_llm"
BASELINE_PIPELINE = "baseline_pipeline"

# Metrics that cannot be measured for the single-pass baseline, with the reason
# surfaced in the comparison table (section 28).
LLM_BASELINE_UNAVAILABLE = {
    "recovery_rate": "single-pass baseline performs no tool calls, so no failure can be injected or recovered",
    "resource_efficiency": "single-pass baseline makes no tool calls, so tool-based efficiency is not comparable",
    "efficiency": "single-pass baseline makes no tool calls, so yield per call is undefined",
    "evidence_quality": "single-pass baseline retrieves no evidence, so there are no sources to score",
}


async def run_baseline_llm(
    goal: str,
    *,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """One model call, no tools. Returns the standard result shape.

    Findings are deliberately empty: the point of this baseline is that it answers
    without retrieving evidence, which is exactly what the groundedness and
    hallucination metrics should expose.
    """
    started = time.perf_counter()
    client = llm if llm is not None else LLMClient()
    run_id = f"bl-{uuid.uuid4().hex[:10]}"
    insights: list[dict[str, Any]] = []
    summary = ""
    llm_calls = 0
    author = "unconfigured"
    # Why the baseline produced nothing, when it produces nothing. Recorded so the
    # comparison table can say *which* external condition blocked it instead of
    # implying the baseline simply scored zero.
    blocked_reason = "" if client.available else "no LLM credential is configured"

    if client.available:
        system = (
            "You are a competitive-intelligence analyst. Answer the user's tracking "
            "goal directly from your own knowledge. You have no search tools and no "
            "retrieved documents. Reply with ONLY a JSON object."
        )
        user = (
            f"GOAL: {sanitize(goal)[0]}\n"
            f"KEYWORDS: {keywords or 'none'}\n"
            f"COMPANIES: {competitors or 'none'}\n\n"
            'Return JSON: {"summary": "two short paragraphs", "insights": ['
            '{"title": "...", "what_happened": "one factual sentence", '
            '"summary": "...", "why_it_matters": "...", '
            '"priority": "HIGH|MEDIUM|LOW", "recommended_action": "..."}]}'
        )
        try:
            data = await client.complete_json(
                purpose="baseline", system=system, user=user, max_tokens=1600, temperature=0.3
            )
        except Exception as exc:  # noqa: BLE001 — a baseline failure must not end the suite
            data = None
            blocked_reason = f"the model call raised {type(exc).__name__}"
        llm_calls = int(getattr(getattr(client, "usage", None), "calls", 0) or 0)
        author = client.reasoner_name
        if not data and not blocked_reason:
            # Surface the provider's own reason (quota, auth, timeout) without
            # leaking credentials — only the provider-side status is reported.
            raw_reason = str(
                getattr(client, "disabled_reason", "") or getattr(client, "last_error", "")
            )
            blocked_reason = _safe_reason(raw_reason) or "the model returned no usable JSON"
        if data:
            summary = str(data.get("summary") or "")[:1600]
            for idx, row in enumerate((data.get("insights") or [])[:12]):
                if not isinstance(row, dict):
                    continue
                insights.append({
                    "id": f"bl_ins_{idx}",
                    # No evidence link exists — this is the substantive difference.
                    "finding_id": "",
                    "title": str(row.get("title") or "")[:200],
                    "what_happened": str(row.get("what_happened") or "")[:400],
                    "summary": str(row.get("summary") or "")[:600],
                    "why_it_matters": str(row.get("why_it_matters") or "")[:600],
                    "priority": str(row.get("priority") or "MEDIUM").upper(),
                    "recommended_action": str(row.get("recommended_action") or "")[:400],
                    "source": "model knowledge",
                    "source_url": "",
                    "provider": "llm",
                    "published_date": None,
                    "competitor": "",
                    "signals": [],
                    "confidence": "unverified",
                    "score": 0.0,
                    "simulated": False,
                    "author": author,
                })

    if not summary:
        summary = (
            "The single-pass baseline produced no evidence-backed answer for this goal. "
            "It has no retrieval tools, so nothing can be cited."
            + (f" It could not answer at all: {blocked_reason}." if blocked_reason else "")
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "completed" if insights else "completed_partial",
        "run_id": run_id,
        "goal": goal,
        "system": BASELINE_LLM,
        "findings": [],
        "insights": insights,
        "summary": summary,
        "activity_log": [],
        "execution_plan": [],
        "agents": [],
        "collaboration_events": [],
        "memory": {},
        "framework": {
            "runtime": "single_pass_llm",
            "graph_steps": 1,
            "plan_version": 0,
            "replan_count": 0,
            "verify_count": 0,
            "selected_agents": [],
            "completed_agents": [],
            "tool_executions": [],
            "tool_errors": [],
            "fallback_history": [],
            "conflicting_evidence": [],
            "uncertainty_flags": [],
            "verification_status": "not_started",
            "verification_findings": [],
            "hypotheses": [],
            "evaluation": {},
            "overall_confidence": 0.0,
            "checkpoints": [],
            "injected_events": [],
            "resource": {
                "tool_calls": 0, "max_tool_calls": 0, "llm_calls": llm_calls,
                "elapsed_ms": duration_ms, "estimated_cost": round(llm_calls * 0.010, 4),
            },
            "termination_reason": "single model call completed",
        },
        "metrics": {
            "runtime": "single_pass_llm",
            "duration_ms": duration_ms,
            "tool_calls": 0,
            "llm_calls": llm_calls,
            "findings_total": 0,
            "findings_relevant": 0,
            "insights": len(insights),
            "parallel_agents": 0,
            "replans": 0,
            "verifications": 0,
            "estimated_cost": round(llm_calls * 0.010, 4),
            "overall_confidence": 0.0,
        },
        "baseline_blocked": bool(blocked_reason and not insights),
        "baseline_blocked_reason": blocked_reason,
        "baseline_notes": (
            "Single LLM call with no retrieval. Reported without evidence links, which "
            "is the property the groundedness metric measures."
            if insights else
            f"The single-pass baseline produced no content ({blocked_reason or 'unknown reason'}). "
            f"Metrics that require output are reported unavailable rather than scored as zero."
        ),
    }


def _safe_reason(raw: str) -> str:
    """Condense a provider error into a short, secret-free explanation."""
    low = (raw or "").lower()
    if "429" in low or "quota" in low or "rate limit" in low:
        return "the model provider returned a quota/rate-limit error (HTTP 429)"
    if "401" in low or "403" in low or "api key" in low or "unauthor" in low:
        return "the model provider rejected the credential"
    if "timeout" in low or "exceeded" in low and "s" in low:
        return "the model call timed out"
    if not raw:
        return ""
    # Never echo the provider payload verbatim — it can contain URLs/identifiers.
    return "the model provider returned an error"


async def run_baseline_pipeline(
    goal: str,
    *,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    simulation_mode: bool = True,
) -> dict[str, Any]:
    """The project's classic fixed-order agent — real tools, no dynamic graph.

    This is a fair like-for-like comparison on evidence quality and completion,
    because it uses the same tools and the same simulation fixtures. What it lacks
    is LangGraph's dynamic replanning, verification and conflict resolution.
    """
    result = await run_agent(
        goal,
        keywords=keywords or [],
        competitors=competitors or [],
        simulation_mode=simulation_mode,
    )
    payload = result.to_dict()
    payload["system"] = BASELINE_PIPELINE

    mx = payload.get("metrics") or {}
    # Project the classic result into the framework shape the evaluators read, using
    # only values the classic run genuinely produced.
    payload["framework"] = {
        "runtime": "fixed_pipeline",
        "graph_steps": int(mx.get("iterations") or 0),
        "plan_version": 1,
        "replan_count": 0,
        "verify_count": 0,
        "selected_agents": mx.get("agents_selected") or [],
        "completed_agents": mx.get("agents_used") or [],
        "tool_executions": [
            {
                "tool_name": c.get("tool"),
                "agent": "",
                "query": (c.get("tool_input") or {}).get("query", ""),
                "status": "ok" if c.get("ok") else "failed",
                "source": (c.get("providers_used") or [""])[0],
                "providers_used": c.get("providers_used") or [],
                "providers_failed": [p.get("provider") for p in (c.get("providers_failed") or [])],
                "result_count": int(c.get("items_returned") or 0),
                "latency_ms": int(c.get("latency_ms") or 0),
                "error": c.get("error") or "",
                "attempt": 1,
                "fallback_used": False,
                "simulated": bool(c.get("simulated")),
            }
            for c in ((payload.get("state") or {}).get("tool_calls") or [])
        ],
        "tool_errors": [],
        "fallback_history": [],
        "conflicting_evidence": [],
        "uncertainty_flags": [],
        "verification_status": "not_started",
        "verification_findings": [],
        "hypotheses": [],
        "evaluation": {},
        "overall_confidence": 0.0,
        "checkpoints": [],
        "injected_events": [],
        "resource": {
            "tool_calls": int(mx.get("tool_calls") or 0),
            "max_tool_calls": int(mx.get("max_iterations") or 10),
            "llm_calls": int((mx.get("llm") or {}).get("calls") or 0),
            "elapsed_ms": int(mx.get("duration_ms") or 0),
            "estimated_cost": round(int(mx.get("tool_calls") or 0) * 0.002, 4),
        },
        "termination_reason": (payload.get("state") or {}).get("stop_reason", ""),
    }
    payload["metrics"] = {
        **mx,
        "runtime": "fixed_pipeline",
        "parallel_agents": len(mx.get("agents_used") or []),
        "replans": 0,
        "verifications": 0,
        "estimated_cost": round(int(mx.get("tool_calls") or 0) * 0.002, 4),
        "llm_calls": int((mx.get("llm") or {}).get("calls") or 0),
    }
    payload["baseline_notes"] = (
        "Fixed-order pipeline using the same tools and fixtures as InsightPulse, but "
        "with no dynamic replanning, verification or conflict resolution."
    )
    return payload


def unavailable_reason(system: str, metric_name: str) -> str:
    """Why a metric is not comparable for this baseline, if it isn't."""
    if system == BASELINE_LLM:
        return LLM_BASELINE_UNAVAILABLE.get(metric_name, "")
    if system == BASELINE_PIPELINE and metric_name == "recovery_rate":
        return "fixed pipeline has no fault-injection path, so recovery is not exercised"
    return ""
