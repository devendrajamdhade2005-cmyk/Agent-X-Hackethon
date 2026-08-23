"""Trace analyzer and root-cause diagnosis.

Two stages, deliberately separated:

  * `TraceAnalyzer` reduces a trace to *observations* — measured facts about latency,
    retries, failures, duplicate work and token usage. No opinions.
  * `RootCauseAnalyzer` turns observations into a ranked diagnosis with a confidence
    score derived from how much evidence supports it, plus a quantified impact and a
    recommended improvement type drawn from the controlled policy registry.

The confidence rule matters: it is computed from the strength and consistency of the
evidence, never asserted. When two causes explain the trace about equally well the
result is `MULTIPLE_POSSIBLE_CAUSES`; when nothing explains it, `UNKNOWN`. A
high-confidence diagnosis from thin evidence would be worse than no diagnosis,
because the improvement engine acts on it.
"""

from __future__ import annotations

from typing import Any

from .schemas import Diagnosis, new_id

# Retry attempts beyond this on one provider are considered excessive when the
# provider kept failing — the attempts bought nothing.
WASTEFUL_ATTEMPT_FLOOR = 2

# A provider span slower than this share of the whole run is a latency hotspot.
HOTSPOT_SHARE = 0.25


class TraceAnalyzer:
    """Reduce a trace to measured observations."""

    def analyze(self, trace: dict[str, Any]) -> dict[str, Any]:
        spans = trace.get("spans") or []
        errors = trace.get("errors") or []
        duration = int(trace.get("duration_ms") or 0)

        provider_spans = [s for s in spans if s.get("kind") == "provider"]
        tool_spans = [s for s in spans if s.get("kind") == "tool"]
        llm_spans = [s for s in spans if s.get("kind") == "llm"]
        agent_spans = [s for s in spans if s.get("kind") == "agent"]

        # ── per-provider behaviour ──
        # `failed_calls` keeps each failed call separate. A retry only exists
        # *within* one provider call, so attempts must never be summed across calls
        # when reasoning about wasted retries — three separate one-attempt calls are
        # not "two wasted retries".
        providers: dict[str, dict[str, Any]] = {}
        failed_calls: dict[str, list[dict[str, int]]] = {}
        for span in provider_spans:
            attrs = span.get("attributes") or {}
            name = str(attrs.get("provider") or span.get("name") or "unknown")
            entry = providers.setdefault(name, {
                "provider": name, "calls": 0, "attempts": 0, "failures": 0,
                "latency_ms": 0, "retry_wait_ms": 0, "results": 0,
                "retry_events": 0, "simulated": False,
            })
            attempts = int(attrs.get("attempts") or 0)
            latency = int(attrs.get("latency_ms") or span.get("duration_ms") or 0)
            retry_wait = int(attrs.get("retry_wait_ms") or 0)
            entry["calls"] += 1
            entry["attempts"] += attempts
            entry["latency_ms"] += latency
            entry["retry_wait_ms"] += retry_wait
            entry["results"] += int(attrs.get("result_count") or 0)
            entry["retry_events"] += sum(
                1 for e in (span.get("events") or []) if e.get("name") == "retry_recorded"
            )
            if span.get("status") == "error":
                entry["failures"] += 1
                failed_calls.setdefault(name, []).append({
                    "attempts": attempts,
                    "latency_ms": latency,
                    "retry_wait_ms": retry_wait,
                })
            if attrs.get("simulated"):
                entry["simulated"] = True

        # ── errors grouped by category and provider ──
        by_category: dict[str, int] = {}
        by_provider: dict[str, list[dict[str, Any]]] = {}
        injected = 0
        for err in errors:
            cat = str(err.get("error_type") or "UNKNOWN")
            by_category[cat] = by_category.get(cat, 0) + 1
            prov = str(err.get("provider") or err.get("tool") or "unknown")
            by_provider.setdefault(prov, []).append(err)
            if err.get("injected"):
                injected += 1

        # ── duplicate tool work ──
        seen: dict[tuple[str, str], int] = {}
        for span in tool_spans:
            attrs = span.get("attributes") or {}
            key = (str(attrs.get("tool") or span.get("name")), str(attrs.get("query") or ""))
            seen[key] = seen.get(key, 0) + 1
        duplicates = [
            {"tool": k[0], "query": k[1], "count": n} for k, n in seen.items() if n > 1
        ]

        # ── latency hotspots ──
        hotspots = []
        for name, entry in providers.items():
            share = (entry["latency_ms"] / duration) if duration else 0.0
            if entry["latency_ms"] > 0 and share >= HOTSPOT_SHARE:
                hotspots.append({
                    "component": name, "latency_ms": entry["latency_ms"],
                    "share_of_run": round(share, 3),
                })
        hotspots.sort(key=lambda h: -h["latency_ms"])

        # ── wasted retries ──
        # Counted per failed call: a call that made N attempts and still failed
        # wasted N-1 of them. Attempts made by calls that succeeded are not waste,
        # and attempts from different calls are not retries of each other.
        wasted = []
        for name, calls in failed_calls.items():
            wasted_attempts = sum(max(0, c["attempts"] - 1) for c in calls)
            if wasted_attempts <= 0:
                continue
            wasted.append({
                "provider": name,
                "failed_calls": len(calls),
                "attempts": sum(c["attempts"] for c in calls),
                "wasted_attempts": wasted_attempts,
                # Only the latency of the calls that failed, so the figure quoted as
                # "spent and recovered nothing" is exactly that.
                "latency_ms": sum(c["latency_ms"] for c in calls),
                # The measured sleep between those attempts — the portion a retry
                # policy change removes directly.
                "retry_wait_ms": sum(c["retry_wait_ms"] for c in calls),
            })
        wasted.sort(key=lambda w: -w["wasted_attempts"])

        tokens = trace.get("token_usage") or {}
        return {
            "trace_id": trace.get("trace_id"),
            "run_id": trace.get("run_id"),
            "scenario": trace.get("scenario"),
            "duration_ms": duration,
            "span_count": len(spans),
            "counts": {
                "agents": len(agent_spans),
                "tool_calls": len(tool_spans),
                "provider_calls": len(provider_spans),
                "llm_calls": len(llm_spans),
                "errors": len(errors),
                "injected_errors": injected,
                "retry_events": sum(p["retry_events"] for p in providers.values()),
                "fallbacks": sum(
                    1 for s in spans if s.get("kind") == "fallback"
                ) or int((trace.get("metrics") or {}).get("fallback_count") or 0),
            },
            "providers": list(providers.values()),
            "errors_by_category": by_category,
            "errors_by_provider": {k: len(v) for k, v in by_provider.items()},
            "duplicate_tool_calls": duplicates,
            "latency_hotspots": hotspots,
            "wasted_retries": wasted,
            "token_usage": tokens,
            "token_status": tokens.get("status", "unavailable"),
            "recovered_errors": sum(
                1 for e in errors if e.get("recovery_status") == "recovered"
            ),
            "unrecovered_errors": sum(
                1 for e in errors if e.get("recovery_status") != "recovered"
            ),
            "final_status": trace.get("status"),
            "optimization_version": trace.get("optimization_version", 0),
        }


class RootCauseAnalyzer:
    """Turn observations into a diagnosis with evidence-weighted confidence."""

    def diagnose(
        self, trace: dict[str, Any], analysis: dict[str, Any] | None = None
    ) -> Diagnosis:
        analysis = analysis or TraceAnalyzer().analyze(trace)
        candidates = self._candidates(trace, analysis)

        if not candidates:
            return Diagnosis(
                diagnosis_id=new_id("dx-"),
                trace_id=str(trace.get("trace_id") or ""),
                root_cause_type="UNKNOWN",
                affected_component="",
                confidence=0.0,
                evidence=["no anomaly was detected in this trace"],
                impact={},
                recommended_improvement="No change recommended — the trace shows no "
                                        "measurable inefficiency or failure pattern.",
                improvement_type="",
                uncertain=True,
            )

        candidates.sort(key=lambda c: -c["confidence"])
        best = candidates[0]

        # Two rival explanations of comparable strength means we should not pretend
        # to have picked one. The improvement engine treats this as "needs review".
        if len(candidates) > 1 and (best["confidence"] - candidates[1]["confidence"]) < 0.15:
            return Diagnosis(
                diagnosis_id=new_id("dx-"),
                trace_id=str(trace.get("trace_id") or ""),
                root_cause_type="MULTIPLE_POSSIBLE_CAUSES",
                affected_component=best["component"],
                confidence=round(min(0.6, best["confidence"]), 3),
                evidence=best["evidence"] + [
                    f"a second explanation ({candidates[1]['root_cause']}) fits the "
                    f"evidence almost equally well"
                ],
                impact=best["impact"],
                recommended_improvement=(
                    f"Evidence supports more than one cause; the strongest is "
                    f"{best['root_cause']} on {best['component']}. Review before applying."
                ),
                improvement_type=best["improvement_type"],
                uncertain=True,
                alternatives=[c["root_cause"] for c in candidates[1:3]],
            )

        return Diagnosis(
            diagnosis_id=new_id("dx-"),
            trace_id=str(trace.get("trace_id") or ""),
            root_cause_type=best["root_cause"],
            affected_component=best["component"],
            confidence=round(best["confidence"], 3),
            evidence=best["evidence"],
            impact=best["impact"],
            recommended_improvement=best["recommendation"],
            improvement_type=best["improvement_type"],
            uncertain=best["confidence"] < 0.6,
            alternatives=[c["root_cause"] for c in candidates[1:3]],
        )

    # ─────────────────────────────────────────────────────────
    def _candidates(
        self, trace: dict[str, Any], analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        duration = max(1, int(analysis.get("duration_ms") or 0))
        by_cat = analysis.get("errors_by_category") or {}
        errors = trace.get("errors") or []

        # ── RATE_LIMIT + EXCESSIVE_RETRY ──
        rate_limited = [e for e in errors if e.get("error_type") == "RATE_LIMIT"]
        if rate_limited:
            provider = str(rate_limited[0].get("provider") or "unknown")
            statuses = sorted({e.get("http_status") for e in rate_limited if e.get("http_status")})
            waste = next(
                (w for w in analysis.get("wasted_retries") or [] if w["provider"] == provider),
                None,
            )
            evidence = [
                f"{len(rate_limited)} rate-limit response(s)"
                + (f" (HTTP {', '.join(str(s) for s in statuses)})" if statuses else "")
                + f" from {provider}",
            ]
            # Confidence grows with the number of consistent signals.
            confidence = 0.55
            if len(rate_limited) >= 2:
                confidence += 0.20
                evidence.append(
                    f"the provider failed on {len(rate_limited)} consecutive attempts, so "
                    f"the retries did not recover it"
                )
            if waste:
                confidence += 0.15
                evidence.append(
                    f"{waste['wasted_attempts']} retry attempt(s) beyond the first were "
                    f"spent across {waste['failed_calls']} failed call(s) and still did "
                    f"not recover the provider"
                )
                if waste.get("retry_wait_ms"):
                    evidence.append(
                        f"{waste['retry_wait_ms']}ms of that was backoff waiting between "
                        f"attempts, measured from the retry loop"
                    )
            hotspot = next(
                (h for h in analysis.get("latency_hotspots") or []
                 if h["component"] == provider), None,
            )
            if hotspot:
                confidence += 0.08
                evidence.append(
                    f"{provider} accounted for {hotspot['share_of_run']:.0%} of total run time"
                )
            if all(e.get("retryable") for e in rate_limited):
                evidence.append("every failure was classified retryable, so the retry "
                                "policy governed the behaviour")

            # Attribute only the measured backoff wait as latency the retry added.
            # The first attempt would have happened regardless, so its duration is
            # not overhead; claiming the whole failed-call latency would overstate it.
            wasted_ms = int(waste.get("retry_wait_ms") or 0) if waste else 0
            wasted_attempts = waste["wasted_attempts"] if waste else 0
            # Excessive retry is the actionable framing when retries were spent and
            # bought nothing; otherwise the cause is the rate limit itself.
            root = "EXCESSIVE_RETRY" if wasted_attempts >= 1 else "RATE_LIMIT"
            out.append({
                "root_cause": root,
                "component": provider,
                "confidence": min(0.97, confidence),
                "evidence": evidence,
                "impact": self._impact(
                    analysis,
                    latency_added=wasted_ms,
                    extra_calls=wasted_attempts,
                    retries=wasted_attempts,
                    errors=len(rate_limited),
                ),
                "recommendation": (
                    f"Reduce the retry ceiling for {provider} to 1 attempt and fall back "
                    f"to the remaining providers immediately. A provider returning "
                    f"HTTP 429 is unlikely to succeed on an immediate retry, so the "
                    f"additional attempts add latency without adding evidence."
                ),
                "improvement_type": "RETRY_POLICY",
            })

        # ── TIMEOUT ──
        timeouts = [e for e in errors if e.get("error_type") == "TIMEOUT"]
        if timeouts:
            provider = str(timeouts[0].get("provider") or "unknown")
            confidence = 0.55 + min(0.25, 0.1 * len(timeouts))
            out.append({
                "root_cause": "TIMEOUT",
                "component": provider,
                "confidence": confidence,
                "evidence": [
                    f"{len(timeouts)} timeout(s) recorded against {provider}",
                    "timeouts consume the full deadline before failing, so they cost "
                    "more latency than an immediate error",
                ],
                "impact": self._impact(
                    analysis, latency_added=sum(
                        int(p.get("latency_ms") or 0) for p in analysis.get("providers") or []
                        if p.get("provider") == provider
                    ), extra_calls=len(timeouts) - 1, retries=len(timeouts) - 1,
                    errors=len(timeouts),
                ),
                "recommendation": (
                    f"Lower the timeout for {provider} so a stalled request fails fast "
                    f"and the remaining providers are reached sooner."
                ),
                "improvement_type": "TIMEOUT",
            })

        # ── REDUNDANT_TOOL_CALL ──
        duplicates = analysis.get("duplicate_tool_calls") or []
        if duplicates:
            worst = max(duplicates, key=lambda d: d["count"])
            extra = sum(d["count"] - 1 for d in duplicates)
            out.append({
                "root_cause": "REDUNDANT_TOOL_CALL",
                "component": str(worst.get("tool") or "tool"),
                "confidence": min(0.9, 0.6 + 0.1 * extra),
                "evidence": [
                    f"{worst['tool']} was called {worst['count']} times with an identical "
                    f"query",
                    f"{extra} repeat call(s) across the run returned no new query shape",
                ],
                "impact": self._impact(analysis, extra_calls=extra, retries=0, errors=0),
                "recommendation": (
                    "Suppress a repeat provider call for an identical query within one "
                    "run, so the second call is served from the first result."
                ),
                "improvement_type": "CACHE_DEDUP",
            })

        # ── RESOURCE_LIMIT ──
        metrics = trace.get("metrics") or {}
        if metrics.get("budget_exhausted") or by_cat.get("RESOURCE_LIMIT"):
            out.append({
                "root_cause": "RESOURCE_LIMIT",
                "component": "resource_governor",
                "confidence": 0.7,
                "evidence": ["the run hit a configured resource ceiling"],
                "impact": self._impact(analysis),
                "recommendation": "Raise the tool-call ceiling or reduce low-value work "
                                  "so the highest-value evidence is still gathered.",
                "improvement_type": "RESOURCE_POLICY",
            })

        # ── MODEL_ERROR ──
        model_errors = [e for e in errors if e.get("error_type") == "MODEL_ERROR"]
        if model_errors:
            out.append({
                "root_cause": "MODEL_ERROR",
                "component": "llm",
                "confidence": 0.6 + min(0.2, 0.1 * len(model_errors)),
                "evidence": [
                    f"{len(model_errors)} model error(s); the run continued on the "
                    f"deterministic reasoner"
                ],
                "impact": self._impact(analysis, errors=len(model_errors)),
                "recommendation": "No automated policy change is safe here — the model "
                                  "provider itself rejected the request.",
                "improvement_type": "",
            })

        return out

    # ── impact (section 21) ─────────────────────────────────
    def _impact(
        self,
        analysis: dict[str, Any],
        *,
        latency_added: int = 0,
        extra_calls: int = 0,
        retries: int = 0,
        errors: int = 0,
    ) -> dict[str, Any]:
        duration = max(1, int(analysis.get("duration_ms") or 0))
        counts = analysis.get("counts") or {}
        tokens = analysis.get("token_usage") or {}
        total_calls = max(1, int(counts.get("provider_calls") or 0))

        # Token overhead is only reportable when token usage was actually measured.
        if tokens.get("status") == "measured" and tokens.get("total_tokens"):
            token_overhead: Any = 0
            token_note = "no additional model calls were attributable to this cause"
        else:
            token_overhead = None
            token_note = (
                "token usage was not reported by the provider for this run, so token "
                "overhead is not measurable"
            )

        return {
            "latency_added_ms": int(latency_added),
            "latency_share_of_run": round(latency_added / duration, 3) if latency_added else 0.0,
            "tool_calls_added": int(extra_calls),
            "retry_calls": int(retries),
            "error_count": int(errors or counts.get("errors") or 0),
            "fallback_count": int(counts.get("fallbacks") or 0),
            "provider_call_share": round(extra_calls / total_calls, 3) if extra_calls else 0.0,
            "token_overhead": token_overhead,
            "token_overhead_note": token_note,
            "estimated_cost_change_usd": None,
            "estimated_cost_note": (
                "cost is derived from token usage, which this provider did not report"
                if token_overhead is None else "no measurable cost change"
            ),
            "task_success_impact": (
                "none — the run still completed"
                if analysis.get("final_status") in {"ok", None}
                else "the run did not complete cleanly"
            ),
        }
