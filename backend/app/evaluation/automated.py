"""Automated evaluators.

Twelve evaluators, one concern each. All deterministic: given the same captured run
they produce the same scores, which is what makes the benchmark repeatable and the
numbers defensible.

No model-based judge is used. Every judgement here is a checkable computation over
the run's own evidence, so a score can always be traced to the data that produced it
rather than to an opinion. That is a deliberate choice: an "LLM says 8/10" signal
would be non-reproducible, would need its own validation layer, and would make the
benchmark depend on provider availability — which this project has already seen fail
(the configured model's quota can be exhausted). Semantic judgement is instead
supplied by the human review layer in `human.py`, where it is attributed to a
reviewer and its disagreement with the automated score is reported rather than hidden.

The central mechanic is claim extraction. The agent's final output is a list of
`Insight` objects, and each one carries a `finding_id` pointing at the evidence it
was written from. That link is what makes groundedness measurable rather than
guessed: a factual claim is grounded when its evidence link resolves to a finding
the run actually collected *and* the claim's content overlaps that finding's text.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from ..sources.credibility import domain_of
from . import metrics as M
from .schemas import (
    CLAIM_INFERRED,
    CLAIM_PARTIAL,
    CLAIM_SUPPORTED,
    CLAIM_UNCERTAIN,
    CLAIM_UNSUPPORTED,
    KIND_FACTUAL,
    KIND_HYPOTHESIS,
    KIND_INTERPRETIVE,
    KIND_RECOMMENDATION,
    ClaimRecord,
    EvaluationCase,
    MetricResult,
)

RELEVANCE_THRESHOLD = 0.35
CREDIBILITY_SCORE = {"high": 1.0, "standard": 0.7, "low": 0.3, "unverified": 0.1}

_WORD = re.compile(r"[a-z0-9$%\-+.]+")
_STOP = {
    "the", "a", "an", "and", "or", "for", "with", "from", "that", "this", "to", "in",
    "on", "of", "at", "by", "as", "is", "are", "was", "were", "has", "have", "had",
    "it", "its", "their", "they", "be", "been", "will", "would", "can", "could",
    "not", "no", "but", "than", "then", "so", "such", "into", "over", "about",
    "one", "two", "new", "more", "most", "also", "which", "who", "what", "when",
    "source", "report", "reports", "reported", "paper", "article", "news",
}

# Language that marks a statement as hedged / uncertain rather than asserted.
_HEDGES = (
    "may ", "might ", "could ", "appears", "appear to", "suggests", "suggest ",
    "indicates", "indicate ", "reportedly", "unconfirmed", "unverified",
    "not clear", "unclear", "possible", "possibly", "potentially", "likely",
    "seems", "seem to", "if confirmed", "remains to be", "no evidence",
    "insufficient evidence", "cannot be confirmed", "treat as", "early indicator",
)

# Language that marks an interpretive/analytical statement (not a factual claim).
_INTERPRETIVE_CUES = (
    "matters", "because", "so this", "implies", "signals that", "worth ",
    "indicates where", "typically precedes", "treat it as", "does not yet change",
)

# Assertive phrasing used when checking whether a refusal case was answered anyway.
#
# These are deliberately *affirming constructions*, never topic vocabulary. A phrase
# like "secretly developed" is part of the subject the user asked about, and it
# legitimately appears in retrieved titles and in the agent's restatement of the
# request — matching on it would flag a correct refusal as an assertion. Only
# language that claims something is established counts.
_ASSERTION_CUES = (
    "confirms that", "confirmed that", "proves that", "proven that",
    "we can confirm", "it is confirmed", "evidence proves", "evidence confirms",
    "established that", "definitively shows", "conclusively shows",
    "verified that openai secretly", "did secretly develop",
)

# Explicit refusal / insufficiency language for the unsupported-conclusion case.
_REFUSAL_CUES = (
    "no evidence", "insufficient evidence", "cannot be confirmed", "cannot confirm",
    "not supported", "unsupported", "no public evidence", "could not be established",
    "does not support", "no confirmation", "unverified", "unconfirmed",
    "nothing cleared", "no findings", "no relevant", "cannot be established",
    "not possible to confirm", "speculative",
)


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2 and t not in _STOP}


def _overlap(a: str, b: str) -> float:
    """Containment of `a`'s content words in `b`. Not text similarity scoring — it is
    only used to ask 'does the cited evidence actually mention this?'."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _hedged(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _HEDGES)


def _age_days(published: str | None) -> int | None:
    if not published:
        return None
    try:
        d = date.fromisoformat(str(published)[:10])
    except (ValueError, TypeError):
        return None
    return max(0, (datetime.now(UTC).date() - d).days)


def _fw(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("framework") or {}


def _mx(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("metrics") or {}


# ═════════════════════════════════════════════════════════════
# CLAIM EXTRACTION  (foundation for groundedness + hallucination)
# ═════════════════════════════════════════════════════════════
class ClaimExtractor:
    """Pull assertions out of the final output and classify each against evidence."""

    # Overlap needed with the cited finding for a claim to count as supported.
    SUPPORT_THRESHOLD = 0.45
    PARTIAL_THRESHOLD = 0.20

    def extract(self, run: dict[str, Any]) -> list[ClaimRecord]:
        findings = {f.get("id"): f for f in (run.get("findings") or [])}
        claims: list[ClaimRecord] = []

        for idx, ins in enumerate(run.get("insights") or []):
            fid = str(ins.get("finding_id") or "")
            evidence = findings.get(fid)

            # `what_happened` and `summary` assert facts about the world.
            for field_name in ("what_happened", "summary"):
                text = str(ins.get(field_name) or "").strip()
                if not text:
                    continue
                claims.append(
                    self._classify(
                        claim_id=f"c{idx}-{field_name}",
                        text=text,
                        kind=KIND_FACTUAL,
                        source_field=f"insight.{field_name}",
                        fid=fid,
                        evidence=evidence,
                    )
                )

            # `why_it_matters` is analysis built on the finding, not a new fact.
            why = str(ins.get("why_it_matters") or "").strip()
            if why:
                claims.append(
                    ClaimRecord(
                        claim_id=f"c{idx}-why_it_matters",
                        text=why,
                        kind=KIND_INTERPRETIVE,
                        source_field="insight.why_it_matters",
                        finding_id=fid,
                        verdict=CLAIM_INFERRED,
                        evidence_overlap=_overlap(why, self._evidence_text(evidence)),
                        hedged=_hedged(why),
                        reason="analytical judgement derived from the cited finding",
                    )
                )

            # Recommendations are actions, not factual assertions (section 15).
            action = str(ins.get("recommended_action") or "").strip()
            if action:
                claims.append(
                    ClaimRecord(
                        claim_id=f"c{idx}-recommended_action",
                        text=action,
                        kind=KIND_RECOMMENDATION,
                        source_field="insight.recommended_action",
                        finding_id=fid,
                        verdict=CLAIM_INFERRED,
                        hedged=False,
                        reason="recommended action — excluded from factual scoring",
                    )
                )

        # Task 5 hypotheses are explicitly labelled as hypotheses (section 15/22).
        for h_idx, hyp in enumerate(_fw(run).get("hypotheses") or []):
            statement = str(hyp.get("statement") or "").strip()
            if not statement:
                continue
            claims.append(
                ClaimRecord(
                    claim_id=f"h{h_idx}",
                    text=statement,
                    kind=KIND_HYPOTHESIS,
                    source_field="framework.hypotheses",
                    verdict=CLAIM_UNCERTAIN,
                    hedged=True,
                    reason=f"labelled hypothesis, status={hyp.get('status', 'PROPOSED')}",
                )
            )
        return claims

    def _evidence_text(self, evidence: dict[str, Any] | None) -> str:
        if not evidence:
            return ""
        return " ".join(
            str(evidence.get(k) or "")
            for k in ("title", "summary", "competitor", "provider")
        ) + " " + " ".join(str(s) for s in (evidence.get("signals") or []))

    def _classify(
        self,
        *,
        claim_id: str,
        text: str,
        kind: str,
        source_field: str,
        fid: str,
        evidence: dict[str, Any] | None,
    ) -> ClaimRecord:
        hedged = _hedged(text)
        rec = ClaimRecord(
            claim_id=claim_id, text=text[:400], kind=kind,
            source_field=source_field, finding_id=fid, hedged=hedged,
        )
        if evidence is None:
            # No resolvable evidence link. A hedged statement is honest uncertainty;
            # an assertive one is unsupported.
            rec.verdict = CLAIM_UNCERTAIN if hedged else CLAIM_UNSUPPORTED
            rec.reason = (
                "no evidence record resolves for this claim's finding_id"
                if fid else "claim carries no evidence link"
            )
            return rec

        rec.evidence_provider = str(evidence.get("provider") or "")
        rec.evidence_credibility = str(evidence.get("credibility") or "")
        overlap = _overlap(text, self._evidence_text(evidence))
        rec.evidence_overlap = round(overlap, 3)

        if any(cue in text.lower() for cue in _INTERPRETIVE_CUES):
            rec.kind = KIND_INTERPRETIVE
            rec.verdict = CLAIM_INFERRED
            rec.reason = "interpretive phrasing; scored as inference, not a fact"
            return rec

        if overlap >= self.SUPPORT_THRESHOLD:
            rec.verdict = CLAIM_SUPPORTED
            rec.reason = f"content matches the cited evidence (overlap {overlap:.2f})"
        elif overlap >= self.PARTIAL_THRESHOLD:
            rec.verdict = CLAIM_PARTIAL
            rec.reason = f"partially matches the cited evidence (overlap {overlap:.2f})"
        elif hedged:
            rec.verdict = CLAIM_UNCERTAIN
            rec.reason = "weak evidence match but the claim is explicitly hedged"
        else:
            rec.verdict = CLAIM_UNSUPPORTED
            rec.reason = f"asserted but the cited evidence does not contain it (overlap {overlap:.2f})"
        return rec


# ═════════════════════════════════════════════════════════════
# 1. CORRECTNESS / ACCURACY
# ═════════════════════════════════════════════════════════════
class CorrectnessEvaluator:
    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.ACCURACY)
        findings = run.get("findings") or []
        insights = run.get("insights") or []

        expected: list[tuple[str, str]] = []
        expected += [("entity", e) for e in case.expected_entities]
        expected += [("source", s) for s in case.expected_sources]
        expected += [("fact", f) for f in case.expected_facts]

        if not expected:
            return spec.unavailable(
                "this case defines no checkable ground truth (accuracy is not scored "
                "for open-ended goals; task completion and groundedness are used instead)"
            )

        blob = " ".join(
            f"{f.get('title','')} {f.get('summary','')} {f.get('competitor','')}"
            for f in findings
        ).lower()
        observed_sources = {str(f.get("source") or "").lower() for f in findings}
        observed_entities = {str(f.get("competitor") or "").lower() for f in findings if f.get("competitor")}
        insight_blob = " ".join(
            f"{i.get('what_happened','')} {i.get('summary','')}" for i in insights
        ).lower()

        hits: list[dict[str, Any]] = []
        misses: list[dict[str, Any]] = []
        for kind, want in expected:
            w = want.lower()
            if kind == "entity":
                ok = w in observed_entities or w in blob
            elif kind == "source":
                ok = w in observed_sources
            else:  # controlled fixture fact
                ok = _overlap(want, blob + " " + insight_blob) >= 0.5
            (hits if ok else misses).append({"kind": kind, "expected": want})

        # Expected-source ground truth is a *set* of acceptable categories for some
        # cases (a competitive goal may legitimately satisfy via web or news), so
        # source recall is credited when any expected source category was reached.
        source_expected = [e for k, e in expected if k == "source"]
        source_hit = any(e.lower() in observed_sources for e in source_expected)

        matched = len(hits)
        if source_expected and source_hit:
            # Credit the source dimension once rather than penalising unreached
            # alternatives within the same acceptable set.
            unmatched_sources = [m for m in misses if m["kind"] == "source"]
            matched += len(unmatched_sources)
            misses = [m for m in misses if m["kind"] != "source"]

        recall = matched / len(expected)
        # Precision proxy: of the reported insights, how many are traceable to
        # evidence the run actually collected. This guards against 'recall by
        # spraying output' without double-counting relevance, which
        # `evidence_quality` already scores on its own scale. A relevance cutoff is
        # deliberately not applied here: the insight generator intentionally reports
        # the best available material when nothing clears the bar, and that is a
        # coverage characteristic rather than an accuracy error.
        finding_ids = {f.get("id") for f in findings}
        on_target = sum(1 for i in insights if i.get("finding_id") in finding_ids)
        precision = (on_target / len(insights)) if insights else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return spec.result(
            round(f1, 4),
            method="F1 of ground-truth recall and evidence-linked insight precision",
            details={
                "expected_total": len(expected),
                "matched": matched,
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "hits": hits,
                "misses": misses,
                "accuracy_method": "structural ground truth (entities/sources/fixture facts)",
                "accuracy_notes": (
                    "Benchmark runs in deterministic simulation mode, so ground truth is "
                    "structural and checkable rather than a live-world fact check."
                ),
            },
        )


# ═════════════════════════════════════════════════════════════
# 2. TASK COMPLETION
# ═════════════════════════════════════════════════════════════
class TaskCompletionEvaluator:
    """Verify each required subtask from execution evidence, never from HTTP status."""

    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.TASK_COMPLETION)
        required = case.expected_subtasks
        if not required:
            return spec.unavailable("case defines no required subtasks")

        fw = _fw(run)
        findings = run.get("findings") or []
        insights = run.get("insights") or []
        tool_execs = fw.get("tool_executions") or []
        summary = str(run.get("summary") or "")
        completed_agents = set(fw.get("completed_agents") or [])

        checks: list[dict[str, Any]] = []
        for subtask in required:
            s = subtask.lower()
            done = False
            evidence = ""

            if "understand" in s:
                done = bool(run.get("goal")) and bool(fw.get("plan_version") or run.get("execution_plan"))
                evidence = "goal captured and plan produced"
            elif "plan" in s and "revise" not in s:
                done = bool(run.get("execution_plan")) or int(fw.get("plan_version") or 0) >= 1
                evidence = f"plan_version={fw.get('plan_version')}"
            elif "research intelligence" in s:
                done = "research_agent" in completed_agents
                evidence = f"completed_agents={sorted(completed_agents)}"
            elif "competitive intelligence" in s:
                done = "competitive_agent" in completed_agents
                evidence = f"completed_agents={sorted(completed_agents)}"
            elif "patent" in s:
                done = (
                    any("patent" in str(t.get("tool_name")) for t in tool_execs)
                    or any(str(f.get("source")) == "patent" for f in findings)
                )
                evidence = "patent tool call or patent-sourced finding present"
            elif "each named competitor" in s or "cover" in s:
                wanted = [c.lower() for c in case.competitors]
                seen = {str(f.get("competitor") or "").lower() for f in findings}
                done = bool(wanted) and all(w in seen for w in wanted)
                evidence = f"covered={sorted(seen & set(wanted))} of {wanted}"
            elif "compare evidence" in s:
                done = bool(fw.get("corroborated_finding_ids")) or len(
                    {str(f.get("source")) for f in findings}
                ) >= 2
                evidence = "cross-source comparison performed"
            elif "recover" in s:
                done = bool(fw.get("fallback_history"))
                evidence = f"fallbacks={len(fw.get('fallback_history') or [])}"
            elif "contradiction" in s or "conflict" in s:
                done = bool(fw.get("conflicting_evidence"))
                evidence = f"conflicts={len(fw.get('conflicting_evidence') or [])}"
            elif "credibility" in s:
                done = any(
                    c.get("verdict") for c in (fw.get("conflicting_evidence") or [])
                )
                evidence = "conflict verdicts recorded"
            elif "verify" in s or "acknowledge uncertainty" in s:
                done = (
                    int(fw.get("verify_count") or 0) > 0
                    or bool(fw.get("uncertainty_flags"))
                    or fw.get("verification_status") in {"resolved", "unresolved"}
                )
                evidence = f"verify_count={fw.get('verify_count')}, status={fw.get('verification_status')}"
            elif "revise the plan" in s:
                done = int(fw.get("replan_count") or 0) > 0
                evidence = f"replans={fw.get('replan_count')}"
            elif "prioritized insights" in s:
                done = len(insights) > 0
                evidence = f"insights={len(insights)}"
            elif "bounded interpretation" in s:
                done = bool(str(run.get("state", {}).get("plan", {}).get("interpretation") or "")) or bool(summary)
                evidence = "interpretation/summary stated"
            elif "missing subject" in s:
                # Detected either by asking, by flagging a gap, or by reporting that
                # nothing could be tracked.
                done = (
                    bool(fw.get("uncertainty_flags"))
                    or any(c in summary.lower() for c in _REFUSAL_CUES)
                    or not case.competitors and not findings
                    or "no companies" in summary.lower()
                    or "widening" in summary.lower()
                )
                evidence = "missing-subject limitation surfaced"
            elif "degrade safely" in s or "clarification" in s:
                done = bool(run.get("summary")) and run.get("status") in {
                    "completed", "completed_partial"
                }
                evidence = f"status={run.get('status')} with a written summary"
            elif "search" in s or "supporting evidence" in s:
                done = len(tool_execs) > 0
                evidence = f"tool_calls={len(tool_execs)}"
            elif "decline" in s or "refus" in s:
                done = _refusal_present(run)
                evidence = "insufficiency stated in the output"
            else:
                done = bool(insights)
                evidence = "generic completion via produced insights"

            checks.append({"subtask": subtask, "completed": bool(done), "evidence": evidence})

        completed = sum(1 for c in checks if c["completed"])
        score = completed / len(required)
        status = "COMPLETED" if score >= 0.999 else ("PARTIAL" if score > 0 else "FAILED")
        return spec.result(
            round(score, 4),
            details={
                "required_subtask_count": len(required),
                "completed_subtask_count": completed,
                "status": status,
                "checks": checks,
            },
        )


def _strip_echoed_goal(text: str, goal: str) -> str:
    """Remove the user's own goal wording (and quoted restatements) from `text`.

    Assertion detection must fire on what the agent *claims*, not on the request it
    is repeating back to the user.
    """
    low = text
    goal_low = (goal or "").strip().lower()
    if goal_low:
        low = low.replace(goal_low, " ")
        # Also drop any quoted span that contains most of the goal's content words,
        # which covers lightly-trimmed restatements.
        goal_tokens = _tokens(goal_low)
        if goal_tokens:
            for quoted in re.findall(r'"([^"]{10,400})"', low):
                if _overlap(quoted, goal_low) >= 0.6 or (
                    len(_tokens(quoted) & goal_tokens) / max(1, len(goal_tokens)) >= 0.6
                ):
                    low = low.replace(f'"{quoted}"', " ")
    return low


def _refusal_present(run: dict[str, Any]) -> bool:
    """True when the output explicitly communicates evidentiary insufficiency."""
    text = " ".join([
        str(run.get("summary") or ""),
        " ".join(str(i.get("what_happened") or "") for i in (run.get("insights") or [])),
        " ".join(str(i.get("summary") or "") for i in (run.get("insights") or [])),
        " ".join(str(i.get("why_it_matters") or "") for i in (run.get("insights") or [])),
        " ".join(str(f) for f in (_fw(run).get("uncertainty_flags") or [])),
        " ".join(str(h.get("status") or "") for h in (_fw(run).get("hypotheses") or [])),
    ]).lower()
    if any(cue in text for cue in _REFUSAL_CUES):
        return True
    # An explicitly unsupported/inconclusive hypothesis is also a refusal signal.
    return any(
        str(h.get("status")) in {"UNSUPPORTED", "INCONCLUSIVE", "PARTIALLY_SUPPORTED"}
        for h in (_fw(run).get("hypotheses") or [])
    )


# ═════════════════════════════════════════════════════════════
# 3. EVIDENCE QUALITY
# ═════════════════════════════════════════════════════════════
class EvidenceEvaluator:
    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.EVIDENCE_QUALITY)
        findings = run.get("findings") or []
        if not findings:
            return spec.unavailable("the run collected no findings to assess")

        relevant = [f for f in findings if float(f.get("relevance") or 0) >= RELEVANCE_THRESHOLD]
        pool = relevant or findings

        credibility = M.mean_of([
            CREDIBILITY_SCORE.get(str(f.get("credibility") or "standard"), 0.5) for f in pool
        ]) or 0.0
        relevance = M.mean_of([float(f.get("relevance") or 0) for f in pool]) or 0.0

        ages = [a for a in (_age_days(f.get("published_date")) for f in pool) if a is not None]
        recency = M.mean_of([max(0.0, 1.0 - (a / 365.0)) for a in ages]) if ages else None

        domains = {domain_of(str(f.get("url") or "")) or str(f.get("provider") or "") for f in pool}
        providers = {str(f.get("provider") or "") for f in pool if f.get("provider")}
        independence = min(1.0, len(domains) / 4.0)

        corroborated = sum(1 for f in pool if f.get("corroborated_by"))
        corroboration = min(1.0, corroborated / max(1, min(len(pool), 4)))

        weak = [f for f in pool if str(f.get("credibility")) in {"low", "unverified"}]
        credible = [f for f in pool if str(f.get("credibility")) in {"high", "standard"}]

        parts = {
            "credibility": (credibility, 0.30),
            "relevance": (relevance, 0.25),
            "recency": (recency if recency is not None else credibility, 0.15),
            "independence": (independence, 0.15),
            "corroboration": (corroboration, 0.15),
        }
        score = sum(v * w for v, w in parts.values())

        # Uncorroborated *claims*: insights whose evidence has no cross-agent support.
        uncorroborated = sum(
            1 for i in (run.get("insights") or [])
            if not next(
                (f.get("corroborated_by") for f in findings if f.get("id") == i.get("finding_id")),
                None,
            )
        )

        return spec.result(
            round(min(1.0, score), 4),
            details={
                "supporting_source_count": len(providers),
                "distinct_domain_count": len(domains),
                "credible_source_count": len(credible),
                "weak_source_count": len(weak),
                "uncorroborated_claim_count": uncorroborated,
                "corroborated_finding_count": corroborated,
                "components": {k: (round(v, 3) if v is not None else None) for k, (v, _) in parts.items()},
                "recency_available": recency is not None,
            },
            notes="" if recency is not None else "no publication dates available; recency substituted by credibility",
        )


# ═════════════════════════════════════════════════════════════
# 4. GROUNDEDNESS   5. HALLUCINATION
# ═════════════════════════════════════════════════════════════
class GroundednessEvaluator:
    def evaluate(self, case: EvaluationCase, claims: list[ClaimRecord]) -> MetricResult:
        spec = M.spec(M.GROUNDEDNESS)
        factual = [c for c in claims if c.kind == KIND_FACTUAL]
        if not factual:
            return spec.unavailable("the run produced no factual claims to assess")

        supported = sum(1 for c in factual if c.verdict == CLAIM_SUPPORTED)
        partial = sum(1 for c in factual if c.verdict == CLAIM_PARTIAL)
        unsupported = sum(1 for c in factual if c.verdict == CLAIM_UNSUPPORTED)
        uncertain = sum(1 for c in factual if c.verdict == CLAIM_UNCERTAIN)

        score = (supported + 0.5 * partial) / len(factual)
        return spec.result(
            round(score, 4),
            details={
                "factual_claims_evaluated": len(factual),
                "supported": supported,
                "partially_supported": partial,
                "unsupported": unsupported,
                "uncertain": uncertain,
                "excluded_recommendations": sum(1 for c in claims if c.kind == KIND_RECOMMENDATION),
                "excluded_hypotheses": sum(1 for c in claims if c.kind == KIND_HYPOTHESIS),
                "interpretive_claims": sum(1 for c in claims if c.kind == KIND_INTERPRETIVE),
            },
        )


class HallucinationEvaluator:
    def evaluate(self, case: EvaluationCase, claims: list[ClaimRecord]) -> MetricResult:
        spec = M.spec(M.HALLUCINATION_RATE)
        factual = [c for c in claims if c.kind == KIND_FACTUAL]
        if not factual:
            return spec.unavailable("the run produced no factual claims to assess")

        unsupported = [c for c in factual if c.verdict == CLAIM_UNSUPPORTED]
        rate = len(unsupported) / len(factual)
        return spec.result(
            round(rate, 4),
            details={
                "total_factual_claims": len(factual),
                "unsupported_factual_claims": len(unsupported),
                "explicit_uncertainty": sum(1 for c in factual if c.verdict == CLAIM_UNCERTAIN),
                "reasonable_inference": sum(1 for c in claims if c.kind == KIND_INTERPRETIVE),
                "labelled_hypotheses_excluded": sum(1 for c in claims if c.kind == KIND_HYPOTHESIS),
                "recommendations_excluded": sum(1 for c in claims if c.kind == KIND_RECOMMENDATION),
                "examples": [
                    {"text": c.text[:160], "reason": c.reason} for c in unsupported[:3]
                ],
            },
        )


# ═════════════════════════════════════════════════════════════
# 6. RECOVERY
# ═════════════════════════════════════════════════════════════
class RecoveryEvaluator:
    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.RECOVERY_RATE)
        fw = _fw(run)
        injected = fw.get("injected_events") or []
        tool_errors = fw.get("tool_errors") or []
        fallbacks = fw.get("fallback_history") or []

        failure_events = [e for e in injected if e.get("type") in {"tool_failed", "tool_timeout"}]
        # Count distinct failing tools, since a timeout is a second attempt on the
        # same underlying failure rather than a new one.
        failed_tools = {e.get("tool") for e in failure_events if e.get("tool")}
        if not failed_tools and tool_errors:
            failed_tools = {e.get("tool") for e in tool_errors if e.get("tool")}

        if not failed_tools:
            if case.expected_recovery:
                return spec.result(
                    0.0,
                    details={"injected_failures": 0,
                             "note": "case expected an injected failure but none occurred"},
                    notes="expected recovery scenario did not inject a failure",
                )
            return spec.unavailable("no failures were injected in this run")

        recovered_tools = {f.get("tool") for f in fallbacks if f.get("recovered")}
        recovered = len(failed_tools & recovered_tools)
        rate = recovered / len(failed_tools)

        latencies = [
            int(t.get("latency_ms") or 0) for t in (fw.get("tool_executions") or [])
            if t.get("fallback_used")
        ]
        return spec.result(
            round(rate, 4),
            details={
                "injected_failures": len(failed_tools),
                "recovered_failures": recovered,
                "failure_types": sorted({str(e.get("type")) for e in failure_events}) or ["tool_error"],
                "failure_sources": sorted({str(f.get("from")) for f in fallbacks if f.get("from")}),
                "retry_attempts": max((int(t.get("attempt") or 1) for t in (fw.get("tool_executions") or [])), default=0),
                "fallback_used": len(fallbacks),
                "fallback_success_rate": round(
                    (sum(1 for f in fallbacks if f.get("recovered")) / len(fallbacks)) if fallbacks else 0.0, 3
                ),
                "mean_recovery_latency_ms": (round(M.mean_of([float(x) for x in latencies]) or 0.0, 1) if latencies else None),
                "post_recovery_task_status": run.get("status"),
                "final_task_status": run.get("status"),
            },
        )


# ═════════════════════════════════════════════════════════════
# 7. LATENCY   8. RESOURCE
# ═════════════════════════════════════════════════════════════
class LatencyEvaluator:
    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.LATENCY)
        mx, fw = _mx(run), _fw(run)
        total = mx.get("duration_ms")
        if not isinstance(total, (int, float)):
            return spec.unavailable("run did not report duration_ms")

        tool_ms = sum(int(t.get("latency_ms") or 0) for t in (fw.get("tool_executions") or []))
        return spec.result(
            float(total),
            details={
                "total_runtime_ms": int(total),
                "tool_time_ms": tool_ms,
                "agent_time_ms": max(0, int(total) - tool_ms),
                "recovery_time_ms": sum(
                    int(t.get("latency_ms") or 0)
                    for t in (fw.get("tool_executions") or []) if t.get("fallback_used")
                ),
                "graph_elapsed_ms": (fw.get("resource") or {}).get("elapsed_ms"),
                "stage_breakdown_note": (
                    "planning/synthesis stages are not separately instrumented; "
                    "tool time is measured and agent time is the remainder"
                ),
            },
        )


class ResourceEvaluator:
    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.RESOURCE_EFFICIENCY)
        mx, fw = _mx(run), _fw(run)
        resource = fw.get("resource") or {}
        tool_calls = int(mx.get("tool_calls") or len(fw.get("tool_executions") or []))
        llm_calls = int(mx.get("llm_calls") or resource.get("llm_calls") or 0)
        completed = run.get("status") in {"completed", "completed_partial"}
        cost = mx.get("estimated_cost", resource.get("estimated_cost"))

        # Efficiency as a bounded score: fewer calls per completed task is better,
        # normalised against the configured ceiling so it is comparable across cases.
        ceiling = float(resource.get("max_tool_calls") or 14)
        used_ratio = min(1.0, tool_calls / ceiling) if ceiling else 1.0
        score = round(max(0.0, 1.0 - used_ratio) if completed else 0.0, 4)

        return spec.result(
            score,
            method="1 - (tool_calls / configured_tool_ceiling), zero when the task did not complete",
            details={
                "tool_calls": tool_calls,
                "agent_calls": len(fw.get("completed_agents") or []),
                "llm_calls": llm_calls,
                "retry_count": sum(
                    1 for t in (fw.get("tool_executions") or []) if int(t.get("attempt") or 1) > 1
                ),
                "fallback_count": len(fw.get("fallback_history") or []),
                "parallel_task_count": int(mx.get("parallel_agents") or 0),
                "runtime_ms": mx.get("duration_ms"),
                "estimated_cost_usd": cost,
                "cost_per_completed_task": (round(float(cost), 5) if completed and isinstance(cost, (int, float)) else None),
                "tool_calls_per_successful_run": tool_calls if completed else None,
                "llm_calls_per_successful_run": llm_calls if completed else None,
                "token_usage": "unavailable — the configured provider does not expose token counts",
                "tool_ceiling": ceiling,
            },
        )


class EfficiencyEvaluator:
    """Useful output per unit of work."""

    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.EFFICIENCY)
        mx, fw = _mx(run), _fw(run)
        tool_calls = int(mx.get("tool_calls") or len(fw.get("tool_executions") or []))
        relevant = int(mx.get("findings_relevant") or 0)
        insights = len(run.get("insights") or [])
        if tool_calls <= 0:
            return spec.unavailable("no tool calls were made, so yield per call is undefined")

        yield_per_call = relevant / tool_calls
        # 4 relevant items per call treated as a strong yield ceiling.
        yield_score = min(1.0, yield_per_call / 4.0)
        produced = 1.0 if insights else 0.0
        score = round(0.7 * yield_score + 0.3 * produced, 4)
        return spec.result(
            score,
            details={
                "relevant_findings": relevant,
                "tool_calls": tool_calls,
                "relevant_per_tool_call": round(yield_per_call, 3),
                "insights_produced": insights,
                "runtime_ms": mx.get("duration_ms"),
            },
        )


# ═════════════════════════════════════════════════════════════
# 9. UNCERTAINTY   10. REFUSAL
# ═════════════════════════════════════════════════════════════
class UncertaintyEvaluator:
    """Is expressed certainty calibrated to evidence strength? (section 20)"""

    WEAK_EVIDENCE = 0.55
    STRONG_EVIDENCE = 0.70
    LOW_CONFIDENCE = 0.60

    def evaluate(
        self, case: EvaluationCase, run: dict[str, Any], evidence_quality: float | None
    ) -> MetricResult:
        spec = M.spec(M.UNCERTAINTY_HANDLING)
        fw = _fw(run)
        confidence = fw.get("overall_confidence")
        if not isinstance(confidence, (int, float)):
            return spec.unavailable("run reported no overall confidence")

        flags = fw.get("uncertainty_flags") or []
        unresolved = [c for c in (fw.get("conflicting_evidence") or []) if not c.get("resolved")]
        hedging = _refusal_present(run)
        expressed_uncertainty = bool(flags) or bool(unresolved) or hedging or confidence < self.LOW_CONFIDENCE

        strength = evidence_quality if isinstance(evidence_quality, (int, float)) else None
        verdict: str
        score: float

        if case.expects_uncertainty:
            # The case is built so that uncertainty is the correct answer.
            score = 1.0 if expressed_uncertainty else 0.0
            verdict = "correctly uncertain" if expressed_uncertainty else "confident despite weak/insufficient evidence"
        elif strength is None:
            score = 1.0 if expressed_uncertainty else 0.5
            verdict = "evidence strength unmeasurable; uncertainty expressed" if expressed_uncertainty else "evidence strength unmeasurable"
        elif strength < self.WEAK_EVIDENCE:
            score = 1.0 if expressed_uncertainty else 0.0
            verdict = "correctly uncertain on weak evidence" if expressed_uncertainty else "confident despite weak evidence"
        elif strength >= self.STRONG_EVIDENCE:
            if confidence >= self.LOW_CONFIDENCE:
                score, verdict = 1.0, "correctly confident with strong evidence"
            else:
                score, verdict = 0.6, "uncertain despite strong evidence — possible calibration issue"
        else:
            score, verdict = 0.8, "moderate evidence with proportionate confidence"

        return spec.result(
            round(score, 4),
            details={
                "verdict": verdict,
                "overall_confidence": round(float(confidence), 3),
                "evidence_quality": (round(strength, 3) if strength is not None else None),
                "uncertainty_flags": flags,
                "unresolved_conflicts": len(unresolved),
                "expressed_uncertainty": expressed_uncertainty,
                "case_expects_uncertainty": case.expects_uncertainty,
                "uncertainty_detection_accuracy": round(score, 3),
            },
        )


class RefusalEvaluator:
    """Did the agent refuse to assert a conclusion the evidence cannot support?"""

    def evaluate(self, case: EvaluationCase, run: dict[str, Any]) -> MetricResult:
        spec = M.spec(M.UNSUPPORTED_CONCLUSION_RATE)
        if not case.expected_refusal:
            return spec.unavailable("this case does not request an unsupported conclusion")

        text = " ".join([
            str(run.get("summary") or ""),
            " ".join(str(i.get("what_happened") or "") for i in (run.get("insights") or [])),
            " ".join(str(i.get("summary") or "") for i in (run.get("insights") or [])),
        ]).lower()

        # Restating the user's request is not asserting it. The agent's output
        # legitimately quotes the goal back ('... relevant to "Prove that X ..."'),
        # so the echoed goal is removed before scanning for assertion language —
        # otherwise a correct refusal is misread as an assertion.
        text = _strip_echoed_goal(text, str(run.get("goal") or ""))
        asserted = [cue for cue in _ASSERTION_CUES if cue in text]
        refused = _refusal_present(run)
        # Rate is over the single conclusion this case asked for.
        rate = 0.0 if (refused and not asserted) else 1.0
        return spec.result(
            round(rate, 4),
            method="1 if the unsupported conclusion was asserted, 0 if correctly declined",
            details={
                "conclusion_opportunities": 1,
                "asserted_unsupported": bool(asserted),
                "assertion_cues_found": asserted,
                "refusal_detected": refused,
                "unsupported_conclusion_refusal": bool(refused and not asserted),
                "verdict": (
                    "correctly declined to assert the unsupported conclusion"
                    if (refused and not asserted)
                    else "asserted a conclusion the evidence does not support"
                ),
            },
        )


# ═════════════════════════════════════════════════════════════
# 11. CONSISTENCY  (across repetitions)
# ═════════════════════════════════════════════════════════════
class ConsistencyEvaluator:
    def evaluate(self, runs: list[dict[str, Any]]) -> MetricResult:
        spec = M.spec(M.CONSISTENCY)
        if len(runs) < 2:
            return spec.unavailable(f"needs >=2 repetitions, have {len(runs)}")

        def ids(run: dict[str, Any]) -> set[str]:
            return {str(f.get("id")) for f in (run.get("findings") or [])}

        def priorities(run: dict[str, Any]) -> dict[str, int]:
            out: dict[str, int] = {}
            for i in run.get("insights") or []:
                p = str(i.get("priority") or "")
                out[p] = out.get(p, 0) + 1
            return out

        pairs = [(a, b) for idx, a in enumerate(runs) for b in runs[idx + 1:]]

        overlaps: list[float] = []
        for a, b in pairs:
            ia, ib = ids(a), ids(b)
            union = ia | ib
            overlaps.append((len(ia & ib) / len(union)) if union else 1.0)

        # Conclusion agreement: do the runs agree on the headline priority mix and
        # on whether a decision-grade (HIGH) item exists?
        conclusion: list[float] = []
        for a, b in pairs:
            ha = any(str(i.get("priority")) == "HIGH" for i in (a.get("insights") or []))
            hb = any(str(i.get("priority")) == "HIGH" for i in (b.get("insights") or []))
            conclusion.append(1.0 if ha == hb else 0.0)

        priority_agree: list[float] = []
        for a, b in pairs:
            pa, pb = priorities(a), priorities(b)
            keys = set(pa) | set(pb)
            if not keys:
                priority_agree.append(1.0)
                continue
            diffs = [
                1.0 - abs(pa.get(k, 0) - pb.get(k, 0)) / max(1, max(pa.get(k, 0), pb.get(k, 0)))
                for k in keys
            ]
            priority_agree.append(max(0.0, M.mean_of(diffs) or 0.0))

        conf_agree: list[float] = []
        for a, b in pairs:
            ca = float((_fw(a).get("overall_confidence") or 0))
            cb = float((_fw(b).get("overall_confidence") or 0))
            conf_agree.append(max(0.0, 1.0 - abs(ca - cb)))

        completion_agree: list[float] = []
        for a, b in pairs:
            completion_agree.append(1.0 if a.get("status") == b.get("status") else 0.0)

        components = {
            "finding_overlap": M.mean_of(overlaps) or 0.0,
            "conclusion_agreement": M.mean_of(conclusion) or 0.0,
            "priority_agreement": M.mean_of(priority_agree) or 0.0,
            "confidence_agreement": M.mean_of(conf_agree) or 0.0,
            "task_completion_agreement": M.mean_of(completion_agree) or 0.0,
        }
        score = (
            0.35 * components["finding_overlap"]
            + 0.25 * components["conclusion_agreement"]
            + 0.20 * components["priority_agreement"]
            + 0.10 * components["confidence_agreement"]
            + 0.10 * components["task_completion_agreement"]
        )
        return spec.result(
            round(score, 4),
            details={
                "repetitions": len(runs),
                "pairs_compared": len(pairs),
                "components": {k: round(v, 3) for k, v in components.items()},
                "note": "substantive agreement; exact wording is deliberately not compared",
            },
        )


# ═════════════════════════════════════════════════════════════
# 12. RELIABILITY + ROBUSTNESS  (suite level)
# ═════════════════════════════════════════════════════════════
class ReliabilityEvaluator:
    def evaluate(self, outcomes: list[str], statuses: list[str], latencies: list[float]) -> MetricResult:
        spec = M.spec(M.RELIABILITY)
        total = len(outcomes)
        if not total:
            return spec.unavailable("no repetitions recorded")
        successful = sum(1 for o in outcomes if o == "PASS")
        partial = sum(1 for o in outcomes if o == "PARTIAL")
        failed = sum(1 for o in outcomes if o in {"FAIL", "ERROR"})
        return spec.result(
            round(successful / total, 4),
            details={
                "total_runs": total,
                "successful_runs": successful,
                "partial_runs": partial,
                "failed_runs": failed,
                "partial_completion_rate": round(partial / total, 3),
                "failure_frequency": round(failed / total, 3),
                "mean_time_to_completion_ms": M.mean_of(latencies),
                "agent_statuses": statuses,
            },
        )


class RobustnessEvaluator:
    """Per-scenario scores, aggregated so one easy category cannot hide a collapse."""

    def evaluate(self, per_scenario: dict[str, dict[str, Any]]) -> MetricResult:
        spec = M.spec(M.ROBUSTNESS)
        measured = {k: v for k, v in per_scenario.items() if v.get("total")}
        if not measured:
            return spec.unavailable("no scenario categories were executed")
        scores = {k: float(v.get("score") or 0.0) for k, v in measured.items()}
        overall = M.mean_of(list(scores.values())) or 0.0
        worst = min(scores.items(), key=lambda kv: kv[1]) if scores else ("", 0.0)
        return spec.result(
            round(overall, 4),
            method="unweighted mean of per-scenario category scores",
            details={
                "per_category": {k: round(v, 3) for k, v in scores.items()},
                "categories_evaluated": len(scores),
                "weakest_category": worst[0],
                "weakest_score": round(worst[1], 3),
            },
        )
