"""Insight generator — turn evidence into a prioritized, actionable report.

Every insight answers five questions, in this order:
    WHAT HAPPENED · SUMMARY · WHY IT MATTERS · PRIORITY · RECOMMENDED ACTION

Priority is HIGH / MEDIUM / LOW and is always accompanied by the reasoning behind
it, because a bare label is not actionable. The LLM writes these when available;
the heuristic writer produces the same structure otherwise so the report is never
missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from ..tools.base import FindingRecord
from ..tools.signals import SIGNAL_LABELS, STRATEGIC_SIGNALS
from .decision_engine import RELEVANCE_THRESHOLD
from .llm import LLMClient
from .sanitize import UNTRUSTED_NOTICE, sanitize, wrap_untrusted
from .state import AgentState

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
MAX_INSIGHTS = 12

# A briefing where everything is urgent is a briefing nobody triages.
MAX_HIGH_PRIORITY = 4

# Per-item evidence budget sent to the model.
EVIDENCE_CHARS = 700

_SOURCE_WORDS = {
    "research": "A research paper",
    "patent": "A patent filing",
    "news": "An industry news report",
    "competitor": "A tracked competitor",
}

_ACTION_TEMPLATES = {
    "research": "Have an engineer review the method and assess whether it changes our technical roadmap.",
    "patent": "Ask IP counsel to review the claims for overlap with our own approach and freedom to operate.",
    "news": "Add this to the market brief and confirm whether it changes near-term positioning.",
    "competitor": "Compare the announced capability against our current offering and note the gap.",
}


@dataclass
class Insight:
    id: str
    finding_id: str
    title: str
    what_happened: str
    summary: str
    why_it_matters: str
    priority: str
    recommended_action: str
    source: str
    source_url: str
    provider: str = ""
    published_date: str | None = None
    competitor: str = ""
    signals: list[str] = field(default_factory=list)
    confidence: str = "standard"
    score: float = 0.0
    simulated: bool = False
    author: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "finding_id": self.finding_id,
            "title": self.title,
            "what_happened": self.what_happened,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "priority": self.priority,
            "recommended_action": self.recommended_action,
            "source": self.source,
            "source_url": self.source_url,
            "provider": self.provider,
            "published_date": self.published_date,
            "competitor": self.competitor,
            "signals": self.signals,
            "confidence": self.confidence,
            "score": round(self.score, 3),
            "simulated": self.simulated,
            "author": self.author,
        }

    def render(self) -> str:
        dot = {HIGH: "🔴", MEDIUM: "🟠", LOW: "⚪"}.get(self.priority, "⚪")
        return (
            f"{dot} {self.priority} PRIORITY\n"
            f"{self.title}\n\n"
            f"What Happened:\n{self.what_happened}\n\n"
            f"Summary:\n{self.summary}\n\n"
            f"Why It Matters:\n{self.why_it_matters}\n\n"
            f"Recommended Action:\n{self.recommended_action}\n\n"
            f"Source:\n{self.source} — {self.source_url or 'no link available'}"
        )


class InsightGenerator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm
        # Populated when the model returns the summary alongside the insights.
        self.executive_summary: str = ""

    async def generate(self, state: AgentState) -> list[Insight]:
        candidates = self._select(state)
        if not candidates:
            return []

        insights: list[Insight] = []
        if self.llm is not None and self.llm.available:
            insights = await self._llm_insights(state, candidates)

        written = {i.finding_id for i in insights}
        for finding in candidates:
            if finding.id not in written:
                insights.append(self._heuristic_insight(state, finding))

        insights = _cap_high_priority(insights)
        order = {HIGH: 0, MEDIUM: 1, LOW: 2}
        insights.sort(key=lambda i: (order.get(i.priority, 3), -i.score))
        return insights[:MAX_INSIGHTS]

    # ── selection ───────────────────────────────────────────
    def _select(self, state: AgentState) -> list[FindingRecord]:
        """Pick what is worth writing up, keeping the source mix balanced."""
        relevant = sorted(
            [f for f in state.findings if f.relevance >= RELEVANCE_THRESHOLD],
            key=lambda f: f.relevance,
            reverse=True,
        )
        if not relevant:
            # Nothing cleared the bar: report the best of what there is rather than
            # returning an empty report and calling that success.
            relevant = sorted(state.findings, key=lambda f: f.relevance, reverse=True)[:5]

        per_source_cap = 4
        counts: dict[str, int] = {}
        picked: list[FindingRecord] = []
        for finding in relevant:
            n = counts.get(finding.source, 0)
            if n >= per_source_cap:
                continue
            counts[finding.source] = n + 1
            picked.append(finding)
            if len(picked) >= MAX_INSIGHTS:
                break
        return picked

    # ── LLM path ────────────────────────────────────────────
    async def _llm_insights(
        self, state: AgentState, findings: list[FindingRecord]
    ) -> list[Insight]:
        assert self.llm is not None
        by_id = {f.id: f for f in findings}

        system = (
            "You are the analysis module of a competitive-intelligence agent. You "
            "convert collected evidence into a prioritized, actionable briefing for a "
            "busy analyst.\n\n"
            "For each item produce:\n"
            "  what_happened  — one factual sentence about the event itself\n"
            "  summary        — 1-2 sentences of plain language, no jargon\n"
            "  why_it_matters — why this affects the user's stated goal specifically\n"
            "  priority       — HIGH, MEDIUM or LOW\n"
            "  recommended_action — one concrete next step a human can take\n\n"
            "Priority rules:\n"
            "  HIGH   — changes a decision: a tracked competitor moved, a filing "
            "protects a capability we care about, or a result invalidates an assumption\n"
            "  MEDIUM — relevant and worth reading, but does not force a decision\n"
            "  LOW    — context or weak/unverified signal\n\n"
            "Never rate an unverified forum post HIGH on its own. Ground every claim "
            "in the provided evidence — do not invent facts, numbers or links.\n"
            f"{UNTRUSTED_NOTICE}\n"
            "Reply with ONLY a JSON object."
        )

        blocks = []
        for f in findings:
            # Keep the evidence tight: the model needs enough to judge, not the
            # whole document. Smaller prompts are cheaper and materially faster.
            body = sanitize(f.raw_text or f.summary, max_chars=EVIDENCE_CHARS)[0]
            blocks.append(
                f"ID: {f.id}\n"
                f"SOURCE_TYPE: {f.source} (provider: {f.provider}, credibility: {f.credibility})\n"
                f"TITLE: {sanitize(f.title, max_chars=200)[0]}\n"
                f"DATE: {f.published_date or 'unknown'}\n"
                f"COMPETITOR: {f.competitor or 'none'}\n"
                f"SIGNALS: {', '.join(f.signals) or 'none'}\n"
                f"CONTENT:\n{wrap_untrusted(f.provider, body)}\n"
            )

        user = (
            f"USER GOAL: {sanitize(state.user_goal)[0]}\n"
            f"TRACKED COMPANIES: {state.competitors or 'none'}\n"
            f"TOPICS: {state.keywords or state.tracking_topics}\n"
            f"TOOLS USED: {state.tools_used()}\n\n"
            f"EVIDENCE ({len(findings)} items):\n\n" + "\n---\n".join(blocks) + "\n\n"
            'Return JSON: {"insights": [{"id": "<the ID above>", "what_happened": "...", '
            '"summary": "...", "why_it_matters": "...", "priority": "HIGH|MEDIUM|LOW", '
            '"recommended_action": "..."}], '
            '"executive_summary": "two short paragraphs leading with the single most '
            'decision-relevant finding"}'
        )

        data = await self.llm.complete_json(
            purpose="insights", system=system, user=user, max_tokens=4096, temperature=0.3
        )
        state.llm_calls = self.llm.usage.calls
        if not data:
            return []

        # The executive summary rides along in the same call — one round trip
        # instead of two for output the user always wants together.
        self.executive_summary = _text(data.get("executive_summary"), 1600)

        raw_items = data.get("insights") or data.get("items") or []
        out: list[Insight] = []
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            finding = by_id.get(str(row.get("id") or ""))
            if finding is None:
                continue
            priority = _clean_priority(row.get("priority"))
            priority = _apply_credibility_ceiling(priority, finding)
            out.append(
                Insight(
                    id=f"ins_{finding.id}",
                    finding_id=finding.id,
                    title=finding.title,
                    what_happened=_text(row.get("what_happened"), 400)
                    or self._what_happened(finding),
                    summary=_text(row.get("summary"), 600) or finding.summary,
                    why_it_matters=_text(row.get("why_it_matters"), 600)
                    or self._why_it_matters(state, finding, priority),
                    priority=priority,
                    recommended_action=_text(row.get("recommended_action"), 400)
                    or _ACTION_TEMPLATES.get(finding.source, "Review and decide."),
                    source=_source_name(finding),
                    source_url=finding.url,
                    provider=finding.provider,
                    published_date=finding.published_date,
                    competitor=finding.competitor,
                    signals=finding.signals,
                    confidence=finding.credibility,
                    score=finding.relevance,
                    simulated=finding.simulated,
                    author=self.llm.reasoner_name,
                )
            )
        return out

    # ── heuristic path ──────────────────────────────────────
    def _heuristic_insight(self, state: AgentState, f: FindingRecord) -> Insight:
        priority = self._priority(state, f)
        return Insight(
            id=f"ins_{f.id}",
            finding_id=f.id,
            title=f.title,
            what_happened=self._what_happened(f),
            summary=f.summary or f.title,
            why_it_matters=self._why_it_matters(state, f, priority),
            priority=priority,
            recommended_action=self._action(f, priority),
            source=_source_name(f),
            source_url=f.url,
            provider=f.provider,
            published_date=f.published_date,
            competitor=f.competitor,
            signals=f.signals,
            confidence=f.credibility,
            score=f.relevance,
            simulated=f.simulated,
            author="heuristic-analyst",
        )

    def _priority(self, state: AgentState, f: FindingRecord) -> str:
        """Rule-based, not score-based.

        Topical relevance alone is never HIGH — otherwise every on-topic item is
        "urgent" and the priority column stops carrying information. HIGH means
        something a human should act on this week.
        """
        strategic = set(f.signals) & STRATEGIC_SIGNALS
        fresh = (_age_days(f.published_date) or 999) <= 30
        trusted = f.credibility in {"high", "standard"}

        high = (
            # A tracked competitor holds IP on something we care about.
            ("competitor-assignee" in f.signals)
            or (f.source == "patent" and bool(f.competitor))
            # A tracked competitor made a concrete strategic move, recently.
            or (bool(f.competitor) and bool(strategic) and fresh)
            # Not competitor-linked, but a strong, credible, strategic development.
            or (f.relevance >= 0.80 and bool(strategic) and trusted and fresh)
        )
        if high:
            return _apply_credibility_ceiling(HIGH, f)

        medium = (
            bool(f.competitor)
            or bool(strategic)
            or f.relevance >= 0.55
            or int(f.meta.get("citation_count") or 0) >= 5
        )
        return MEDIUM if medium else LOW

    def _what_happened(self, f: FindingRecord) -> str:
        who = f.competitor or (f.author.split(",")[0].strip() if f.author else "")
        lead = _SOURCE_WORDS.get(f.source, "A source")
        when = f" ({f.published_date})" if f.published_date else ""
        if f.source == "patent":
            assignee = f.meta.get("assignee") or who or "an unnamed assignee"
            return f"{lead} assigned to {assignee}{when} covers: {f.title}"
        if f.source == "competitor" and who:
            return f"{who} was reported in connection with: {f.title}{when}"
        if who:
            return f"{lead} from {who}{when}: {f.title}"
        return f"{lead}{when}: {f.title}"

    def _why_it_matters(self, state: AgentState, f: FindingRecord, priority: str) -> str:
        reasons: list[str] = []

        if f.competitor:
            reasons.append(
                f"{f.competitor} is on the tracked competitor list, so this is direct "
                f"movement inside the monitored competitive set"
            )
        matched = [k for k in (state.keywords + state.tracking_topics) if k.lower() in
                   f"{f.title} {f.summary}".lower()]
        if matched:
            reasons.append(f"it lands squarely on the tracked topic '{matched[0]}'")

        for signal in f.signals:
            if signal in SIGNAL_LABELS:
                reasons.append(SIGNAL_LABELS[signal])
                break

        if f.source == "patent":
            reasons.append("patent activity indicates where a company is locking in position")
        elif f.source == "research":
            citations = int(f.meta.get("citation_count") or 0)
            reasons.append(
                f"published research typically precedes product moves by months"
                + (f", and this one already has {citations} citations" if citations >= 5 else "")
            )

        if f.credibility == "unverified":
            reasons.append(
                "this is unverified community discussion, so treat it as an early "
                "indicator rather than evidence"
            )

        if not reasons:
            reasons.append(
                f"it is topically relevant to the goal but does not yet change a decision"
            )

        body = "; ".join(reasons[:3])
        return body[0].upper() + body[1:] + "."

    def _action(self, f: FindingRecord, priority: str) -> str:
        base = _ACTION_TEMPLATES.get(f.source, "Review and decide whether to act.")
        if priority == HIGH:
            if f.competitor and f.source == "patent":
                return (
                    f"Escalate to IP counsel this week: check {f.competitor}'s claims against "
                    f"our roadmap and confirm freedom to operate."
                )
            if f.competitor:
                return (
                    f"Brief the product team on {f.competitor}'s move and decide within the "
                    f"week whether positioning needs to change."
                )
            return base + " Treat as time-sensitive."
        if priority == LOW and f.credibility == "unverified":
            return "Log for pattern-watching only; do not act on a single unverified post."
        return base


# ─────────────────────────────────────────────────────────────
# Executive summary
# ─────────────────────────────────────────────────────────────
class SummaryWriter:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    async def write(self, state: AgentState, insights: list[Insight]) -> str:
        highs = [i for i in insights if i.priority == HIGH]
        mediums = [i for i in insights if i.priority == MEDIUM]

        if self.llm is not None and self.llm.available and insights:
            text = await self._llm_summary(state, insights)
            if text:
                return text

        return self._heuristic_summary(state, insights, highs, mediums)

    async def _llm_summary(self, state: AgentState, insights: list[Insight]) -> str:
        assert self.llm is not None
        payload = [
            {
                "priority": i.priority,
                "title": i.title[:160],
                "why_it_matters": i.why_it_matters[:240],
                "source": i.source,
            }
            for i in insights[:10]
        ]
        data = await self.llm.complete_json(
            purpose="summary",
            system=(
                "You write the two-paragraph executive summary at the top of an "
                "intelligence briefing. Lead with the single most decision-relevant "
                "finding. Be specific and concrete, no filler, no hedging. "
                "Reply with ONLY a JSON object."
            ),
            user=(
                f"GOAL: {sanitize(state.user_goal)[0]}\n"
                f"TOOLS USED: {state.tools_used()}\n"
                f"ITEMS COLLECTED: {len(state.findings)} "
                f"({len(state.relevant_findings())} relevant)\n"
                f"INSIGHTS:\n{json.dumps(payload, indent=2)[:2500]}\n\n"
                'Return JSON: {"summary": "two short paragraphs"}'
            ),
            max_tokens=700,
            temperature=0.35,
        )
        state.llm_calls = self.llm.usage.calls
        if not data:
            return ""
        return _text(data.get("summary"), 1600)

    def _heuristic_summary(
        self,
        state: AgentState,
        insights: list[Insight],
        highs: list[Insight],
        mediums: list[Insight],
    ) -> str:
        if not insights:
            return (
                f"No findings cleared the relevance bar for the goal \"{state.user_goal}\". "
                f"The agent ran {len(state.tool_calls)} tool call(s) across "
                f"{', '.join(state.tools_used()) or 'no tools'} and recorded "
                f"{len(state.errors)} error(s). Widening the keywords or the time window "
                f"is the next step."
            )

        lead = highs[0] if highs else insights[0]
        sources = ", ".join(f"{k} ({v})" for k, v in sorted(state.coverage().items()))
        parts = [
            f"Across {len(state.tool_calls)} tool call(s) over "
            f"{len(state.tools_used())} tool(s), the agent collected "
            f"{len(state.findings)} item(s) — {len(state.relevant_findings())} relevant "
            f"to \"{state.user_goal}\" — and produced {len(insights)} prioritized "
            f"insight(s): {len(highs)} high, {len(mediums)} medium, "
            f"{len(insights) - len(highs) - len(mediums)} low. Coverage by source: {sources}."
        ]

        headline = (
            f"The most decision-relevant item is \"{lead.title[:140]}\". "
            f"{lead.why_it_matters} Recommended next step: {lead.recommended_action}"
        )
        parts.append(headline)

        if state.competitors:
            cov = state.competitor_coverage()
            covered = [c for c, n in cov.items() if n]
            missing = [c for c, n in cov.items() if not n]
            line = f"Competitor coverage: {', '.join(covered) or 'none'} showed activity"
            if missing:
                line += f"; no recent activity surfaced for {', '.join(missing)}"
            parts.append(line + ".")

        if state.errors or state.simulated_data_used:
            notes = []
            if state.errors:
                notes.append(f"{len(state.errors)} provider issue(s) were handled without "
                             f"stopping the run")
            if state.simulated_data_used:
                notes.append("some providers had no API key configured and returned "
                             "clearly-labelled simulated data")
            parts.append("Caveats: " + "; ".join(notes) + ".")

        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _cap_high_priority(insights: list[Insight]) -> list[Insight]:
    """Keep the HIGH band scarce enough to mean something.

    Applies to LLM output too: a model asked to rate 12 items independently will
    happily call ten of them HIGH.
    """
    highs = [i for i in insights if i.priority == HIGH]
    if len(highs) <= MAX_HIGH_PRIORITY:
        return insights
    highs.sort(key=lambda i: i.score, reverse=True)
    for demoted in highs[MAX_HIGH_PRIORITY:]:
        demoted.priority = MEDIUM
        demoted.why_it_matters = (
            demoted.why_it_matters.rstrip(".")
            + f"; ranked below the top {MAX_HIGH_PRIORITY} items competing for attention "
            f"this cycle, so it is worth reading rather than acting on immediately."
        )
    return insights


def _clean_priority(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {HIGH, MEDIUM, LOW} else MEDIUM


def _apply_credibility_ceiling(priority: str, f: FindingRecord) -> str:
    """Unverified chatter cannot be the sole basis for a HIGH."""
    if f.credibility == "unverified" and priority == HIGH:
        return MEDIUM
    if f.credibility == "low" and priority == HIGH and not f.competitor:
        return MEDIUM
    return priority


def _source_name(f: FindingRecord) -> str:
    label = {
        "research": "Research",
        "patent": "Patent",
        "news": "News",
        "competitor": "Competitor activity",
    }.get(f.source, f.source.title())
    return f"{label} · {f.provider}" if f.provider else label


def _text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[: limit - 1] + "…" if len(text) > limit else text


def _age_days(published: str | None) -> int | None:
    if not published:
        return None
    try:
        d = date.fromisoformat(published[:10])
    except (ValueError, TypeError):
        return None
    return max(0, (datetime.now(UTC).date() - d).days)
