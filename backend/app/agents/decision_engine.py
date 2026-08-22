"""Decision engine — the ReAct core.

Responsibilities:
  1. ANALYZE an observation: score every item's relevance, detect strategic
     signals, decide whether an information need is now satisfied.
  2. DECIDE the next action from the *current state* — which tool, with which
     input, or stop.

The critical property: this is a policy over state, not a pipeline. The same goal
can produce different tool sequences depending on what earlier calls returned. In
particular:
  * `patent_search` is only reached when the goal is about IP, or when an earlier
    observation actually mentioned a filing.
  * `competitor_search` is only reached when there are companies to look at, and
    it re-targets whichever company is still uncovered.
  * A thin result triggers a *refined* retry, not a blind repeat.
  * A dead tool is dropped and the run continues with the rest.

An LLM chooses among the legal candidate actions when a key is configured; the
deterministic policy runs otherwise, and also acts as a guard-rail on the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any

from ..tools.base import FindingRecord, ToolInput, ToolResult
from ..tools.registry import ToolRegistry
from ..tools.signals import SIGNAL_LABELS, detect_signals
from .llm import LLMClient, clamp_int
from .sanitize import UNTRUSTED_NOTICE, sanitize
from .state import AgentState, Decision, InformationNeed, Observation

# Which tool satisfies which information need.
NEED_TO_TOOL: dict[str, str] = {
    "research": "research_search",
    "news": "news_search",
    "competitor": "competitor_search",
    "patent": "patent_search",
}
TOOL_TO_NEED = {v: k for k, v in NEED_TO_TOOL.items()}

RELEVANCE_THRESHOLD = 0.35
MIN_TOTAL_RELEVANT = 4
MAX_ATTEMPTS_PER_NEED = 3
MAX_CALLS_PER_TOOL = 4

_WORD = re.compile(r"[a-z0-9\-+]+")


# ─────────────────────────────────────────────────────────────
# Candidate actions
# ─────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    tool: str
    tool_input: ToolInput
    reason: str
    score: float
    need_key: str = ""
    kind: str = "satisfy_need"  # satisfy_need | follow_signal | fill_gap | refine

    def to_hint(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "why": self.reason,
            "kind": self.kind,
            "need": self.need_key,
            "priority": round(self.score, 2),
            "suggested_input": self.tool_input.describe(),
        }


# ─────────────────────────────────────────────────────────────
# Observation analysis
# ─────────────────────────────────────────────────────────────
class ObservationAnalyzer:
    """Turns a raw ToolResult into an Observation and folds it into state."""

    def analyze(
        self, state: AgentState, result: ToolResult, *, need_key: str = ""
    ) -> Observation:
        obs = Observation(
            iteration=state.iteration_count,
            tool=result.tool,
            items_returned=result.count,
            ok=result.ok,
            error=result.error,
        )

        new_items = 0
        duplicates = 0
        relevant = 0
        signals: set[str] = set()
        companies: set[str] = set()

        for item in result.items:
            item.relevance = self.score_relevance(state, item)
            item_signals = self.detect_signals(item)
            item.signals = sorted(set(item.signals) | set(item_signals))

            if state.register_finding(item):
                new_items += 1
            else:
                duplicates += 1
                continue

            if item.relevance >= RELEVANCE_THRESHOLD:
                relevant += 1
                signals.update(item_signals)
                for company in self.detect_companies(state, item):
                    companies.add(company)

        obs.new_items = new_items
        obs.duplicates = duplicates
        obs.relevant_items = relevant
        obs.signals = sorted(signals)
        obs.competitors_seen = sorted(companies)
        obs.top_titles = [
            i.title[:120]
            for i in sorted(result.items, key=lambda x: x.relevance, reverse=True)[:3]
        ]

        # Yield quality is what the next decision keys off.
        if not result.ok and not result.items:
            obs.yield_quality = "failed"
        elif result.count == 0:
            obs.yield_quality = "empty"
        elif relevant == 0 or (new_items and relevant / max(new_items, 1) < 0.34):
            obs.yield_quality = "thin"
        else:
            obs.yield_quality = "good"

        obs.summary = self._summarize(obs, result)

        state.detected_signals.update(signals)
        state.mentioned_companies.update(companies)
        state.observations.append(obs)

        # Update the plan: is the need this call was serving now satisfied?
        key = need_key or TOOL_TO_NEED.get(result.tool, "")
        need = state.plan.need(key) if key else None
        if need is not None:
            need.attempts += 1
            have = len(
                [
                    f
                    for f in state.findings_by_source(_source_for_need(key))
                    if f.relevance >= RELEVANCE_THRESHOLD
                ]
            )
            need.satisfied = have >= need.min_items
            if need.key == "competitor" and state.competitors and need.satisfied:
                # Volume is not coverage. The need is only met once every tracked
                # company has been accounted for, or we have run out of attempts.
                if state.uncovered_competitors() and need.attempts < MAX_ATTEMPTS_PER_NEED:
                    need.satisfied = False

        return obs

    # ── scoring ─────────────────────────────────────────────
    def score_relevance(self, state: AgentState, item: FindingRecord) -> float:
        text = f"{item.title} {item.summary} {item.raw_text}".lower()
        tokens = set(_WORD.findall(text))

        score = 0.0

        # 1. Topical overlap — the dominant term.
        keyword_terms = [k.lower() for k in (state.keywords + state.tracking_topics) if k]
        if keyword_terms:
            hits = 0
            for term in keyword_terms:
                parts = [p for p in _WORD.findall(term) if len(p) > 2]
                if not parts:
                    continue
                if term in text:
                    hits += 1
                elif all(p in tokens for p in parts):
                    hits += 0.7
                elif any(p in tokens for p in parts):
                    hits += 0.3
            score += min(0.45, 0.45 * (hits / max(1, len(keyword_terms))))
        else:
            score += 0.2

        # 2. Tracked-company mention.
        if item.competitor:
            score += 0.20
        elif any(c.lower() in text for c in state.competitors):
            score += 0.14

        # 3. Recency.
        age = _age_days(item.published_date)
        if age is not None:
            if age <= 7:
                score += 0.15
            elif age <= 30:
                score += 0.09
            elif age <= 90:
                score += 0.03

        # 4. Source credibility.
        score += {"high": 0.08, "standard": 0.03, "low": -0.05, "unverified": -0.08}.get(
            item.credibility, 0.0
        )

        # 5. Strategic signals.
        score += min(0.15, 0.05 * len(self.detect_signals(item)))

        # 6. Cheap authority proxy for papers.
        citations = int(item.meta.get("citation_count") or 0)
        if citations >= 20:
            score += 0.06
        elif citations >= 5:
            score += 0.03

        return round(max(0.0, min(1.0, score)), 3)

    def detect_signals(self, item: FindingRecord) -> list[str]:
        return detect_signals(item.title, item.summary, item.raw_text)

    def detect_companies(self, state: AgentState, item: FindingRecord) -> list[str]:
        text = f"{item.title} {item.summary} {item.raw_text} {item.author}".lower()
        found = [c for c in state.competitors if c.lower() in text]
        if item.competitor and item.competitor not in found:
            found.append(item.competitor)
        return found

    def _summarize(self, obs: Observation, result: ToolResult) -> str:
        if obs.yield_quality == "failed":
            return f"{result.tool} failed: {result.error or 'unknown error'}"
        bits = [f"{obs.items_returned} item(s) returned"]
        if obs.new_items != obs.items_returned:
            bits.append(f"{obs.new_items} new, {obs.duplicates} already seen")
        bits.append(f"{obs.relevant_items} relevant to the goal")
        if obs.signals:
            bits.append("signals: " + ", ".join(obs.signals))
        if result.providers_failed:
            failed = ", ".join(p["provider"] for p in result.providers_failed)
            bits.append(f"providers degraded: {failed}")
        if result.simulated:
            bits.append("some data simulated")
        return "; ".join(bits)


# ─────────────────────────────────────────────────────────────
# Decision engine
# ─────────────────────────────────────────────────────────────
class DecisionEngine:
    def __init__(self, tools: ToolRegistry, llm: LLMClient | None = None) -> None:
        self.tools = tools
        self.llm = llm

    # ── public API ──────────────────────────────────────────
    async def decide(self, state: AgentState) -> Decision:
        """Choose the next action. Never raises."""
        if state.iteration_count >= state.max_iterations:
            return Decision(
                action="finalize",
                reasoning=(
                    f"Iteration limit of {state.max_iterations} reached — summarizing "
                    f"what has been collected instead of continuing."
                ),
                author="guardrail",
                iteration=state.iteration_count,
            )

        candidates = self.candidates(state)
        done, why = self.assess_completion(state, candidates)
        if done:
            return Decision(
                action="finalize", reasoning=why, author="policy", iteration=state.iteration_count
            )

        if not candidates:
            return Decision(
                action="finalize",
                reasoning="No further productive tool call is available; moving to analysis.",
                author="policy",
                iteration=state.iteration_count,
            )

        # Only consult the model when there is an actual choice to make. With a
        # single legal action there is nothing to deliberate about, and spending a
        # model call (and its latency) on a foregone conclusion is waste.
        if self.llm is not None and self.llm.available and len(candidates) > 1:
            decision = await self._llm_decide(state, candidates)
            if decision is not None:
                return decision

        best = candidates[0]
        reasoning = best.reason
        if len(candidates) == 1 and self.llm is not None and self.llm.available:
            reasoning += " (Only one productive action available, so no deliberation needed.)"
        return Decision(
            action="call_tool",
            tool=best.tool,
            tool_input=best.tool_input,
            reasoning=reasoning,
            expected_gain=best.kind,
            confidence=min(0.95, 0.5 + best.score / 20),
            author="heuristic-policy",
            iteration=state.iteration_count,
        )

    # ── candidate generation (the actual policy) ─────────────
    def candidates(self, state: AgentState) -> list[Candidate]:
        out: list[Candidate] = []
        usable = set(self.tools.usable_names())

        for need in state.plan.needs:
            tool_name = NEED_TO_TOOL.get(need.key)
            if tool_name is None or tool_name not in usable:
                continue
            if state.call_count(tool_name) >= MAX_CALLS_PER_TOOL:
                continue
            if need.attempts >= MAX_ATTEMPTS_PER_NEED:
                continue

            # 1. Unsatisfied need → go get it.
            if not need.satisfied:
                gate = self._gate(state, need)
                if gate is None:
                    continue
                score, extra_reason, kind = gate
                out.append(
                    Candidate(
                        tool=tool_name,
                        tool_input=self._build_input(state, need),
                        reason=self._reason_for(state, need, extra_reason),
                        score=score,
                        need_key=need.key,
                        kind=kind,
                    )
                )
                continue

            # 2. Satisfied, but a competitor is still uncovered.
            if need.key == "competitor":
                uncovered = state.uncovered_competitors()
                if uncovered:
                    out.append(
                        Candidate(
                            tool=tool_name,
                            tool_input=self._build_input(state, need, focus=uncovered[:2]),
                            reason=(
                                f"Coverage is uneven — no activity found yet for "
                                f"{', '.join(uncovered[:2])}. Checking them specifically."
                            ),
                            score=6.0,
                            need_key=need.key,
                            kind="fill_gap",
                        )
                    )

        # 3. Follow a signal the agent discovered, even if the plan didn't ask for it.
        out.extend(self._signal_followups(state, usable))

        # 4. Still too little to report on → widen with the best remaining tool.
        if len(state.relevant_findings()) < MIN_TOTAL_RELEVANT:
            for name in ("news_search", "research_search"):
                if name not in usable or state.call_count(name) >= MAX_CALLS_PER_TOOL:
                    continue
                need = state.plan.need(TOOL_TO_NEED[name]) or InformationNeed(
                    key=TOOL_TO_NEED[name], reason="broadening for coverage", required=False
                )
                if need.attempts >= MAX_ATTEMPTS_PER_NEED:
                    continue
                out.append(
                    Candidate(
                        tool=name,
                        tool_input=self._build_input(state, need),
                        reason=(
                            f"Only {len(state.relevant_findings())} relevant item(s) so far — "
                            f"below the {MIN_TOTAL_RELEVANT} needed for a useful report. "
                            f"Widening the search."
                        ),
                        score=4.0,
                        need_key=need.key,
                        kind="fill_gap",
                    )
                )

        # Drop anything already tried with identical input, and anything whose tool
        # has failed hard twice.
        filtered = [
            c
            for c in out
            if c.tool_input.signature(c.tool) not in state.call_signatures
            and self._tool_failures(state, c.tool) < 2
        ]
        filtered.sort(key=lambda c: c.score, reverse=True)

        # One candidate per tool is enough for a single decision.
        seen_tools: set[str] = set()
        unique: list[Candidate] = []
        for c in filtered:
            if c.tool in seen_tools:
                continue
            seen_tools.add(c.tool)
            unique.append(c)
        return unique

    def _gate(
        self, state: AgentState, need: InformationNeed
    ) -> tuple[float, str, str] | None:
        """Should this unsatisfied need be pursued *now*? Returns (score, why, kind)."""
        base = 10.0 if need.required else 3.0
        # Earlier needs in the plan carry slightly more weight, so the opening move
        # matches the plan without being hard-coded.
        base -= 0.1 * state.plan.needs.index(need)

        if need.key == "competitor":
            if not state.competitors:
                return None
            return base + 1.0, "", "satisfy_need"

        if need.key == "patent":
            if need.required:
                return base + 1.0, "", "satisfy_need"
            # Not required by the plan: only worth a call if something pointed at IP.
            if state.has_signal("patent"):
                return (
                    7.5,
                    "an earlier observation referenced a patent or filing",
                    "follow_signal",
                )
            # Or if a tracked competitor is clearly active and IP posture matters.
            if state.competitors and len(state.relevant_findings()) >= 3:
                active = [c for c, n in state.competitor_coverage().items() if n >= 2]
                if active:
                    return (
                        5.0,
                        f"{active[0]} is active across sources, so its IP posture matters",
                        "follow_signal",
                    )
            return None

        if need.attempts >= 1:
            last = self._last_observation_for(state, NEED_TO_TOOL[need.key])
            if last is not None and last.yield_quality in {"thin", "empty"}:
                return base - 1.0, f"the previous attempt was {last.yield_quality}", "refine"
        return base, "", "satisfy_need"

    def _signal_followups(self, state: AgentState, usable: set[str]) -> list[Candidate]:
        out: list[Candidate] = []

        # A company surfaced that we are tracking but have not investigated directly.
        if "competitor_search" in usable and state.competitors:
            untouched = [
                c
                for c in state.mentioned_companies
                if c in state.competitors and state.competitor_coverage().get(c, 0) == 0
            ]
            if untouched and state.call_count("competitor_search") < MAX_CALLS_PER_TOOL:
                need = state.plan.need("competitor") or InformationNeed(key="competitor")
                if need.attempts < MAX_ATTEMPTS_PER_NEED:
                    out.append(
                        Candidate(
                            tool="competitor_search",
                            tool_input=self._build_input(state, need, focus=untouched[:2]),
                            reason=(
                                f"{', '.join(untouched[:2])} appeared in results but has not "
                                f"been investigated directly yet."
                            ),
                            score=6.5,
                            need_key="competitor",
                            kind="follow_signal",
                        )
                    )

        # Strong market signals with no news coverage yet → confirm in the news.
        market_signals = {"launch", "funding", "acquisition", "partnership", "regulatory"}
        if (
            "news_search" in usable
            and state.detected_signals & market_signals
            and not state.findings_by_source("news")
            and state.call_count("news_search") < MAX_CALLS_PER_TOOL
        ):
            need = state.plan.need("news") or InformationNeed(key="news", required=False)
            if need.attempts < MAX_ATTEMPTS_PER_NEED:
                hit = sorted(state.detected_signals & market_signals)[0]
                out.append(
                    Candidate(
                        tool="news_search",
                        tool_input=self._build_input(state, need),
                        reason=(
                            f"{SIGNAL_LABELS.get(hit, hit)} but there is no news coverage in "
                            f"the findings yet — checking whether this reached the market."
                        ),
                        score=6.0,
                        need_key="news",
                        kind="follow_signal",
                    )
                )
        return out

    # ── input construction ──────────────────────────────────
    def _build_input(
        self,
        state: AgentState,
        need: InformationNeed,
        *,
        focus: list[str] | None = None,
    ) -> ToolInput:
        keywords = state.keywords or state.tracking_topics or [state.user_goal[:60]]
        attempt = need.attempts

        # Vary the query across attempts so a retry is a genuine refinement.
        if attempt == 0:
            use_keywords = keywords[:3]
            since = 45
        elif attempt == 1:
            use_keywords = keywords[:1]
            since = 120
        else:
            use_keywords = keywords[1:3] or keywords[:1]
            since = 180

        base = ToolInput(
            query=" ".join(use_keywords[:2]),
            keywords=use_keywords,
            competitors=focus or state.competitors,
            limit=10,
            since_days=since,
        )

        if need.key == "competitor":
            targets = focus or state.uncovered_competitors() or state.competitors
            base.competitors = targets[:3]
            base.limit = 12
        elif need.key == "patent":
            base.since_days = max(since, 180)
            base.limit = 8
        elif need.key == "research":
            base.since_days = max(since, 60)

        return base.normalized()

    def _reason_for(self, state: AgentState, need: InformationNeed, extra: str) -> str:
        label = {
            "research": "research coverage",
            "news": "industry news coverage",
            "competitor": "competitor coverage",
            "patent": "patent coverage",
        }[need.key]

        if need.attempts == 0:
            head = f"No {label} yet."
            if need.key == "competitor" and state.competitors:
                head = f"No {label} yet for {', '.join(state.competitors[:3])}."
        else:
            head = f"{label.capitalize()} is still short of the {need.min_items} item(s) needed."

        tail = need.reason if need.attempts == 0 else "Retrying with a refined query."
        if extra:
            tail = extra.capitalize() + ". " + tail
        return f"{head} {tail}".strip()

    # ── completion ──────────────────────────────────────────
    def assess_completion(
        self, state: AgentState, candidates: list[Candidate]
    ) -> tuple[bool, str]:
        outstanding = state.plan.unsatisfied_required()
        relevant = len(state.relevant_findings())

        if not state.tool_calls:
            return False, ""

        if outstanding:
            reachable = {c.need_key for c in candidates}
            if any(n.key in reachable for n in outstanding):
                return False, ""
            names = ", ".join(n.key for n in outstanding)
            return True, (
                f"Cannot make further progress on: {names} (tool exhausted or "
                f"unavailable). Proceeding to analysis with {relevant} relevant item(s)."
            )

        if relevant >= MIN_TOTAL_RELEVANT:
            sources = ", ".join(sorted(state.coverage()))
            return True, (
                f"All required information needs are satisfied with {relevant} relevant "
                f"item(s) across {sources}. Enough evidence to prioritize."
            )

        if not candidates:
            return True, (
                f"No productive action remains. Reporting on {relevant} relevant item(s)."
            )
        return False, ""

    # ── LLM decision with guard-rails ───────────────────────
    async def _llm_decide(
        self, state: AgentState, candidates: list[Candidate]
    ) -> Decision | None:
        assert self.llm is not None
        system = (
            "You are the decision module of an autonomous competitive-intelligence "
            "agent running a ReAct loop. Given the goal, the plan, what has already "
            "been collected and the legal next actions, choose ONE next action.\n\n"
            "Rules:\n"
            "- Choose `call_tool` only with a tool from `legal_actions`.\n"
            "- Do not repeat a call that has already been made.\n"
            "- Choose `finalize` when the evidence is sufficient to write a "
            "prioritized report, or when no action would add value.\n"
            "- `reasoning` must be one or two plain sentences an analyst could read: "
            "what you are doing and why, based on the observations. Do not include "
            "internal deliberation.\n"
            f"- {UNTRUSTED_NOTICE}\n"
            "Reply with ONLY a JSON object."
        )
        user = (
            f"GOAL: {sanitize(state.user_goal)[0]}\n"
            f"PLAN: {state.plan.interpretation}\n"
            f"REQUIRED NEEDS: {[n.key for n in state.plan.needs if n.required]}\n"
            f"NEEDS STATUS: {_needs_status(state)}\n"
            f"ITERATION: {state.iteration_count + 1} of {state.max_iterations}\n"
            f"COVERAGE BY SOURCE: {state.coverage()}\n"
            f"COMPETITOR COVERAGE: {state.competitor_coverage()}\n"
            f"RELEVANT ITEMS SO FAR: {len(state.relevant_findings())}\n"
            f"SIGNALS DETECTED: {sorted(state.detected_signals) or 'none'}\n\n"
            f"RECENT OBSERVATIONS:\n{self._observation_digest(state)}\n\n"
            f"LEGAL ACTIONS (choose one tool from here):\n"
            f"{_format_hints(candidates)}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "action": "call_tool" | "finalize",\n'
            '  "tool": "<tool name or null>",\n'
            '  "tool_input": {"query": "...", "keywords": ["..."], '
            '"competitors": ["..."], "since_days": 45, "limit": 10},\n'
            '  "reasoning": "one or two sentences",\n'
            '  "expected_gain": "what this should add",\n'
            '  "confidence": 0.0\n'
            "}"
        )

        data = await self.llm.complete_json(
            purpose="decide", system=system, user=user, max_tokens=800
        )
        state.llm_calls = self.llm.usage.calls
        if not data:
            return None

        action = str(data.get("action") or "").strip().lower()
        reasoning = str(data.get("reasoning") or "").strip()[:600]

        if action == "finalize":
            # Guard-rail: don't let the model quit while a required need is still
            # reachable and there is budget left to satisfy it.
            outstanding = state.plan.unsatisfied_required()
            reachable = [c for c in candidates if c.need_key in {n.key for n in outstanding}]
            if outstanding and reachable and state.budget_left() > 2:
                best = reachable[0]
                return Decision(
                    action="call_tool",
                    tool=best.tool,
                    tool_input=best.tool_input,
                    reasoning=(
                        f"{best.reason} (Overriding an early stop: '{outstanding[0].key}' is "
                        f"still required and reachable.)"
                    ),
                    expected_gain=best.kind,
                    confidence=0.7,
                    author="policy-guardrail",
                    iteration=state.iteration_count,
                )
            return Decision(
                action="finalize",
                reasoning=reasoning or "Evidence is sufficient to prioritize.",
                author=self.llm.reasoner_name,
                iteration=state.iteration_count,
            )

        tool = str(data.get("tool") or "").strip()
        legal = {c.tool: c for c in candidates}
        if tool not in legal:
            return None  # fall back to the deterministic policy

        chosen = legal[tool]
        merged = self._merge_tool_input(chosen.tool_input, data.get("tool_input"))
        if merged.signature(tool) in state.call_signatures:
            merged = chosen.tool_input  # model tried to repeat; use the policy's input

        return Decision(
            action="call_tool",
            tool=tool,
            tool_input=merged,
            reasoning=reasoning or chosen.reason,
            expected_gain=str(data.get("expected_gain") or chosen.kind)[:200],
            confidence=_confidence(data.get("confidence")),
            author=self.llm.reasoner_name,
            iteration=state.iteration_count,
        )

    def _merge_tool_input(self, base: ToolInput, raw: Any) -> ToolInput:
        if not isinstance(raw, dict):
            return base
        from .llm import coerce_str_list

        return replace(
            base,
            query=str(raw.get("query") or base.query)[:300],
            keywords=coerce_str_list(raw.get("keywords"), limit=6) or base.keywords,
            competitors=coerce_str_list(raw.get("competitors"), limit=4) or base.competitors,
            since_days=clamp_int(raw.get("since_days"), 1, 365, base.since_days),
            limit=clamp_int(raw.get("limit"), 1, 25, base.limit),
        ).normalized()

    # ── small helpers ───────────────────────────────────────
    def _observation_digest(self, state: AgentState, limit: int = 4) -> str:
        if not state.observations:
            return "  (none yet)"
        lines = []
        for obs in state.observations[-limit:]:
            lines.append(
                f"  - step {obs.iteration} {obs.tool}: {obs.summary} "
                f"[yield: {obs.yield_quality}]"
            )
            for title in obs.top_titles[:2]:
                clean, _ = sanitize(title, max_chars=110)
                lines.append(f"      • {clean}")
        return "\n".join(lines)

    def _tool_failures(self, state: AgentState, tool: str) -> int:
        return sum(1 for c in state.tool_calls if c.tool == tool and not c.ok)

    def _last_observation_for(self, state: AgentState, tool: str) -> Observation | None:
        for obs in reversed(state.observations):
            if obs.tool == tool:
                return obs
        return None


# ─────────────────────────────────────────────────────────────
# module helpers
# ─────────────────────────────────────────────────────────────
def _source_for_need(need_key: str) -> str:
    return {
        "research": "research",
        "news": "news",
        "competitor": "competitor",
        "patent": "patent",
    }.get(need_key, need_key)


def _age_days(published: str | None) -> int | None:
    if not published:
        return None
    try:
        d = date.fromisoformat(published[:10])
    except (ValueError, TypeError):
        return None
    return max(0, (datetime.now(UTC).date() - d).days)


def _confidence(value: Any) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.6
    return round(max(0.0, min(1.0, c)), 2)


def _format_hints(candidates: list[Candidate]) -> str:
    import json

    return json.dumps([c.to_hint() for c in candidates], indent=2)[:2600]


def _needs_status(state: AgentState) -> str:
    parts = []
    for need in state.plan.needs:
        status = "satisfied" if need.satisfied else f"{need.attempts} attempt(s)"
        parts.append(f"{need.key}: {status}")
    return "{" + ", ".join(parts) + "}"
