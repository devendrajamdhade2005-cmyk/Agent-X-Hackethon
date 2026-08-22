"""Build the *minimum relevant* context for one agent and one objective.

This is the part that makes context sharing meaningful rather than decorative. The
easy implementation — hand every agent the whole run history — is explicitly what the
requirement rules out, and it is also bad engineering: it grows without bound, it
buries the one fact that matters, and it lets an unrelated earlier topic steer a new
search.

So context is assembled per (agent, objective):

  * the Research Agent gets research framing and, when it is following up on a
    competitor claim, only that claim;
  * the Competitive Agent gets tracked companies and the specific research findings
    that justify checking them — not the whole research feed;
  * the Orchestrator gets the wider view, because deciding what happens next is its
    job.

Every packet records what was included, what was deliberately withheld and why. The
UI reads that record, so the per-agent "context received" list is derived from the
real construction rather than hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..agents.messages import COMPETITIVE_AGENT, ORCHESTRATOR, RESEARCH_AGENT
from .working import MemoryFact, WorkingMemory

_WORD = re.compile(r"[a-z0-9\-+]+")

# A fact must clear this to be worth spending an agent's context on.
RELEVANCE_FLOOR = 0.28

# Hard caps. Context is a budget, not a dump.
MAX_FACTS_PER_AGENT = 5
MAX_MEMORIES_PER_AGENT = 3
MAX_FACT_CHARS = 220

# Signals that make a research finding worth a competitive check.
MARKET_SIGNALS = {"launch", "funding", "acquisition", "partnership", "regulatory",
                  "hiring", "benchmark", "patent"}

# Human labels for the section keys, used by the UI's "context received" ticks.
SECTION_LABELS = {
    "goal": "Original goal",
    "objective": "Current objective",
    "topics": "Topics",
    "research_topics": "Research focus",
    "competitors": "Tracked competitors",
    "entities": "Relevant entities",
    "time_scope": "Time scope",
    "constraints": "User constraints",
    "follow_up_reason": "Reason for follow-up",
    "research_findings": "Relevant research findings",
    "competitive_findings": "Relevant competitive findings",
    "prior_context": "Relevant historical context",
    "historical_summary": "Earlier-run summary",
    "coverage_gaps": "Coverage gaps",
    "pending_questions": "Open questions",
    "agent_status": "Agent statuses",
    "plan_state": "Execution plan state",
    "run_summary": "Compressed run history",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2}


@dataclass
class AgentContextPacket:
    """A scoped, auditable context hand-off."""

    target_agent: str
    objective: str
    goal: str = ""
    kind: str = "primary"
    sections: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, str]] = field(default_factory=list)
    omitted: list[dict[str, str]] = field(default_factory=list)
    # Concrete phrases the receiving agent should steer its search toward. This is
    # the lever that makes shared context change behaviour, not just appear in logs.
    focus_terms: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    source_agents: list[str] = field(default_factory=list)
    memory_version: int = 0

    @property
    def included(self) -> list[str]:
        return list(self.sections.keys())

    @property
    def size_chars(self) -> int:
        return len(str(self.sections))

    def label_list(self) -> list[str]:
        """Section labels, for the UI's per-agent context checklist."""
        return [SECTION_LABELS.get(k, k.replace("_", " ").title()) for k in self.sections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_agent": self.target_agent,
            "objective": self.objective,
            "goal": self.goal,
            "kind": self.kind,
            "sections": self.sections,
            "included": self.included,
            "labels": self.label_list(),
            "provenance": self.provenance,
            "omitted": self.omitted,
            "focus_terms": self.focus_terms,
            "fact_ids": self.fact_ids,
            "memory_ids": self.memory_ids,
            "source_agents": self.source_agents,
            "memory_version": self.memory_version,
            "size_chars": self.size_chars,
        }


class ContextBuilder:
    """Assembles `AgentContextPacket`s from working memory."""

    def build(
        self,
        *,
        target_agent: str,
        objective: str,
        memory: WorkingMemory,
        kind: str = "primary",
        reason: str = "",
        trigger_facts: list[MemoryFact] | None = None,
    ) -> AgentContextPacket:
        ctx = memory.task_context
        packet = AgentContextPacket(
            target_agent=target_agent,
            objective=objective,
            goal=(ctx.user_goal if ctx else ""),
            kind=kind,
            memory_version=memory.version,
        )

        # Always present: the goal and this agent's objective. Everything else is
        # earned by relevance.
        if packet.goal:
            packet.sections["goal"] = packet.goal
            packet.provenance.append({"section": "goal", "why": "the run's original goal"})
        packet.sections["objective"] = objective
        packet.provenance.append(
            {"section": "objective", "why": "what this agent is being asked to do now"}
        )

        if target_agent == RESEARCH_AGENT.key:
            self._research(packet, memory, reason, trigger_facts or [])
        elif target_agent == COMPETITIVE_AGENT.key:
            self._competitive(packet, memory, reason, trigger_facts or [])
        else:
            self._orchestrator(packet, memory)

        self._add_prior_context(packet, memory)
        packet.focus_terms = self._focus_terms(packet, memory)
        return packet

    # ── Research Intelligence Agent ─────────────────────────
    def _research(
        self,
        packet: AgentContextPacket,
        memory: WorkingMemory,
        reason: str,
        trigger_facts: list[MemoryFact],
    ) -> None:
        ctx = memory.task_context
        if ctx is not None:
            topics = ctx.research_topics or ctx.topics
            if topics:
                packet.sections["research_topics"] = topics[:6]
                packet.provenance.append(
                    {"section": "research_topics", "why": "the technical subject to search"}
                )
            if ctx.entities:
                packet.sections["entities"] = ctx.entities[:6]
                packet.provenance.append(
                    {"section": "entities", "why": "named things worth matching in results"}
                )
            if ctx.time_scope and ctx.time_scope != "unspecified":
                packet.sections["time_scope"] = ctx.time_scope
                packet.provenance.append(
                    {"section": "time_scope", "why": "recency the user asked for"}
                )
            if ctx.constraints:
                packet.sections["constraints"] = ctx.constraints
                packet.provenance.append(
                    {"section": "constraints", "why": "explicit user restrictions"}
                )

        # The competitive feed is withheld unless this *is* a validation follow-up.
        # A research agent asked to survey a field does not need the news feed, and
        # giving it one would let market noise steer an academic search.
        competitive = memory.facts_by_agent(COMPETITIVE_AGENT.key)
        if packet.kind == "follow_up" and (trigger_facts or competitive):
            source = trigger_facts or competitive
            picked = self._pick(source, packet.objective, memory, limit=2)
            if picked:
                packet.sections["competitive_findings"] = [self._render(f) for f in picked]
                packet.provenance.append({
                    "section": "competitive_findings",
                    "why": "the competitor claim this agent is being asked to validate",
                })
                self._track(packet, picked)
            withheld = len(competitive) - len(picked)
            if withheld > 0:
                packet.omitted.append({
                    "section": "competitive_findings",
                    "why": f"{withheld} other competitive finding(s) withheld — not part of "
                           f"this validation objective",
                })
        elif competitive:
            packet.omitted.append({
                "section": "competitive_findings",
                "why": f"{len(competitive)} competitive finding(s) withheld — this is a "
                       f"research objective, not a validation follow-up",
            })

        if reason:
            packet.sections["follow_up_reason"] = reason
            packet.provenance.append(
                {"section": "follow_up_reason", "why": "why this work was requested"}
            )

    # ── Competitive Intelligence Agent ──────────────────────
    def _competitive(
        self,
        packet: AgentContextPacket,
        memory: WorkingMemory,
        reason: str,
        trigger_facts: list[MemoryFact],
    ) -> None:
        ctx = memory.task_context
        if ctx is not None:
            if ctx.competitors:
                packet.sections["competitors"] = ctx.competitors[:6]
                packet.provenance.append(
                    {"section": "competitors", "why": "the companies being tracked"}
                )
            if ctx.entities:
                packet.sections["entities"] = ctx.entities[:6]
                packet.provenance.append(
                    {"section": "entities", "why": "named things worth matching in results"}
                )
            if ctx.constraints:
                packet.sections["constraints"] = ctx.constraints
                packet.provenance.append(
                    {"section": "constraints", "why": "explicit user restrictions"}
                )

        # The heart of the requirement: research findings from an earlier step become
        # this agent's context — but only the ones with competitive bearing.
        research = memory.facts_by_agent(RESEARCH_AGENT.key)
        candidates = trigger_facts or [f for f in research if self._competitively_relevant(f)]
        picked = self._pick(candidates, packet.objective, memory, limit=3)
        if picked:
            packet.sections["research_findings"] = [self._render(f) for f in picked]
            packet.provenance.append({
                "section": "research_findings",
                "why": "earlier research findings with competitive bearing, carried "
                       "forward from working memory",
            })
            self._track(packet, picked)

        withheld = len(research) - len(picked)
        if withheld > 0:
            packet.omitted.append({
                "section": "research_findings",
                "why": f"{withheld} research finding(s) withheld — no tracked company or "
                       f"market signal, so not relevant to this objective",
            })

        if reason:
            packet.sections["follow_up_reason"] = reason
            packet.provenance.append(
                {"section": "follow_up_reason", "why": "why this check was requested"}
            )

    # ── Orchestrator ────────────────────────────────────────
    def _orchestrator(self, packet: AgentContextPacket, memory: WorkingMemory) -> None:
        """The orchestrator sees the accumulated picture — it has to decide next steps."""
        ctx = memory.task_context
        if ctx is not None:
            if ctx.topics:
                packet.sections["topics"] = ctx.topics[:6]
            if ctx.competitors:
                packet.sections["competitors"] = ctx.competitors[:6]
        if memory.plan_steps:
            packet.sections["plan_state"] = [
                {"step": s.step_name, "status": s.status} for s in memory.plan_steps
            ]
        if memory.agent_status:
            packet.sections["agent_status"] = dict(memory.agent_status)

        ranked = memory.ranked_facts()[:8]
        if ranked:
            packet.sections["research_findings"] = [
                self._render(f) for f in ranked if f.source_agent == RESEARCH_AGENT.key
            ][:4]
            packet.sections["competitive_findings"] = [
                self._render(f) for f in ranked if f.source_agent == COMPETITIVE_AGENT.key
            ][:4]
            # Drop the empty ones so `included` reflects what is actually present.
            for key in ("research_findings", "competitive_findings"):
                if not packet.sections.get(key):
                    packet.sections.pop(key, None)
            self._track(packet, ranked)
        if memory.coverage_gaps:
            packet.sections["coverage_gaps"] = memory.coverage_gaps[:5]
        if memory.pending_questions:
            packet.sections["pending_questions"] = memory.pending_questions[:5]
        if memory.narrative_summary:
            packet.sections["run_summary"] = memory.narrative_summary[:600]

        for key in packet.sections:
            if not any(p["section"] == key for p in packet.provenance):
                packet.provenance.append(
                    {"section": key, "why": "accumulated run context the orchestrator reasons over"}
                )

        if len(memory.facts) > len(ranked):
            packet.omitted.append({
                "section": "facts",
                "why": f"{len(memory.facts) - len(ranked)} lower-ranked fact(s) withheld — "
                       f"only the top {len(ranked)} are carried into the decision",
            })

    # ── shared helpers ──────────────────────────────────────
    def _add_prior_context(self, packet: AgentContextPacket, memory: WorkingMemory) -> None:
        """Attach retrieved long-term memory, filtered to this agent's concern."""
        if not memory.retrieved_memories:
            return
        wanted = self._memory_types_for(packet.target_agent)
        relevant = [
            m for m in memory.retrieved_memories
            if not wanted or m.get("memory_type") in wanted
        ]
        if not relevant:
            packet.omitted.append({
                "section": "prior_context",
                "why": f"{len(memory.retrieved_memories)} retrieved memory item(s) withheld — "
                       f"none of a type this agent acts on",
            })
            return

        picked = relevant[:MAX_MEMORIES_PER_AGENT]
        packet.sections["prior_context"] = [
            {
                "type": m.get("type_label") or m.get("memory_type"),
                "summary": (m.get("summary") or m.get("content") or "")[:MAX_FACT_CHARS],
                "from_run": m.get("source_run_id", ""),
            }
            for m in picked
        ]
        packet.provenance.append({
            "section": "prior_context",
            "why": "relevant context retrieved from previous monitoring runs",
        })
        for m in picked:
            mid = m.get("memory_id")
            if mid and mid not in packet.memory_ids:
                packet.memory_ids.append(mid)
        if len(relevant) > len(picked):
            packet.omitted.append({
                "section": "prior_context",
                "why": f"{len(relevant) - len(picked)} lower-ranked memory item(s) withheld",
            })

    @staticmethod
    def _memory_types_for(agent: str) -> set[str]:
        if agent == RESEARCH_AGENT.key:
            return {"RESEARCH_CONTEXT", "TRACKED_TOPIC", "IMPORTANT_FINDING",
                    "HISTORICAL_BASELINE", "UNRESOLVED_QUESTION"}
        if agent == COMPETITIVE_AGENT.key:
            return {"COMPETITIVE_CONTEXT", "TRACKED_COMPETITOR", "IMPORTANT_FINDING",
                    "HISTORICAL_BASELINE"}
        return set()  # orchestrator: no filter

    def _pick(
        self,
        facts: list[MemoryFact],
        objective: str,
        memory: WorkingMemory,
        *,
        limit: int,
    ) -> list[MemoryFact]:
        """Rank facts against this objective and keep the few that clear the floor."""
        ctx = memory.task_context
        objective_terms = _tokens(objective)
        topic_terms = _tokens(" ".join(ctx.topics + ctx.research_topics)) if ctx else set()
        companies = {c.lower() for c in (ctx.competitors if ctx else [])}

        scored: list[tuple[float, MemoryFact]] = []
        for fact in facts:
            score = self._relevance(fact, objective_terms, topic_terms, companies)
            if score >= RELEVANCE_FLOOR:
                scored.append((score, fact))
        scored.sort(key=lambda pair: (pair[0], pair[1].rank), reverse=True)
        return [fact for _, fact in scored[:limit]]

    @staticmethod
    def _relevance(
        fact: MemoryFact,
        objective_terms: set[str],
        topic_terms: set[str],
        companies: set[str],
    ) -> float:
        terms = fact.match_terms()
        if not terms:
            return 0.0
        score = 0.0
        if objective_terms:
            score += 0.40 * (len(objective_terms & terms) / max(1, len(objective_terms)))
        if topic_terms:
            score += 0.25 * (len(topic_terms & terms) / max(1, len(topic_terms)))
        if companies and {c.lower() for c in fact.competitors} & companies:
            score += 0.20
        if set(fact.signals) & MARKET_SIGNALS:
            score += 0.08
        score += {0: 0.0, 1: 0.04, 2: 0.10, 3: 0.15}.get(fact.rank, 0.0)
        score += 0.10 * min(1.0, fact.relevance)
        return round(min(1.0, score), 3)

    @staticmethod
    def _competitively_relevant(fact: MemoryFact) -> bool:
        return bool(fact.competitors or (set(fact.signals) & MARKET_SIGNALS))

    @staticmethod
    def _render(fact: MemoryFact) -> dict[str, Any]:
        """Compact projection of a fact. Never the raw provider payload."""
        return {
            "id": fact.id,
            "text": fact.text[:MAX_FACT_CHARS],
            "summary": fact.summary[:MAX_FACT_CHARS],
            "importance": fact.importance,
            "signals": fact.signals[:4],
            "competitors": fact.competitors[:4],
            "from_agent": fact.source_agent,
            "url": fact.url,
            "simulated": fact.simulated,
        }

    @staticmethod
    def _track(packet: AgentContextPacket, facts: list[MemoryFact]) -> None:
        for fact in facts:
            if fact.id not in packet.fact_ids:
                packet.fact_ids.append(fact.id)
            if fact.source_agent and fact.source_agent not in packet.source_agents:
                packet.source_agents.append(fact.source_agent)

    def _focus_terms(self, packet: AgentContextPacket, memory: WorkingMemory) -> list[str]:
        """Search terms implied by the shared context.

        Preference order matters: terms drawn from the findings that were actually
        shared come first, because those are what make a follow-up query different
        from the original one. Task-context topics are the fallback.
        """
        terms: list[str] = []

        def add(value: str) -> None:
            clean = (value or "").strip()
            if clean and clean.lower() not in {t.lower() for t in terms}:
                terms.append(clean)

        shared = [f for f in memory.facts if f.id in packet.fact_ids]
        for fact in shared:
            for value in fact.entities[:2]:
                add(value)
            for value in fact.topics[:2]:
                add(value)

        ctx = memory.task_context
        if ctx is not None:
            for value in (ctx.research_topics if packet.target_agent == RESEARCH_AGENT.key
                          else ctx.topics):
                add(value)
        return terms[:4]


__all__ = ["AgentContextPacket", "ContextBuilder", "SECTION_LABELS", "ORCHESTRATOR"]
