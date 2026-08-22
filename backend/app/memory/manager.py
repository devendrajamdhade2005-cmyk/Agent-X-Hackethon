"""MemoryManager — the only memory surface the agents talk to.

The orchestrator should decide *what happens next*, not manage persistence, scoring
and compression, so all of that lives here. It owns:

  begin_run   → task context + working memory + relevant long-term retrieval
  record_plan → execution plan as tracked, status-bearing steps
  context_for → a scoped context packet for one agent and one objective
  record_agent_report → important findings become memory; statuses and gaps update
  compare_with_baseline → NEW / PREVIOUSLY KNOWN / TREND ACCELERATING, evidence-only
  consolidate → select what deserves to outlive the run, and persist it

Failure policy is uniform and deliberate: **memory never breaks a run.** Retrieval
failure leaves the run on current context. Persistence failure leaves the completed
intelligence intact. Both are reported honestly instead of being hidden, and neither
is ever papered over with invented memories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..agents.llm import LLMClient
from ..agents.messages import COMPETITIVE_AGENT, ORCHESTRATOR, RESEARCH_AGENT
from .context_builder import AgentContextPacket, ContextBuilder
from .long_term import (
    COMPETITIVE_CONTEXT,
    HISTORICAL_BASELINE,
    IMPORTANT_FINDING,
    RESEARCH_CONTEXT,
    RUN_SUMMARY,
    TRACKED_COMPETITOR,
    TRACKED_TOPIC,
    UNRESOLVED_QUESTION,
    LongTermMemoryItem,
    LongTermStore,
    long_term_store,
)
from .task_context import TaskContext, TaskContextExtractor
from .working import (
    CRITICAL,
    HIGH,
    LOW,
    MEDIUM,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_IN_PROGRESS,
    STEP_SKIPPED,
    MemoryFact,
    PlanStepState,
    WorkingMemory,
    fact_id,
)

_WORD = re.compile(r"[a-z0-9\-+]+")

# Findings at or above this importance become memory facts. Everything else stays
# in `AgentState.findings` as raw evidence and never enters reusable context.
FACT_FLOOR = MEDIUM

MARKET_SIGNALS = {"launch", "funding", "acquisition", "partnership", "regulatory",
                  "hiring", "benchmark", "patent"}
STRATEGIC_SIGNALS = {"acquisition", "funding", "regulatory"}

# Change classifications
NEW = "NEW"
PREVIOUSLY_KNOWN = "PREVIOUSLY KNOWN"
TREND_ACCELERATING = "TREND ACCELERATING"


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2}


def _fingerprint(title: str) -> str:
    """Stable-ish identity for a development, used for historical comparison."""
    toks = sorted(t for t in _tokens(title) if len(t) > 3)[:8]
    return " ".join(toks)


@dataclass
class ChangeReport:
    """Historical comparison result. Empty unless a real baseline was retrieved."""

    compared: bool = False
    baseline_run_id: str = ""
    baseline_at: str = ""
    new_count: int = 0
    known_count: int = 0
    verdict: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compared": self.compared,
            "baseline_run_id": self.baseline_run_id,
            "baseline_at": self.baseline_at,
            "new_count": self.new_count,
            "known_count": self.known_count,
            "verdict": self.verdict,
            "detail": self.detail,
        }


class MemoryManager:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        logger: Any | None = None,
        store: LongTermStore | None = None,
    ) -> None:
        self.llm = llm
        self.logger = logger
        self.store = store if store is not None else long_term_store
        self.extractor = TaskContextExtractor(llm)
        self.builder = ContextBuilder()
        self.change = ChangeReport()
        self.consolidation: dict[str, Any] = {}
        self.retrieval_status = "not attempted"

    # ── safe logging ────────────────────────────────────────
    def _emit(self, method: str, title: str, detail: str = "", **data: Any) -> None:
        """Log through the host logger if it supports the call.

        Uses `getattr` rather than assuming the interface: the orchestrator is also
        driven by hand-rolled logger doubles in tests, and a logging call must never
        be the reason a run — or a test — fails.
        """
        logger = self.logger
        if logger is None:
            return
        # Bound methods are created fresh on each attribute access, so identity
        # comparison between two `getattr` results is always False. Resolve once.
        shorthand = getattr(logger, method, None)
        try:
            if callable(shorthand):
                shorthand(title, detail, **data)
                return
            generic = getattr(logger, "log", None)
            if callable(generic):
                generic(method, title, detail, **data)
        except Exception:  # noqa: BLE001 — logging is never load-bearing
            pass

    # ─────────────────────────────────────────────────────────
    # 1. UNDERSTAND
    # ─────────────────────────────────────────────────────────
    async def begin_run(
        self,
        *,
        run_id: str,
        goal: str,
        topics: list[str] | None = None,
        keywords: list[str] | None = None,
        competitors: list[str] | None = None,
        required_domains: list[str] | None = None,
        optional_domains: list[str] | None = None,
    ) -> WorkingMemory:
        memory = WorkingMemory(run_id=run_id)
        try:
            ctx = await self.extractor.extract(
                run_id=run_id,
                goal=goal,
                topics=topics,
                keywords=keywords,
                competitors=competitors,
                required_domains=required_domains,
                optional_domains=optional_domains,
            )
        except Exception as exc:  # noqa: BLE001 — never block a run on extraction
            ctx = TaskContext(run_id=run_id, user_goal=goal, author="fallback")
            memory.note(f"task-context extraction degraded ({type(exc).__name__})")

        memory.set_task_context(ctx)
        self._emit(
            "context",
            "Task context extracted",
            ctx.headline(),
            topics=ctx.topics,
            competitors=ctx.competitors,
            domains=ctx.requested_domains,
            time_scope=ctx.time_scope,
            constraints=ctx.constraints,
            continuation=ctx.continuation,
            author=ctx.author,
            memory_version=memory.version,
        )
        self.retrieve(memory)
        return memory

    # ─────────────────────────────────────────────────────────
    # 2. RETRIEVE relevant long-term memory
    # ─────────────────────────────────────────────────────────
    def retrieve(self, memory: WorkingMemory) -> list[dict[str, Any]]:
        ctx = memory.task_context
        if ctx is None:
            return []

        terms = ctx.retrieval_terms()
        # A continuation goal ("continue monitoring this") carries no subject, so the
        # stored monitoring preferences are the only way to know what "this" means.
        if not terms and not ctx.continuation:
            self.retrieval_status = "no terms to search"
            self._emit(
                "retrieval",
                "No relevant previous context found",
                "This goal has no topics or companies to match against stored memory. "
                "Starting with current task context.",
                retrieved=0,
            )
            return []

        try:
            if ctx.subjectless:
                # Nothing to match on: fall back to the most important recent
                # monitoring memories so a bare "continue monitoring" still works.
                matches = self._continuation_matches(exclude_run_id=memory.run_id)
            else:
                matches = self.store.search(
                    terms=terms,
                    topics=ctx.topics + ctx.research_topics,
                    competitors=ctx.competitors,
                    limit=5,
                    exclude_run_id=memory.run_id,
                )
        except Exception as exc:  # noqa: BLE001 — retrieval must not end the run
            self.retrieval_status = f"failed: {type(exc).__name__}"
            memory.note("long-term memory unavailable for this run")
            self._emit(
                "retrieval",
                "Previous context unavailable. Continuing with current task context.",
                f"The memory store could not be read ({type(exc).__name__}). "
                f"This run proceeds on its own task context.",
                retrieved=0,
                degraded=True,
            )
            return []

        if not matches:
            self.retrieval_status = "no relevant memory"
            self._emit(
                "retrieval",
                "No relevant previous context found",
                "Nothing in long-term memory is related to this goal. "
                "Starting with current task context.",
                retrieved=0,
            )
            return []

        payload: list[dict[str, Any]] = []
        for item, score in matches:
            record = item.public()
            record["relevance"] = score
            payload.append(record)
        memory.retrieved_memories = payload
        memory.bump("long_term_retrieved", f"{len(payload)} item(s)")
        self.retrieval_status = f"{len(payload)} item(s) retrieved"

        try:
            self.store.mark_accessed([m["memory_id"] for m in payload])
        except Exception:  # noqa: BLE001 — bookkeeping only
            pass

        # A continuation run may legitimately inherit its subject from memory. A
        # normal run must not: its own goal governs, or the earlier topic would
        # quietly take over a new task.
        if ctx.subjectless:
            self._restore_subject(memory, payload)

        self._emit(
            "retrieval",
            f"{len(payload)} relevant memory item(s) retrieved",
            "; ".join(
                f"{m['type_label']}: {(m['summary'] or m['content'])[:70]}" for m in payload[:3]
            ),
            retrieved=len(payload),
            types=sorted({m["memory_type"] for m in payload}),
            memory_version=memory.version,
        )
        return payload

    def _continuation_matches(
        self, *, exclude_run_id: str
    ) -> list[tuple[LongTermMemoryItem, float]]:
        """Best stored monitoring context when the goal names no subject.

        Delegates the ordering to the store, which puts the memories that define
        *what* is being monitored ahead of those describing past runs.
        """
        items = self.store.continuation_items(exclude_run_id=exclude_run_id, limit=6)
        return [(item, 0.50) for item in items]

    @staticmethod
    def _restore_subject(memory: WorkingMemory, payload: list[dict[str, Any]]) -> None:
        ctx = memory.task_context
        if ctx is None:
            return
        restored: list[str] = []
        for record in payload:
            if record["memory_type"] == TRACKED_TOPIC:
                for topic in record.get("topics") or [record["content"]]:
                    if topic and topic.lower() not in {t.lower() for t in ctx.topics}:
                        ctx.topics.append(topic)
                        restored.append(f"topic:{topic}")
            if record["memory_type"] == TRACKED_COMPETITOR:
                for company in record.get("competitors") or [record["content"]]:
                    if company and company.lower() not in {c.lower() for c in ctx.competitors}:
                        ctx.competitors.append(company)
                        restored.append(f"company:{company}")
        if restored:
            ctx.metadata["restored_from_memory"] = restored
            ctx.topics = ctx.topics[:6]
            ctx.competitors = ctx.competitors[:8]
            if not ctx.research_topics:
                ctx.research_topics = list(ctx.topics)
            memory.bump("subject_restored_from_memory", ", ".join(restored[:4]))

    # ─────────────────────────────────────────────────────────
    # 3. PLAN state
    # ─────────────────────────────────────────────────────────
    def record_plan(
        self, memory: WorkingMemory, plan_entries: list[dict[str, Any]]
    ) -> list[PlanStepState]:
        """Turn the orchestrator's agent selection into tracked plan steps.

        Two synthetic trailing steps — cross-agent analysis and final intelligence —
        make the plan reflect the whole run rather than only the delegations, which
        is what lets the UI show progression through to the end.
        """
        steps: list[PlanStepState] = []
        order = 0
        agent_step_ids: list[str] = []

        for entry in sorted(plan_entries, key=lambda e: e.get("order", 0)):
            selected = bool(entry.get("selected"))
            order += 1
            step_id = f"step-{order}"
            if selected:
                agent_step_ids.append(step_id)
            steps.append(PlanStepState(
                step_id=step_id,
                step_name=entry.get("name") or entry.get("agent", "agent"),
                agent_type=entry.get("agent", ""),
                objective=entry.get("reason", "")[:240],
                status="pending" if selected else STEP_SKIPPED,
                safe_reason_summary=entry.get("reason", "")[:240],
            ))

        order += 1
        steps.append(PlanStepState(
            step_id=f"step-{order}",
            step_name="Cross-Agent Analysis",
            agent_type=ORCHESTRATOR.key,
            objective="Cross-validate findings and consolidate the evidence.",
            depends_on=list(agent_step_ids),
        ))
        order += 1
        steps.append(PlanStepState(
            step_id=f"step-{order}",
            step_name="Final Intelligence",
            agent_type=ORCHESTRATOR.key,
            objective="Prioritize insights and write the briefing.",
            depends_on=[steps[-1].step_id],
        ))

        memory.set_plan(steps)
        self._emit(
            "memory",
            "Execution plan stored in working memory",
            "; ".join(f"{s.step_name}: {s.status}" for s in steps),
            steps=[s.to_dict() for s in steps],
            memory_version=memory.version,
        )
        return steps

    def start_step(self, memory: WorkingMemory, agent_type: str, objective: str = "") -> str:
        step = next(
            (s for s in memory.plan_steps
             if s.agent_type == agent_type and s.status == "pending"),
            None,
        )
        if step is None:
            return ""
        memory.mark_step(step.step_id, STEP_IN_PROGRESS, reason=objective)
        memory.set_agent_status(agent_type, "working")
        return step.step_id

    def finish_step(
        self, memory: WorkingMemory, step_id: str, *, status: str, reference: str = ""
    ) -> None:
        if step_id:
            memory.mark_step(step_id, status, result_reference=reference)

    # ─────────────────────────────────────────────────────────
    # 4. CONTEXT for an agent
    # ─────────────────────────────────────────────────────────
    def context_for(
        self,
        memory: WorkingMemory,
        *,
        target_agent: str,
        objective: str,
        kind: str = "primary",
        reason: str = "",
        trigger_facts: list[MemoryFact] | None = None,
    ) -> AgentContextPacket:
        packet = self.builder.build(
            target_agent=target_agent,
            objective=objective,
            memory=memory,
            kind=kind,
            reason=reason,
            trigger_facts=trigger_facts,
        )
        shared_from = [a for a in packet.source_agents if a != target_agent]
        detail = "Context received: " + ", ".join(packet.label_list())
        if shared_from:
            detail += (
                f" — including {len(packet.fact_ids)} finding(s) carried over from "
                f"{', '.join(a.replace('_', ' ') for a in shared_from)}"
            )
        self._emit(
            "context",
            f"{_name(target_agent)} received relevant task context",
            detail,
            agent=target_agent,
            included=packet.included,
            labels=packet.label_list(),
            omitted=packet.omitted,
            focus_terms=packet.focus_terms,
            shared_from=shared_from,
            fact_ids=packet.fact_ids,
            memory_version=packet.memory_version,
            size_chars=packet.size_chars,
        )
        return packet

    # ─────────────────────────────────────────────────────────
    # 5. RECORD what an agent produced
    # ─────────────────────────────────────────────────────────
    def record_agent_report(
        self,
        memory: WorkingMemory,
        *,
        report: Any,
        findings: list[Any],
        corroborated_ids: set[str] | None = None,
    ) -> int:
        """Fold an agent's result into working memory.

        Only findings that clear the importance floor become facts. Raw provider
        output stays in `AgentState.findings`; what lands here is the residue a later
        step could act on.
        """
        agent = getattr(report, "from_agent", "") or ""
        corroborated = corroborated_ids or set()
        facts: list[MemoryFact] = []

        for finding in findings:
            importance = self._importance(finding, corroborated=corroborated)
            if _rank(importance) < _rank(FACT_FLOOR):
                continue
            title = getattr(finding, "title", "") or ""
            facts.append(MemoryFact(
                id=fact_id("finding", title, getattr(finding, "url", "") or ""),
                kind="finding",
                text=title[:300],
                summary=(getattr(finding, "summary", "") or "")[:300],
                topics=self._topics_of(memory, finding),
                entities=[e for e in [getattr(finding, "competitor", "") or ""] if e],
                competitors=[c for c in [getattr(finding, "competitor", "") or ""] if c],
                signals=list(getattr(finding, "signals", []) or [])[:6],
                importance=importance,
                source_agent=agent,
                finding_ids=[getattr(finding, "id", "")],
                url=getattr(finding, "url", "") or "",
                relevance=float(getattr(finding, "relevance", 0.0) or 0.0),
                simulated=bool(getattr(finding, "simulated", False)),
            ))

        # Agent-level conclusions are memory too: a trend is reusable in a way an
        # individual paper is not.
        # Agent-level facts inherit the report's coverage, so a conclusion drawn
        # from simulated evidence is not laundered into looking real.
        synthetic = getattr(report, "coverage", "") == "simulated"

        for trend in (getattr(report, "research_trends", None) or [])[:3]:
            facts.append(MemoryFact(
                id=fact_id("trend", trend, agent),
                kind="trend",
                text=str(trend)[:200],
                summary=f"Recurring research theme reported by {_name(agent)}.",
                topics=[str(trend)[:80]],
                importance=HIGH,
                source_agent=agent,
                simulated=synthetic,
            ))
        for signal in (getattr(report, "market_signals", None) or [])[:4]:
            facts.append(MemoryFact(
                id=fact_id("signal", signal, agent),
                kind="signal",
                text=f"Market signal observed: {signal}",
                summary=f"{_name(agent)} observed a '{signal}' signal.",
                signals=[str(signal)[:40]],
                competitors=list(getattr(report, "competitors_analyzed", []) or [])[:4],
                importance=HIGH if str(signal) in STRATEGIC_SIGNALS else MEDIUM,
                source_agent=agent,
                simulated=synthetic,
            ))

        # The Research Agent's competitive-relevance judgement is itself memory: it
        # is what a later step reads to justify a follow-up. Stored as a fact so the
        # decision is traceable to stored evidence rather than to a live variable.
        if getattr(report, "potential_competitive_relevance", False):
            leads = list(getattr(report, "competitive_leads", []) or [])
            reason = getattr(report, "competitive_relevance_reason", "") or ""
            facts.append(MemoryFact(
                id=fact_id("relevance", agent, reason or "competitive-relevance"),
                kind="competitive_relevance",
                text=reason or "Research findings may have competitive relevance.",
                summary=reason,
                competitors=leads,
                entities=leads,
                signals=["competitive-relevance"],
                importance=HIGH,
                source_agent=agent,
            ))

        added = memory.add_facts(facts, event="agent_findings_recorded")
        memory.set_agent_status(agent, getattr(report, "status", "completed") or "completed")

        summary = getattr(report, "reasoning_summary", "") or ""
        if summary:
            memory.add_decision(
                f"{_name(agent)} reported: {summary[:160]}", by=agent
            )

        coverage = getattr(report, "coverage", "")
        if coverage in {"simulated", "unavailable", "partial"}:
            memory.note_gap(f"{_name(agent)} coverage was {coverage}")
        for question in (getattr(report, "recommended_next_step", "") or "")[:1] and [
            getattr(report, "recommended_next_step", "")
        ]:
            if question:
                memory.add_question(question[:200])

        self._emit(
            "memory",
            f"{_name(agent)} findings added to working memory",
            f"{added} new fact(s) retained from {len(findings)} finding(s); "
            f"working memory is now at version {memory.version}.",
            agent=agent,
            facts_added=added,
            fact_total=len(memory.facts),
            important=len(memory.important_facts()),
            memory_version=memory.version,
        )
        return added

    def record_decision(
        self, memory: WorkingMemory, summary: str, *, detail: str = "", by: str = ORCHESTRATOR.key
    ) -> None:
        memory.add_decision(summary, by=by, detail=detail)

    @staticmethod
    def competitive_relevance_flags(memory: WorkingMemory) -> list[MemoryFact]:
        """Stored research judgements that a competitive check is warranted."""
        return [f for f in memory.facts if f.kind == "competitive_relevance"]

    def observe_context(self, memory: WorkingMemory) -> AgentContextPacket:
        """The orchestrator's read of accumulated memory before deciding next."""
        packet = self.builder.build(
            target_agent=ORCHESTRATOR.key,
            objective="Decide whether further intelligence work is required.",
            memory=memory,
        )
        self._emit(
            "memory",
            "Orchestrator updated context after agent results",
            f"Reading working memory v{memory.version}: {len(memory.facts)} fact(s), "
            f"{len(memory.important_facts())} important, "
            f"{len(memory.coverage_gaps)} coverage gap(s).",
            agent=ORCHESTRATOR.key,
            memory_version=memory.version,
            facts=len(memory.facts),
            gaps=memory.coverage_gaps,
        )
        return packet

    # ── importance scoring ──────────────────────────────────
    @staticmethod
    def _importance(finding: Any, *, corroborated: set[str]) -> str:
        """Bounded, deterministic importance.

        Deliberately not an LLM call: this runs per finding, and spending a model
        round-trip on each one would be slow, costly and non-reproducible.
        """
        score = 0.0
        relevance = float(getattr(finding, "relevance", 0.0) or 0.0)
        if relevance >= 0.60:
            score += 2
        elif relevance >= 0.45:
            score += 1.5
        elif relevance >= 0.35:
            score += 1

        signals = set(getattr(finding, "signals", []) or [])
        if signals & STRATEGIC_SIGNALS:
            score += 1.5
        elif signals & MARKET_SIGNALS:
            score += 1

        if getattr(finding, "competitor", ""):
            score += 1.5

        credibility = getattr(finding, "credibility", "standard")
        score += {"high": 1.0, "standard": 0.0, "low": -1.0, "unverified": -1.5}.get(
            credibility, 0.0
        )

        if getattr(finding, "id", "") in corroborated:
            score += 1.5
        if not getattr(finding, "simulated", False):
            # Evidence strength is rewarded rather than synthetic data punished. A
            # penalty large enough to keep placeholders out of memory would also
            # empty working memory on a keyless run, and the guarantee that matters
            # — simulated data never becomes durable history — is enforced
            # separately and absolutely in `consolidate`.
            score += 1.0

        if score >= 5.0:
            return CRITICAL
        if score >= 3.5:
            return HIGH
        if score >= 2.0:
            return MEDIUM
        return LOW

    @staticmethod
    def _topics_of(memory: WorkingMemory, finding: Any) -> list[str]:
        ctx = memory.task_context
        if ctx is None:
            return []
        blob = _tokens(f"{getattr(finding, 'title', '')} {getattr(finding, 'summary', '')}")
        return [t for t in (ctx.topics + ctx.research_topics) if _tokens(t) & blob][:4]

    # ─────────────────────────────────────────────────────────
    # 6. HISTORICAL COMPARISON
    # ─────────────────────────────────────────────────────────
    def compare_with_baseline(self, memory: WorkingMemory) -> ChangeReport:
        """Classify current evidence against a real retrieved baseline.

        Returns an empty, `compared=False` report when there is no baseline. The
        requirement is explicit that historical change must never be fabricated.
        """
        ctx = memory.task_context
        if ctx is None:
            return self.change

        baseline = None
        try:
            baseline = self.store.baseline_for(
                terms=ctx.retrieval_terms(), competitors=ctx.competitors
            )
        except Exception:  # noqa: BLE001
            baseline = None
        if baseline is None:
            self.change = ChangeReport(compared=False)
            return self.change

        prior = baseline.metadata.get("fingerprints")
        if not isinstance(prior, list) or not prior:
            self.change = ChangeReport(compared=False)
            return self.change

        prior_set = {str(p) for p in prior}
        current = memory.important_facts(_rank(MEDIUM))
        new_items = [f for f in current if _fingerprint(f.text) not in prior_set]
        known = len(current) - len(new_items)

        prior_count = baseline.metadata.get("finding_count")
        verdict = NEW if new_items else PREVIOUSLY_KNOWN
        detail = (
            f"{len(new_items)} development(s) not seen in the previous run; "
            f"{known} already known."
        )
        if (
            isinstance(prior_count, int)
            and prior_count > 0
            and len(current) >= prior_count * 1.5
            and new_items
        ):
            verdict = TREND_ACCELERATING
            detail += (
                f" Volume rose from {prior_count} to {len(current)} important "
                f"finding(s), so the trend is accelerating."
            )

        self.change = ChangeReport(
            compared=True,
            baseline_run_id=baseline.source_run_id,
            baseline_at=baseline.created_at,
            new_count=len(new_items),
            known_count=known,
            verdict=verdict,
            detail=detail,
        )
        memory.bump("compared_with_baseline", f"{verdict}: {detail}")
        self._emit(
            "memory",
            f"Compared with previous monitoring: {verdict}",
            detail,
            agent=ORCHESTRATOR.key,
            baseline_run_id=baseline.source_run_id,
            new_count=len(new_items),
            known_count=known,
        )
        return self.change

    # ─────────────────────────────────────────────────────────
    # 7. CONSOLIDATE into long-term memory
    # ─────────────────────────────────────────────────────────
    def consolidate(self, memory: WorkingMemory, *, summary: str = "") -> dict[str, Any]:
        """Select what deserves to outlive this run and persist it.

        Selection is the point. Transient errors, duplicates, low-relevance results
        and simulated placeholders are not future context, so they are never offered
        to the store in the first place.
        """
        ctx = memory.task_context
        if ctx is None:
            self.consolidation = {"stored": 0, "reason": "no task context"}
            return self.consolidation

        candidates: list[LongTermMemoryItem] = []
        run_id = memory.run_id
        # One monitoring identity per subject, so the store does not accumulate a
        # near-identical baseline and summary on every run.
        scope = (
            ",".join(sorted(t.lower() for t in ctx.topics[:4]))
            or ctx.user_goal[:80].lower()
        )

        for topic in ctx.topics[:4]:
            candidates.append(LongTermMemoryItem(
                memory_id=fact_id("ltm", TRACKED_TOPIC, topic),
                memory_type=TRACKED_TOPIC,
                content=topic,
                summary=f"The user monitors '{topic}'.",
                topics=[topic],
                importance=HIGH,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
            ))
        for company in ctx.competitors[:6]:
            candidates.append(LongTermMemoryItem(
                memory_id=fact_id("ltm", TRACKED_COMPETITOR, company),
                memory_type=TRACKED_COMPETITOR,
                content=company,
                summary=f"The user tracks {company}.",
                topics=ctx.topics[:3],
                competitors=[company],
                importance=HIGH,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
            ))

        # Important, real findings only. Simulated data is excluded outright: it
        # would otherwise become a fake historical baseline for later runs.
        important = [
            f for f in memory.important_facts(_rank(HIGH))
            if not f.simulated and f.kind in {"finding", "trend", "signal"}
        ]
        for fact in important[:8]:
            memory_type = (
                RESEARCH_CONTEXT if fact.source_agent == RESEARCH_AGENT.key
                else COMPETITIVE_CONTEXT if fact.source_agent == COMPETITIVE_AGENT.key
                else IMPORTANT_FINDING
            )
            candidates.append(LongTermMemoryItem(
                memory_id=fact_id("ltm", memory_type, fact.text),
                memory_type=memory_type,
                content=fact.text,
                summary=fact.summary or fact.text,
                topics=fact.topics or ctx.topics[:3],
                entities=fact.entities,
                competitors=fact.competitors,
                importance=fact.importance,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
                url=fact.url,
                metadata={"signals": fact.signals, "from_agent": fact.source_agent},
            ))

        for question in memory.pending_questions[:2]:
            candidates.append(LongTermMemoryItem(
                memory_id=fact_id("ltm", UNRESOLVED_QUESTION, question),
                memory_type=UNRESOLVED_QUESTION,
                content=question,
                summary="Open follow-up carried over from a previous run.",
                topics=ctx.topics[:3],
                competitors=ctx.competitors[:3],
                importance=HIGH,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
            ))

        # The baseline is what makes the *next* run able to detect change.
        real_facts = [f for f in memory.important_facts(_rank(MEDIUM)) if not f.simulated]
        if real_facts:
            candidates.append(LongTermMemoryItem(
                dedup_scope=f"baseline::{scope}",
                memory_id=fact_id("ltm", HISTORICAL_BASELINE, scope),
                memory_type=HISTORICAL_BASELINE,
                content=f"Monitoring baseline for {', '.join(ctx.topics[:3]) or ctx.user_goal[:60]}",
                summary=(
                    f"{len(real_facts)} important finding(s) as of this run"
                    + (f"; companies: {', '.join(ctx.competitors[:3])}" if ctx.competitors else "")
                ),
                topics=ctx.topics[:4],
                competitors=ctx.competitors[:4],
                importance=HIGH,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
                metadata={
                    "finding_count": len(real_facts),
                    "fingerprints": [_fingerprint(f.text) for f in real_facts][:40],
                    "signals": sorted({s for f in real_facts for s in f.signals})[:8],
                },
            ))

        if summary:
            candidates.append(LongTermMemoryItem(
                dedup_scope=f"summary::{scope}",
                memory_id=fact_id("ltm", RUN_SUMMARY, scope),
                memory_type=RUN_SUMMARY,
                content=summary[:900],
                summary=summary[:300],
                topics=ctx.topics[:4],
                competitors=ctx.competitors[:4],
                importance=HIGH,
                source_run_id=run_id,
                source_goal=ctx.user_goal,
            ))

        try:
            outcome = self.store.save_many(candidates)
        except Exception as exc:  # noqa: BLE001 — a completed run must survive this
            self.consolidation = {
                "stored": 0,
                "offered": len(candidates),
                "persisted": False,
                "error": f"{type(exc).__name__}",
            }
            memory.note("long-term memory could not be written for this run")
            self._emit(
                "consolidation",
                "Memory consolidation failed — the completed intelligence is unaffected",
                f"Long-term memory could not be written ({type(exc).__name__}). "
                f"This run's findings, insights and report are intact.",
                degraded=True,
            )
            return self.consolidation

        outcome["offered"] = len(candidates)
        self.consolidation = outcome
        memory.bump(
            "memory_consolidated",
            f"{outcome['stored']} new, {outcome['refreshed']} refreshed",
        )
        if not outcome.get("persisted", True):
            memory.note("long-term memory held in process only — the store could not be written")
        self._emit(
            "consolidation",
            "Important run context consolidated for future monitoring",
            f"{outcome['stored']} new memory item(s), {outcome['refreshed']} refreshed, "
            f"{outcome['rejected']} rejected as not durable. "
            f"Store now holds {outcome['total']} item(s).",
            stored=outcome["stored"],
            refreshed=outcome["refreshed"],
            rejected=outcome["rejected"],
            total=outcome["total"],
            persisted=outcome.get("persisted", False),
            types=outcome.get("types", []),
        )
        return outcome

    # ── projection ──────────────────────────────────────────
    def public(self, memory: WorkingMemory | None) -> dict[str, Any]:
        """The `memory` block returned by the API and rendered by the UI."""
        if memory is None:
            return {"available": False}
        try:
            stats = self.store.stats()
        except Exception:  # noqa: BLE001
            stats = {"total": 0, "degraded": "unavailable"}
        return {
            "available": True,
            "working": memory.public(),
            "long_term": {
                "retrieved": memory.retrieved_memories,
                "retrieved_count": len(memory.retrieved_memories),
                "retrieval_status": self.retrieval_status,
                "consolidation": self.consolidation,
                "store": stats,
            },
            "change": self.change.to_dict(),
        }


def _rank(importance: str) -> int:
    return {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}.get(importance, 1)


def _name(agent_key: str) -> str:
    return {
        RESEARCH_AGENT.key: "Research Intelligence Agent",
        COMPETITIVE_AGENT.key: "Competitive Intelligence Agent",
        ORCHESTRATOR.key: "Intelligence Orchestrator",
    }.get(agent_key, agent_key.replace("_", " ").title() or "Agent")
