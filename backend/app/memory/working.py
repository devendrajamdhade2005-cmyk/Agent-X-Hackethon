"""Short-term (working) memory for a single intelligence run.

One instance per run, identified by `run_id`, so nothing leaks between runs. It is
written after every meaningful step and read by the orchestrator before every
decision — that read-after-write is the whole point, not the storage.

What it holds is deliberately *not* the raw tool output. Raw provider responses stay
in `AgentState.findings`; what lands here is the structured residue that a later step
could actually act on: which findings mattered, what each agent concluded, what is
still missing. That distinction is what keeps context from growing without bound.

Compression follows the same principle: recent detail stays verbatim, older verbose
detail folds into a narrative summary, and structured facts above a importance floor
are never folded away. Anything already summarised is never summarised again, so the
summary cannot drift.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .task_context import TaskContext

# ── importance ──────────────────────────────────────────────
LOW, MEDIUM, HIGH, CRITICAL = "LOW", "MEDIUM", "HIGH", "CRITICAL"
IMPORTANCE_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}

# Facts at or above this rank survive compression and are eligible for long-term
# memory. MEDIUM and below are run-local detail.
KEEP_FLOOR = IMPORTANCE_RANK[HIGH]

# Compression thresholds. Chosen so a normal 2-agent run never compresses (it
# produces well under 24 facts) and only genuinely long runs pay the cost.
FACT_SOFT_LIMIT = 24
FACTS_KEPT_VERBATIM = 12

_WORD = re.compile(r"[a-z0-9\-+]+")

STEP_PENDING = "pending"
STEP_IN_PROGRESS = "in_progress"
STEP_COMPLETED = "completed"
STEP_SKIPPED = "skipped"
STEP_FAILED = "failed"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2}


@dataclass
class MemoryFact:
    """One thing worth remembering. Structured, not prose."""

    id: str
    kind: str  # finding | trend | signal | decision | gap | question | baseline
    text: str
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    importance: str = MEDIUM
    source_agent: str = ""
    finding_ids: list[str] = field(default_factory=list)
    url: str = ""
    relevance: float = 0.0
    simulated: bool = False
    created_at: str = field(default_factory=_now)
    version: int = 0

    @property
    def rank(self) -> int:
        return IMPORTANCE_RANK.get(self.importance, 1)

    def match_terms(self) -> set[str]:
        """Everything this fact can be matched on."""
        blob = " ".join([self.text, self.summary, *self.topics, *self.entities,
                         *self.competitors, *self.signals])
        return _tokens(blob)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "summary": self.summary,
            "topics": self.topics,
            "entities": self.entities,
            "competitors": self.competitors,
            "signals": self.signals,
            "importance": self.importance,
            "source_agent": self.source_agent,
            "finding_ids": self.finding_ids,
            "url": self.url,
            "relevance": round(self.relevance, 3),
            "simulated": self.simulated,
            "created_at": self.created_at,
            "version": self.version,
        }


@dataclass
class PlanStepState:
    """One line of the execution plan, with live status.

    Holds a *safe* reason summary only — what was decided and why, in the same
    register as the activity log. No prompts, no private reasoning.
    """

    step_id: str
    step_name: str
    agent_type: str = ""
    objective: str = ""
    status: str = STEP_PENDING
    depends_on: list[str] = field(default_factory=list)
    result_reference: str = ""
    safe_reason_summary: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "agent_type": self.agent_type,
            "objective": self.objective,
            "status": self.status,
            "depends_on": self.depends_on,
            "result_reference": self.result_reference,
            "safe_reason_summary": self.safe_reason_summary,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class WorkingMemory:
    """Everything the current run knows, in a form later steps can use."""

    run_id: str = ""
    task_context: TaskContext | None = None
    plan_steps: list[PlanStepState] = field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    facts: list[MemoryFact] = field(default_factory=list)
    decisions: list[dict[str, str]] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    agent_status: dict[str, str] = field(default_factory=dict)
    # Prose summary of context that has been compressed away.
    narrative_summary: str = ""
    # How many facts have already been folded into `narrative_summary`. Guarantees
    # nothing is summarised twice, which is what causes summary drift.
    compressed_count: int = 0
    compressions: int = 0
    version: int = 0
    updated_at: str = field(default_factory=_now)
    # Ordered record of *when* memory changed, for the UI progression view.
    timeline: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ── lifecycle ───────────────────────────────────────────
    def bump(self, event: str, detail: str = "") -> int:
        self.version += 1
        self.updated_at = _now()
        self.timeline.append({
            "version": self.version,
            "event": event,
            "detail": detail[:200],
            "facts": len(self.facts),
            "at": self.updated_at,
        })
        return self.version

    def set_task_context(self, ctx: TaskContext) -> None:
        self.task_context = ctx
        self.run_id = ctx.run_id or self.run_id
        self.bump("task_context_captured", ctx.headline())

    def set_plan(self, steps: list[PlanStepState]) -> None:
        self.plan_steps = steps
        self.bump(
            "plan_stored",
            f"{len(steps)} step(s): " + ", ".join(s.step_name for s in steps[:4]),
        )

    def step(self, step_id: str) -> PlanStepState | None:
        return next((s for s in self.plan_steps if s.step_id == step_id), None)

    def steps_for(self, agent_type: str) -> list[PlanStepState]:
        return [s for s in self.plan_steps if s.agent_type == agent_type]

    def mark_step(
        self,
        step_id: str,
        status: str,
        *,
        result_reference: str = "",
        reason: str = "",
    ) -> PlanStepState | None:
        step = self.step(step_id)
        if step is None:
            return None
        step.status = status
        if result_reference:
            step.result_reference = result_reference
        if reason:
            step.safe_reason_summary = reason[:240]
        if status == STEP_IN_PROGRESS and not step.started_at:
            step.started_at = _now()
        if status in {STEP_COMPLETED, STEP_SKIPPED, STEP_FAILED}:
            step.finished_at = _now()
        self.bump("plan_state_updated", f"{step.step_name} → {status}")
        return step

    def plan_progress(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.plan_steps:
            counts[step.status] = counts.get(step.status, 0) + 1
        return counts

    def current_step(self) -> PlanStepState | None:
        for step in self.plan_steps:
            if step.status == STEP_IN_PROGRESS:
                return step
        return next((s for s in self.plan_steps if s.status == STEP_PENDING), None)

    # ── writes ──────────────────────────────────────────────
    def add_facts(self, facts: list[MemoryFact], *, event: str = "memory_updated") -> int:
        """Add facts, skipping ones already held. Returns how many were new.

        Dedup is on the fact id, which callers derive from stable content — the same
        finding arriving twice must not appear as two memories.
        """
        known = {f.id for f in self.facts}
        added = 0
        for fact in facts:
            if fact.id in known:
                continue
            fact.version = self.version + 1
            self.facts.append(fact)
            known.add(fact.id)
            added += 1
        if added:
            self.bump(event, f"{added} fact(s) retained")
            self.compress_if_needed()
        return added

    def add_decision(self, summary: str, *, by: str = "orchestrator", detail: str = "") -> None:
        """Record a safe decision summary — what was decided, never how."""
        entry = {"by": by, "summary": summary[:240], "detail": detail[:400], "at": _now()}
        self.decisions.append(entry)
        self.bump("decision_recorded", summary)

    def note_gap(self, gap: str) -> None:
        if gap and gap not in self.coverage_gaps:
            self.coverage_gaps.append(gap)
            self.bump("gap_noted", gap)

    def resolve_gap(self, gap: str) -> None:
        if gap in self.coverage_gaps:
            self.coverage_gaps.remove(gap)
            self.bump("gap_closed", gap)

    def add_question(self, question: str) -> None:
        if question and question not in self.pending_questions:
            self.pending_questions.append(question)
            self.bump("question_raised", question)

    def set_agent_status(self, agent: str, status: str) -> None:
        if self.agent_status.get(agent) != status:
            self.agent_status[agent] = status
            self.bump("agent_status", f"{agent} → {status}")

    def note(self, text: str) -> None:
        """A safe operational note (e.g. a degraded memory subsystem)."""
        if text and text not in self.notes:
            self.notes.append(text[:200])

    # ── reads ───────────────────────────────────────────────
    def facts_by_agent(self, agent: str) -> list[MemoryFact]:
        return [f for f in self.facts if f.source_agent == agent]

    def facts_by_kind(self, kind: str) -> list[MemoryFact]:
        return [f for f in self.facts if f.kind == kind]

    def important_facts(self, floor: int = KEEP_FLOOR) -> list[MemoryFact]:
        return [f for f in self.facts if f.rank >= floor]

    def ranked_facts(self) -> list[MemoryFact]:
        """Most useful first: importance, then relevance, then recency."""
        return sorted(
            self.facts,
            key=lambda f: (f.rank, f.relevance, f.created_at),
            reverse=True,
        )

    def competitive_relevance(self) -> list[MemoryFact]:
        """Facts that justify sending work to the competitive agent.

        A research finding is competitively relevant when it names a tracked company
        or carries a market signal — that is a property of the stored fact, so the
        orchestrator reads it from memory rather than re-deriving it.
        """
        market = {"launch", "funding", "acquisition", "partnership", "regulatory",
                  "hiring", "benchmark", "patent"}
        out: list[MemoryFact] = []
        for fact in self.facts:
            if fact.kind == "competitive_relevance":
                out.append(fact)
                continue
            if fact.kind not in {"finding", "trend"}:
                continue
            if fact.competitors or (set(fact.signals) & market):
                out.append(fact)
        return sorted(out, key=lambda f: (f.rank, f.relevance), reverse=True)

    def focus_terms(self, limit: int = 4) -> list[str]:
        """Concrete phrases a later agent should search for.

        Drawn from what has actually been found, so a follow-up query is shaped by
        earlier results rather than only by the original goal.
        """
        terms: list[str] = []
        for fact in self.ranked_facts():
            for value in [*fact.topics, *fact.entities]:
                clean = value.strip()
                if clean and clean.lower() not in {t.lower() for t in terms}:
                    terms.append(clean)
                if len(terms) >= limit:
                    return terms
        return terms

    # ── compression ─────────────────────────────────────────
    def compress_if_needed(self) -> bool:
        """Fold older verbose facts into a summary once the run gets long.

        Structured facts at or above `KEEP_FLOOR` are never folded: the requirement
        is to control context size without losing important facts, so the summary
        replaces *detail*, not evidence.
        """
        if len(self.facts) <= FACT_SOFT_LIMIT:
            return False

        keep_recent = self.facts[-FACTS_KEPT_VERBATIM:]
        older = self.facts[:-FACTS_KEPT_VERBATIM]
        # Anything important stays as a first-class fact regardless of age.
        retained = [f for f in older if f.rank >= KEEP_FLOOR]
        folded = [f for f in older if f.rank < KEEP_FLOOR]
        if not folded:
            return False

        self.narrative_summary = self._extend_summary(folded)
        self.compressed_count += len(folded)
        self.compressions += 1
        self.facts = [*retained, *keep_recent]
        self.bump(
            "context_compressed",
            f"{len(folded)} lower-importance fact(s) summarised; "
            f"{len(retained)} important fact(s) kept verbatim",
        )
        return True

    def _extend_summary(self, folded: list[MemoryFact]) -> str:
        """Append a clause describing the folded facts.

        Deterministic and additive: the existing summary text is never rewritten, so
        repeated compression cannot degrade what earlier passes recorded.
        """
        agents: dict[str, int] = {}
        signals: set[str] = set()
        companies: set[str] = set()
        for fact in folded:
            if fact.source_agent:
                agents[fact.source_agent] = agents.get(fact.source_agent, 0) + 1
            signals.update(fact.signals)
            companies.update(fact.competitors)

        bits = [f"{n} finding(s) from {agent.replace('_', ' ')}"
                for agent, n in sorted(agents.items())]
        clause = f"Earlier in this run: {'; '.join(bits) or f'{len(folded)} finding(s)'}"
        if companies:
            clause += f"; companies seen: {', '.join(sorted(companies)[:5])}"
        if signals:
            clause += f"; signals: {', '.join(sorted(signals)[:5])}"
        clause += "."
        return f"{self.narrative_summary} {clause}".strip()

    # ── serialisation ───────────────────────────────────────
    def public(self) -> dict[str, Any]:
        """Safe projection for the API, UI and report.

        Facts are capped and summarised; no prompts or raw provider payloads.
        """
        return {
            "run_id": self.run_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "task_context": self.task_context.public() if self.task_context else None,
            "plan_steps": [s.to_dict() for s in self.plan_steps],
            "plan_progress": self.plan_progress(),
            "facts": [f.to_dict() for f in self.ranked_facts()[:20]],
            "fact_count": len(self.facts),
            "important_fact_count": len(self.important_facts()),
            "decisions": self.decisions[-10:],
            "coverage_gaps": self.coverage_gaps,
            "pending_questions": self.pending_questions,
            "agent_status": self.agent_status,
            "retrieved_memories": self.retrieved_memories,
            "retrieved_count": len(self.retrieved_memories),
            "narrative_summary": self.narrative_summary,
            "compressed_count": self.compressed_count,
            "compressions": self.compressions,
            "timeline": self.timeline,
            "notes": self.notes,
            "focus_terms": self.focus_terms(),
        }


def fact_id(*parts: str) -> str:
    """Stable id from content, so the same thing never lands twice."""
    raw = "|".join(p.strip().lower() for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]  # noqa: S324 — not security
