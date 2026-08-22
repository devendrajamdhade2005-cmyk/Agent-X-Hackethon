"""UNDERSTAND — turn a free-text goal into structured, reusable task context.

This is the first thing that happens in a run and the thing every later stage keys
off: the plan, the agent selection, the per-agent context packets and the long-term
memory lookup all read from `TaskContext`.

Two paths behind one call, matching `Planner`:
  * the LLM enriches time scope, constraints and entities when a key is configured;
  * a deterministic reader always produces a valid context on its own.

The deterministic path is not a stub — it is the guarantee. LLM output is treated as
untrusted: every field is coerced, length-capped and validated against the goal, and
anything malformed is dropped rather than propagated. A run must never fail, or
silently mis-scope itself, because a model returned bad JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..agents.llm import LLMClient, coerce_str_list
from ..agents.sanitize import sanitize

VALID_DOMAINS = ("research", "news", "web", "competitor", "patent")

# Human labels for the domains, used in safe log lines and the UI.
DOMAIN_LABELS = {
    "research": "Research",
    "patent": "Patents / IP",
    "news": "Industry News",
    "web": "Live Web",
    "competitor": "Competitive Intelligence",
}

# ── time scope lexicon ──────────────────────────────────────
# Ordered: the first match wins, so the tightest phrasing is preferred.
_TIME_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("today", ("today", "right now", "as of now")),
    ("this week", ("this week", "past week", "last week", "last 7 days")),
    ("this month", ("this month", "past month", "last month", "last 30 days")),
    ("this quarter", ("this quarter", "past quarter", "last quarter", "q1", "q2", "q3", "q4")),
    ("this year", ("this year", "past year", "last year", "last 12 months")),
    ("recent", ("recent", "recently", "latest", "current", "currently", "emerging",
                "up to date", "up-to-date", "new", "newest", "real time", "real-time")),
)
_YEAR = re.compile(r"\b(?:since|after|from)\s+(20\d{2})\b", re.I)

# ── constraint lexicon ──────────────────────────────────────
# Each entry is (regex, template). The captured group is folded into the template so
# the constraint reads as a sentence in the UI and the report.
_CONSTRAINT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bonly\s+([a-z0-9 ,\-]{3,60})", re.I), "restricted to {0}"),
    (re.compile(r"\bexclude\s+([a-z0-9 ,\-]{3,60})", re.I), "excluding {0}"),
    (re.compile(r"\bwithout\s+([a-z0-9 ,\-]{3,60})", re.I), "excluding {0}"),
    (re.compile(r"\bfocus(?:ed)?\s+on\s+([a-z0-9 ,\-]{3,60})", re.I), "focused on {0}"),
    (re.compile(r"\bignore\s+([a-z0-9 ,\-]{3,60})", re.I), "ignoring {0}"),
    (re.compile(r"\bno\s+(simulated|paywalled|social|forum)\b", re.I), "no {0} sources"),
)

# ── continuation lexicon ────────────────────────────────────
# "continue monitoring this" carries no subject of its own, so the subject has to
# come from memory. Detecting that is what makes continuation runs work.
_CONTINUATION_HINTS = (
    "continue monitoring", "continue tracking", "continue with", "continue this",
    "keep monitoring", "keep tracking", "keep watching", "same as last",
    "same as before", "as before", "like last time", "resume monitoring",
    "resume tracking", "carry on monitoring", "pick up where", "follow up on this",
    "any updates", "what's new", "whats new", "check again", "run it again",
    "update me", "since last time",
)
# A bare deictic goal ("continue this", "monitor that") has no subject at all.
_BARE_REFERENCE = re.compile(
    r"^\s*(?:please\s+)?(?:continue|resume|repeat|rerun|re-run|update|refresh|check)"
    r"\s*(?:monitoring|tracking|this|that|it|again|the same)?\s*[.!]?\s*$",
    re.I,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class TaskContext:
    """The structured reading of one goal. Immutable in spirit: set once per run."""

    run_id: str = ""
    user_goal: str = ""
    topics: list[str] = field(default_factory=list)
    research_topics: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    requested_domains: list[str] = field(default_factory=list)
    time_scope: str = "unspecified"
    constraints: list[str] = field(default_factory=list)
    # True when the goal refers back to earlier monitoring instead of naming a
    # subject. Long-term retrieval is allowed to supply the subject in that case.
    continuation: bool = False
    # True when the goal carries no subject of its own at all.
    subjectless: bool = False
    author: str = "heuristic"
    extracted_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── derived views ───────────────────────────────────────
    def retrieval_terms(self) -> list[str]:
        """Terms used to look for relevant long-term memory.

        Topics and competitors only — not the raw goal text, which carries verbs
        like "track" and "monitor" that match everything and rank nothing.
        """
        terms: list[str] = []
        for value in [*self.topics, *self.research_topics, *self.competitors, *self.entities]:
            low = value.strip().lower()
            if low and low not in terms:
                terms.append(low)
        return terms

    def headline(self) -> str:
        """One safe line for the activity log. No chain-of-thought, no prompts."""
        bits: list[str] = []
        if self.topics:
            bits.append(", ".join(self.topics[:3]))
        if self.competitors:
            bits.append(f"companies: {', '.join(self.competitors[:3])}")
        domains = [DOMAIN_LABELS.get(d, d) for d in self.requested_domains]
        if domains:
            bits.append(f"domains: {', '.join(domains)}")
        if self.time_scope and self.time_scope != "unspecified":
            bits.append(f"scope: {self.time_scope}")
        return " · ".join(bits) or "no explicit topic detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_goal": self.user_goal,
            "topics": self.topics,
            "research_topics": self.research_topics,
            "competitors": self.competitors,
            "entities": self.entities,
            "requested_domains": self.requested_domains,
            "time_scope": self.time_scope,
            "constraints": self.constraints,
            "continuation": self.continuation,
            "subjectless": self.subjectless,
            "author": self.author,
            "extracted_at": self.extracted_at,
            "metadata": self.metadata,
        }

    def public(self) -> dict[str, Any]:
        """What the UI and report may show. Same shape minus internals."""
        payload = self.to_dict()
        payload.pop("metadata", None)
        payload["domain_labels"] = [
            DOMAIN_LABELS.get(d, d) for d in self.requested_domains
        ]
        return payload


class TaskContextExtractor:
    """Builds `TaskContext`. Never raises; always returns a usable context."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    async def extract(
        self,
        *,
        run_id: str,
        goal: str,
        topics: list[str] | None = None,
        keywords: list[str] | None = None,
        competitors: list[str] | None = None,
        required_domains: list[str] | None = None,
        optional_domains: list[str] | None = None,
    ) -> TaskContext:
        goal_clean, _ = sanitize(goal, max_chars=1200)

        ctx = self._deterministic(
            run_id=run_id,
            goal=goal_clean,
            topics=coerce_str_list(topics, limit=8),
            keywords=coerce_str_list(keywords, limit=10),
            competitors=coerce_str_list(competitors, limit=8),
            required_domains=required_domains or [],
            optional_domains=optional_domains or [],
        )

        # The model only ever *adds* to a context that is already valid, and only
        # in fields where free text is genuinely better than a lexicon: entities,
        # constraints, time scope.
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                await self._enrich(ctx)
            except Exception:  # noqa: BLE001 — enrichment is best-effort by design
                ctx.metadata["enrichment"] = "failed"
        return ctx

    # ── deterministic path ──────────────────────────────────
    def _deterministic(
        self,
        *,
        run_id: str,
        goal: str,
        topics: list[str],
        keywords: list[str],
        competitors: list[str],
        required_domains: list[str],
        optional_domains: list[str],
    ) -> TaskContext:
        low = goal.lower()

        subject_terms = topics or keywords
        continuation = any(hint in low for hint in _CONTINUATION_HINTS)
        subjectless = bool(_BARE_REFERENCE.match(goal)) or (
            continuation and not subject_terms and not competitors
        )

        domains = [d for d in VALID_DOMAINS if d in set(required_domains)]
        research_topics = subject_terms[:6] if {"research", "patent"} & set(domains) else []

        return TaskContext(
            run_id=run_id,
            user_goal=goal,
            topics=subject_terms[:6],
            research_topics=research_topics,
            competitors=competitors[:8],
            entities=self._entities(competitors, subject_terms),
            requested_domains=domains,
            time_scope=self._time_scope(low),
            constraints=self._constraints(goal),
            continuation=continuation,
            subjectless=subjectless,
            author="heuristic",
            metadata={
                "optional_domains": [d for d in VALID_DOMAINS if d in set(optional_domains)],
                "keywords": keywords[:10],
            },
        )

    @staticmethod
    def _entities(competitors: list[str], topics: list[str]) -> list[str]:
        """Named things worth matching on. Companies first, then topic phrases."""
        out: list[str] = []
        for value in [*competitors, *topics]:
            clean = value.strip()
            if clean and clean.lower() not in {o.lower() for o in out}:
                out.append(clean)
        return out[:12]

    @staticmethod
    def _time_scope(low: str) -> str:
        year = _YEAR.search(low)
        if year:
            return f"since {year.group(1)}"
        for label, hints in _TIME_PATTERNS:
            if any(hint in low for hint in hints):
                return label
        return "unspecified"

    @staticmethod
    def _constraints(goal: str) -> list[str]:
        out: list[str] = []
        for pattern, template in _CONSTRAINT_PATTERNS:
            match = pattern.search(goal)
            if not match:
                continue
            captured = " ".join(match.group(1).split())[:60].rstrip(" ,.")
            if not captured:
                continue
            phrase = template.format(captured)
            if phrase not in out:
                out.append(phrase)
        return out[:4]

    # ── LLM enrichment ──────────────────────────────────────
    async def _enrich(self, ctx: TaskContext) -> None:
        assert self.llm is not None
        system = (
            "You extract structured task context for an autonomous "
            "competitive-intelligence agent. You do not perform research and you do "
            "not plan. Extract only what the goal actually says — never invent "
            "companies, topics or constraints that are not present.\n"
            "Reply with ONLY a JSON object."
        )
        user = (
            f"GOAL: {ctx.user_goal}\n"
            f"TOPICS ALREADY DETECTED: {ctx.topics or 'none'}\n"
            f"COMPANIES ALREADY DETECTED: {ctx.competitors or 'none'}\n\n"
            "Return JSON with this exact shape:\n"
            "{\n"
            '  "entities": ["named companies, products, technologies or standards"],\n'
            '  "constraints": ["explicit user restrictions, [] if none"],\n'
            '  "time_scope": "one short phrase, or \\"unspecified\\"",\n'
            '  "continuation": false\n'
            "}"
        )
        data = await self.llm.complete_json(
            purpose="task_context", system=system, user=user, max_tokens=600
        )
        if not isinstance(data, dict) or not data:
            ctx.metadata["enrichment"] = "no usable output"
            return

        added: list[str] = []

        # Entities: union, but only strings that actually occur in the goal. A model
        # that hallucinates a competitor must not be able to widen the run's scope.
        goal_low = ctx.user_goal.lower()
        for value in coerce_str_list(data.get("entities"), limit=12):
            clean = value.strip()
            if not clean or clean.lower() not in goal_low:
                continue
            if clean.lower() not in {e.lower() for e in ctx.entities}:
                ctx.entities.append(clean)
                added.append("entities")
        ctx.entities = ctx.entities[:12]

        for value in coerce_str_list(data.get("constraints"), limit=4):
            clean = value.strip()[:80]
            if clean and clean not in ctx.constraints:
                ctx.constraints.append(clean)
                added.append("constraints")
        ctx.constraints = ctx.constraints[:6]

        # Time scope: only fill a gap. A detected scope is evidence from the goal
        # text itself and outranks a model's paraphrase of it.
        if ctx.time_scope == "unspecified":
            scope = str(data.get("time_scope") or "").strip()[:40]
            if scope and scope.lower() != "unspecified":
                ctx.time_scope = scope
                added.append("time_scope")

        # Continuation may only be turned ON, never off: the lexicon match is a
        # concrete phrase in the goal, so a model saying "false" cannot erase it.
        if not ctx.continuation and data.get("continuation") is True:
            ctx.continuation = True
            added.append("continuation")

        if added:
            ctx.author = f"{self.llm.reasoner_name}+heuristic"
            ctx.metadata["enriched_fields"] = sorted(set(added))
        else:
            ctx.metadata["enrichment"] = "nothing to add"
