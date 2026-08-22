"""Tool contract for the agent.

A "tool" is what the agent reasons about and chooses to call. Each tool sits on
top of one or more *providers* (the connectors in `app/sources/`), so a missing
API key degrades a provider — never the tool, and never the run.

Design rules that matter for genuine agentic behaviour:
  * Tools describe themselves (`description`, `when_to_use`, `input_schema`) so
    the decision engine can pick between them instead of following a script.
  * Tools return structured observations, not prose.
  * Tools never raise. Failure is a `ToolResult` with `ok=False`.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..sources.base import RawItem, SourceQuery
from ..sources.registry import SourceRegistry
from ..sources.resilience import collect_from_source

# What the agent is allowed to ask for in one call.
MAX_TOOL_LIMIT = 25


# ─────────────────────────────────────────────────────────────
# Normalized data shapes
# ─────────────────────────────────────────────────────────────
@dataclass
class FindingRecord:
    """One piece of evidence, normalized across every tool and provider."""

    id: str
    title: str
    source: str                 # research | news | competitor | patent
    summary: str
    url: str
    published_date: str | None
    provider: str = ""          # arxiv, openalex, rss, patentsview, ...
    tool: str = ""
    author: str = ""
    raw_text: str = ""
    competitor: str = ""        # set when the item is competitor-scoped
    credibility: str = "standard"
    simulated: bool = False
    signals: list[str] = field(default_factory=list)
    relevance: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    # Multi-agent provenance: which specialist found this, and which other agents
    # independently surfaced the same development.
    discovered_by: str = ""
    corroborated_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Provider metadata worth surfacing in the UI. Whitelisted rather than passing
    # `meta` wholesale, so internal/debug keys never leak into a client response.
    PUBLIC_META_KEYS = (
        "citation_count",
        "venue",
        "institutions",
        "institution",
        "concepts",
        "categories",
        "assignee",
        "patent_number",
        "filing_date",
        "cpc",
        "outlet",
        "domain",
        "stars",
        "language",
        "subreddit",
        "score",
        "num_comments",
        "points",
        "tavily_score",
    )

    def public(self) -> dict[str, Any]:
        """The compact shape the API returns for `findings`."""
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "summary": self.summary,
            "url": self.url,
            "published_date": self.published_date,
            "provider": self.provider,
            "tool": self.tool,
            "author": self.author,
            "competitor": self.competitor,
            "credibility": self.credibility,
            "simulated": self.simulated,
            "signals": self.signals,
            "relevance": round(self.relevance, 3),
            "discovered_by": self.discovered_by,
            "corroborated_by": self.corroborated_by,
            "meta": {
                k: self.meta[k]
                for k in self.PUBLIC_META_KEYS
                if k in self.meta and self.meta[k] not in (None, "", [], {})
            },
        }


@dataclass
class ToolInput:
    """What the agent decided to ask this tool for."""

    query: str = ""
    keywords: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    limit: int = 10
    since_days: int = 45
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ToolInput":
        self.query = (self.query or "").strip()[:300]
        self.keywords = [k.strip()[:120] for k in self.keywords if str(k).strip()][:8]
        self.competitors = [c.strip()[:120] for c in self.competitors if str(c).strip()][:8]
        self.limit = max(1, min(int(self.limit or 10), MAX_TOOL_LIMIT))
        self.since_days = max(1, min(int(self.since_days or 45), 365))
        if not self.query and self.keywords:
            self.query = self.keywords[0]
        return self

    def signature(self, tool: str = "") -> str:
        """Identity of a *tool call*, used to stop the agent repeating itself.

        The tool name is part of the identity. Without it, asking the news tool and
        the competitor tool the same question hashes identically, and the second
        one gets silently suppressed as a duplicate — which starves a required
        information need.
        """
        raw = "|".join(
            [
                tool,
                self.query.lower(),
                ",".join(sorted(k.lower() for k in self.keywords)),
                ",".join(sorted(c.lower() for c in self.competitors)),
                str(self.since_days),
            ]
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "keywords": self.keywords,
            "competitors": self.competitors,
            "limit": self.limit,
            "since_days": self.since_days,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class ToolResult:
    """An observation. Never an exception."""

    tool: str
    ok: bool = True
    items: list[FindingRecord] = field(default_factory=list)
    error: str = ""
    latency_ms: int = 0
    providers_used: list[str] = field(default_factory=list)
    providers_failed: list[dict[str, str]] = field(default_factory=list)
    simulated: bool = False
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.items)


@dataclass
class ToolAvailability:
    name: str
    available: bool
    providers_live: list[str] = field(default_factory=list)
    providers_simulated: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ToolContext:
    """Shared per-run execution context handed to every tool."""

    http_client: Any
    registry: SourceRegistry
    simulation_mode: bool = False


# ─────────────────────────────────────────────────────────────
# Base tool
# ─────────────────────────────────────────────────────────────
class Tool(ABC):
    name: str = "tool"
    display_name: str = "Tool"
    source_label: str = "news"
    description: str = ""
    when_to_use: str = ""
    provider_names: tuple[str, ...] = ()

    input_schema: dict[str, str] = {
        "query": "string — the search phrase to use",
        "keywords": "string[] — topic keywords to match",
        "competitors": "string[] — company names, only if relevant to this tool",
        "since_days": "int — how far back to look",
        "limit": "int — max results (<=25)",
    }

    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry

    # ── self-description for the decision engine ────────────
    def catalog_entry(self) -> dict[str, Any]:
        av = self.availability()
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "input_schema": self.input_schema,
            "available": av.available,
            "providers_live": av.providers_live,
            "providers_simulated": av.providers_simulated,
        }

    def availability(self) -> ToolAvailability:
        live: list[str] = []
        simulated: list[str] = []
        for name in self.provider_names:
            connector = self.registry.get(name)
            if connector is None:
                continue
            (live if connector.available() else simulated).append(name)
        if not live and not simulated:
            return ToolAvailability(
                name=self.name, available=False, reason="no providers registered"
            )
        return ToolAvailability(
            name=self.name,
            available=True,
            providers_live=live,
            providers_simulated=simulated,
            reason="" if live else "no live provider — serving clearly-labelled simulated data",
        )

    # ── execution ───────────────────────────────────────────
    async def run(self, tool_input: ToolInput, ctx: ToolContext) -> ToolResult:
        """Execute the tool. Guaranteed not to raise."""
        tool_input = tool_input.normalized()
        started = time.perf_counter()
        result = ToolResult(tool=self.name)
        try:
            await self._execute(tool_input, ctx, result)
        except Exception as exc:  # noqa: BLE001 — a tool bug must not end the run
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        if result.items:
            result.ok = True
        elif not result.error:
            result.note = result.note or "no matching results in the requested window"
        return result

    @abstractmethod
    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        """Populate `result`. May raise; `run` contains the damage."""

    # ── shared provider plumbing ────────────────────────────
    async def _collect(
        self,
        provider_names: tuple[str, ...] | list[str],
        tool_input: ToolInput,
        ctx: ToolContext,
        result: ToolResult,
        *,
        source_label: str | None = None,
        competitor: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[FindingRecord]:
        """Fan out over providers, tolerate individual failures, normalize output."""
        collected: list[FindingRecord] = []
        for name in provider_names:
            connector = self.registry.get(name)
            if connector is None:
                continue
            query = SourceQuery(
                source=name,
                source_type=connector.source_type,
                query=tool_input.query,
                keywords=tool_input.keywords,
                competitors=tool_input.competitors,
                limit=tool_input.limit,
                since_days=tool_input.since_days,
                extra=dict(extra or {}),
            )
            outcome = await collect_from_source(
                connector, ctx.http_client, query, simulation_mode=ctx.simulation_mode
            )
            if outcome.simulated:
                result.simulated = True
            if (outcome.ok or outcome.items) and name not in result.providers_used:
                # Tools that sweep per-competitor hit the same provider repeatedly;
                # the observation should report each provider once.
                result.providers_used.append(name)
            if not outcome.ok:
                # Dedupe on provider name, not the whole dict: a tool that sweeps
                # per-competitor hits the same provider repeatedly and the breaker
                # note changes between attempts, so identical failures look distinct.
                if not any(f["provider"] == name for f in result.providers_failed):
                    result.providers_failed.append(
                        {
                            "provider": name,
                            "error": outcome.error or "unknown",
                            "note": outcome.note,
                        }
                    )
            if outcome.broadened and not result.note:
                result.note = outcome.note
            for item in outcome.items:
                collected.append(
                    self._normalize(
                        item,
                        source_label=source_label or self.source_label,
                        competitor=competitor,
                    )
                )
        return collected

    def _normalize(
        self, item: RawItem, *, source_label: str, competitor: str = ""
        ) -> FindingRecord:
        summary = _first_sentences(item.raw_text or item.title, limit=2)
        return FindingRecord(
            id=_stable_id(item),
            title=item.title,
            source=source_label,
            summary=summary,
            url=item.url,
            published_date=item.published_at.date().isoformat() if item.published_at else None,
            provider=item.source_name,
            tool=self.name,
            author=item.author,
            raw_text=item.raw_text,
            competitor=competitor,
            credibility=item.credibility,
            simulated=item.is_simulated,
            meta=dict(item.meta),
        )

    def unavailable_reason(self) -> str:
        """Human-readable explanation used in the activity log when unusable."""
        return self.availability().reason or "tool unavailable"


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, limit: int = 2, max_chars: int = 420) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    out = " ".join(parts[:limit])
    return (out[: max_chars - 1] + "…") if len(out) > max_chars else out


def _stable_id(item: RawItem) -> str:
    """Deterministic id from the canonical identity of the item."""
    basis = (item.url or "").strip().lower() or f"{item.source_name}:{item.title.strip().lower()}"
    basis = re.sub(r"[?#].*$", "", basis).rstrip("/")
    return hashlib.sha1(basis.encode()).hexdigest()[:16]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
