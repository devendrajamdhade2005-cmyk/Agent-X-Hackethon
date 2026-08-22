"""Specialist agents.

Each specialist runs its own bounded ReAct loop over a *restricted* tool set and
reports back a structured `AgentReport`. Specialisation is enforced structurally,
not by naming: the decision engine handed to each agent physically cannot see the
other agent's tools, so a research agent has no way to call live web search and a
competitive agent has no way to query arXiv.

They write into the shared `AgentState` (findings, tool calls, observations,
decisions) so dedup, scoring and the activity log stay single-sourced. What is
per-agent is the *decision scope*.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..tools.base import ToolContext
from ..tools.signals import SIGNAL_LABELS
from .decision_engine import RELEVANCE_THRESHOLD, DecisionEngine
from .messages import AgentProfile, AgentReport, AgentTask
from .state import Decision

if TYPE_CHECKING:  # pragma: no cover
    from .agent import InsightPulseAgent


class SpecialistAgent:
    """Base specialist. Owns a decision scope; borrows the host's execution plumbing."""

    def __init__(self, profile: AgentProfile) -> None:
        self.profile = profile

    # ── public API ──────────────────────────────────────────
    async def execute(
        self, host: "InsightPulseAgent", task: AgentTask, ctx: ToolContext
    ) -> AgentReport:
        started = time.perf_counter()
        state = host.state
        logger = host.logger

        report = AgentReport(
            run_id=state.run_id,
            from_agent=self.profile.key,
            task=task.task,
        )

        state.active_agent = self.profile.key
        logger.speaking_as(self.profile.key)

        before_ids = set(state.seen_finding_ids)
        my_tool_calls_before = len(state.tool_calls)

        # A scoped engine: only this agent's tools are reachable.
        engine = DecisionEngine(
            host.tools, host.llm, allowed_tools=set(task.allowed_tools)
        )

        logger.log(
            "thought",
            f"{self.profile.name} starting",
            task.task,
            agent=self.profile.key,
            allowed_tools=list(task.allowed_tools),
            need_keys=list(task.need_keys),
        )

        iterations = 0
        try:
            while iterations < task.max_iterations:
                if state.iteration_count >= state.max_iterations:
                    report.observations.append(
                        "Stopped early: the run-wide reasoning-step limit was reached."
                    )
                    break

                decision = await self._decide(host, engine)
                state.decisions.append(decision)

                if decision.action != "call_tool" or not decision.tool or not decision.tool_input:
                    report.observations.append(decision.reasoning)
                    break

                iterations += 1
                state.iteration_count += 1
                step = state.iteration_count

                logger.log(
                    "decision",
                    f"{self.profile.name} → {decision.tool}",
                    decision.reasoning,
                    iteration=step,
                    agent=self.profile.key,
                    tool=decision.tool,
                    author=decision.author,
                    confidence=round(decision.confidence, 2),
                    tool_input=decision.tool_input.describe(),
                )

                result = await host._call_tool(decision, ctx, step)  # noqa: SLF001
                host._observe(decision, result, iteration=step, agent=self.profile.key)  # noqa: SLF001

                obs = state.observations[-1] if state.observations else None
                if obs is not None:
                    report.observations.append(obs.summary)
                    for sig in obs.signals:
                        if sig not in report.signals:
                            report.signals.append(sig)
                if result.tool not in report.tools_used:
                    report.tools_used.append(result.tool)
                for provider in result.providers_used:
                    if provider not in report.sources_checked:
                        report.sources_checked.append(provider)
                for failure in result.providers_failed:
                    entry = {
                        "provider": failure.get("provider", ""),
                        "error": failure.get("error", ""),
                    }
                    if entry not in report.degraded_providers:
                        report.degraded_providers.append(entry)
                if result.error:
                    report.errors.append(f"{result.tool}: {result.error}")

        except Exception as exc:  # noqa: BLE001 — one agent must not end the run
            report.status = "failed"
            report.errors.append(f"{type(exc).__name__}: {exc}")
            state.record_error(f"agent:{self.profile.key}", f"{type(exc).__name__}: {exc}")
            logger.error(
                f"{self.profile.name} failed",
                f"{type(exc).__name__}: {exc}. The orchestrator will continue with the "
                f"other agents.",
                agent=self.profile.key,
            )

        # Attribute everything this agent newly discovered.
        mine = [f for f in state.findings if f.id not in before_ids]
        for finding in mine:
            if not finding.discovered_by:
                finding.discovered_by = self.profile.key

        report.finding_ids = [f.id for f in mine]
        report.findings_count = len(mine)
        report.relevant_count = sum(1 for f in mine if f.relevance >= RELEVANCE_THRESHOLD)
        report.duration_ms = int((time.perf_counter() - started) * 1000)
        report.coverage = self._coverage(mine, report)
        report.confidence = self._confidence(mine, report)

        if report.status != "failed":
            report.status = self._status(report, state.tool_calls[my_tool_calls_before:])

        self.enrich(report, mine, state)
        report.reasoning_summary = self.summarize(report, mine)
        report.recommended_next_step = self.next_step(report, mine, state)

        state.completed_agents.append(self.profile.key)
        state.active_agent = ""
        logger.speaking_as("orchestrator")
        return report

    # ── overridable specialisation ──────────────────────────
    def enrich(self, report: AgentReport, findings: list[Any], state: Any) -> None:
        """Populate the agent-specific fields of the report."""

    def summarize(self, report: AgentReport, findings: list[Any]) -> str:
        return (
            f"Checked {len(report.sources_checked)} source(s) via "
            f"{', '.join(report.tools_used) or 'no tools'}; "
            f"{report.findings_count} finding(s), {report.relevant_count} relevant."
        )

    def next_step(self, report: AgentReport, findings: list[Any], state: Any) -> str:
        return ""

    # ── internals ───────────────────────────────────────────
    async def _decide(self, host: "InsightPulseAgent", engine: DecisionEngine) -> Decision:
        try:
            return await engine.decide(host.state)
        except Exception as exc:  # noqa: BLE001
            host.state.record_error(
                f"decision:{self.profile.key}", f"{type(exc).__name__}: {exc}"
            )
            return Decision(
                action="finalize",
                reasoning=(
                    f"{self.profile.name} could not choose a next action "
                    f"({type(exc).__name__}); handing back what it has."
                ),
                author="guardrail",
                iteration=host.state.iteration_count,
            )

    def _status(self, report: AgentReport, calls: list[Any]) -> str:
        if not calls:
            return "skipped"
        if report.degraded_providers and report.findings_count:
            return "partial"
        if not report.findings_count:
            return "partial"
        return "completed"

    def _coverage(self, findings: list[Any], report: AgentReport) -> str:
        if not findings:
            return "unavailable" if report.errors or report.degraded_providers else "partial"
        simulated = sum(1 for f in findings if f.simulated)
        if simulated == 0:
            return "live"
        if simulated == len(findings):
            return "simulated"
        return "partial"

    def _confidence(self, findings: list[Any], report: AgentReport) -> float:
        """Mean relevance of what was found, discounted for degraded coverage."""
        if not findings:
            return 0.0
        base = sum(f.relevance for f in findings) / len(findings)
        live_share = sum(1 for f in findings if not f.simulated) / len(findings)
        penalty = 0.12 * len(report.degraded_providers)
        return max(0.0, min(1.0, base * (0.6 + 0.4 * live_share) - penalty))


# ─────────────────────────────────────────────────────────────
class ResearchIntelligenceAgent(SpecialistAgent):
    """Academic and technological research: papers, methods, benchmarks, filings."""

    def enrich(self, report: AgentReport, findings: list[Any], state: Any) -> None:
        # Trends = recurring technical phrases across the titles this agent found.
        report.research_trends = _recurring_phrases(findings)
        report.key_developments = [
            f.title for f in sorted(findings, key=lambda x: -x.relevance)[:3]
        ]

    def summarize(self, report: AgentReport, findings: list[Any]) -> str:
        if not findings:
            return (
                "No research matched the goal in the requested window. "
                + ("Providers were degraded, so coverage is incomplete."
                   if report.degraded_providers else
                   "The academic sources returned nothing relevant.")
            )
        cited = [f for f in findings if int((f.meta or {}).get("citation_count") or 0) > 0]
        parts = [
            f"Searched {', '.join(report.sources_checked) or 'no providers'} and returned "
            f"{report.findings_count} item(s), {report.relevant_count} relevant to the goal."
        ]
        if report.research_trends:
            parts.append(f"Recurring themes: {', '.join(report.research_trends[:3])}.")
        if cited:
            parts.append(f"{len(cited)} carry citation counts.")
        return " ".join(parts)

    def next_step(self, report: AgentReport, findings: list[Any], state: Any) -> str:
        if not findings:
            return "Broaden the keywords or extend the publication window."
        if state.competitors and not state.findings_by_agent("competitive_agent"):
            return (
                "Hand to the Competitive Intelligence Agent to check whether the tracked "
                "companies are commercialising these methods."
            )
        return ""


# ─────────────────────────────────────────────────────────────
class CompetitiveIntelligenceAgent(SpecialistAgent):
    """Companies and the market: launches, funding, partnerships, shipped code."""

    def enrich(self, report: AgentReport, findings: list[Any], state: Any) -> None:
        seen: list[str] = []
        for f in findings:
            if f.competitor and f.competitor not in seen:
                seen.append(f.competitor)
        # Companies the agent was asked about, plus any it actually attributed.
        report.competitors_analyzed = list(
            dict.fromkeys([*state.competitors, *seen])
        )[:8]
        report.market_signals = [
            SIGNAL_LABELS.get(s, s) for s in report.signals
        ][:6]

    def summarize(self, report: AgentReport, findings: list[Any]) -> str:
        if not findings:
            return (
                "No company activity surfaced for the tracked names. "
                + ("Some providers were unavailable, so coverage is partial."
                   if report.degraded_providers else
                   "Curated news and live web search returned nothing attributable.")
            )
        attributed = sum(1 for f in findings if f.competitor)
        parts = [
            f"Swept {', '.join(report.sources_checked) or 'no providers'} and returned "
            f"{report.findings_count} item(s); {attributed} directly attributed to a "
            f"tracked company."
        ]
        if report.market_signals:
            parts.append(f"Market signals detected: {', '.join(report.market_signals[:3])}.")
        uncovered = [
            c for c in report.competitors_analyzed
            if not any((f.competitor or "").lower() == c.lower() for f in findings)
        ]
        if uncovered:
            parts.append(f"No activity found for {', '.join(uncovered[:2])}.")
        return " ".join(parts)

    def next_step(self, report: AgentReport, findings: list[Any], state: Any) -> str:
        technical = {"patent", "benchmark"} & set(report.signals)
        if technical and not state.findings_by_agent("research_agent"):
            return (
                "Ask the Research Intelligence Agent whether published work underpins the "
                "capability these companies are claiming."
            )
        if not findings and "web_search" not in report.tools_used:
            return "Escalate to live web search — curated feeds had no coverage."
        return ""


# ─────────────────────────────────────────────────────────────
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "its", "new",
    "using", "based", "toward", "towards", "via", "system", "method", "systems",
    "methods", "approach", "model", "models", "data", "study", "paper", "research",
    "learning", "network", "networks", "framework", "analysis", "generation",
}


def _recurring_phrases(findings: list[Any], limit: int = 4) -> list[str]:
    """Two-word phrases appearing in more than one title — an observed trend."""
    counts: dict[str, int] = {}
    for f in findings:
        words = [
            w for w in str(f.title or "").lower()
            .replace("-", " ").replace(":", " ").split()
            if len(w) > 3 and w.isalpha() and w not in _STOP
        ]
        seen_here: set[str] = set()
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if phrase in seen_here:
                continue
            seen_here.add(phrase)
            counts[phrase] = counts.get(phrase, 0) + 1
    return [p for p, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 1][:limit]


RESEARCH = ResearchIntelligenceAgent
COMPETITIVE = CompetitiveIntelligenceAgent
