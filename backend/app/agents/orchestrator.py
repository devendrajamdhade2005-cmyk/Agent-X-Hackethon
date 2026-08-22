"""Intelligence Orchestrator — the coordinating agent.

Responsibilities, in order:

  1. ANALYSE the goal (reusing the existing planner's information needs).
  2. PLAN which specialists are needed — and record *why each was or wasn't chosen*.
     It does not run every agent every time; that is the whole point.
  3. DELEGATE a scoped task to each selected specialist, in a deliberate order.
  4. REVIEW each returned report and decide whether it changes the plan.
  5. COLLABORATE — issue follow-up tasks when one agent's findings imply work for
     another, and cross-validate overlapping evidence.
  6. MERGE — deduplicate across agents, keep every source reference, and raise
     confidence where independent agents corroborate the same development.
  7. HAND OFF the merged evidence to the existing analyst (insight generator).

The orchestrator never touches a tool directly. It only sends `AgentTask`s and
reads `AgentReport`s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tools.base import FindingRecord, ToolContext
from ..tools.signals import STRATEGIC_SIGNALS
from .messages import (
    COMPETITIVE_AGENT,
    ORCHESTRATOR,
    RESEARCH_AGENT,
    AgentProfile,
    AgentReport,
    AgentTask,
    CollaborationEvent,
    ExecutionPlanEntry,
)
from .specialists import CompetitiveIntelligenceAgent, ResearchIntelligenceAgent

if TYPE_CHECKING:  # pragma: no cover
    from .agent import InsightPulseAgent

# Corroboration: two findings from different agents describing one development.
_TITLE_OVERLAP = 0.34
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "its", "new",
    "using", "based", "announces", "announced", "launches", "launched", "says", "said",
    "report", "reports", "a", "an", "to", "in", "on", "of", "at", "by", "as", "is", "are",
}


class IntelligenceOrchestrator:
    """Plans, delegates, cross-validates and merges. Owns no tools."""

    profile: AgentProfile = ORCHESTRATOR

    def __init__(self) -> None:
        self.specialists = {
            RESEARCH_AGENT.key: ResearchIntelligenceAgent(RESEARCH_AGENT),
            COMPETITIVE_AGENT.key: CompetitiveIntelligenceAgent(COMPETITIVE_AGENT),
        }
        self.reports: dict[str, AgentReport] = {}

    # ─────────────────────────────────────────────────────────
    async def run(self, host: "InsightPulseAgent", ctx: ToolContext) -> None:
        state = host.state
        logger = host.logger
        logger.speaking_as(self.profile.key)

        plan = self._build_plan(host)
        state.execution_plan = [e.to_dict() for e in plan]

        selected = [e for e in plan if e.selected]
        skipped = [e for e in plan if not e.selected]

        logger.orchestration(
            "Execution plan created",
            "; ".join(f"{e.agent.replace('_', ' ')}: {e.reason}" for e in plan),
            agent=self.profile.key,
            selected=[e.agent for e in selected],
            skipped=[e.agent for e in skipped],
        )
        for entry in skipped:
            logger.orchestration(
                f"{_short(entry.agent)} not selected",
                entry.reason,
                agent=self.profile.key,
            )

        if not selected:
            logger.warning(
                "No specialist matched the goal",
                "Falling back to the research agent so the run still produces evidence.",
                agent=self.profile.key,
            )
            selected = [self._entry(RESEARCH_AGENT, True, "fallback: no other agent matched", 1)]
            state.execution_plan = [e.to_dict() for e in [*plan, *selected]]

        # ── delegate, in planned order ───────────────────────
        for entry in sorted(selected, key=lambda e: e.order):
            task = self._task(host, entry, kind="primary")
            await self._delegate(host, ctx, task)

            # Review what came back before deciding anything else.
            report = self.reports.get(entry.agent)
            if report is not None:
                self._review(host, report)

        # ── collaboration: follow-up work implied by the results ──
        await self._collaborate(host, ctx, plan)

        # ── merge and cross-validate ─────────────────────────
        self._merge(host)

        logger.speaking_as(self.profile.key)
        logger.orchestration(
            "Evidence consolidated",
            f"{len(state.findings)} finding(s) from "
            f"{len(state.completed_agents)} agent(s); "
            f"{len(state.corroborated_finding_ids)} cross-validated.",
            agent=self.profile.key,
            agents=list(state.completed_agents),
        )

    # ─────────────────────────────────────────────────────────
    # 2. PLAN — goal-driven agent selection
    # ─────────────────────────────────────────────────────────
    def _build_plan(self, host: "InsightPulseAgent") -> list[ExecutionPlanEntry]:
        state = host.state
        needs = state.plan.needs
        required = {n.key for n in needs if n.required}
        conditional = {n.key for n in needs if not n.required}

        entries: list[ExecutionPlanEntry] = []

        # ── Research agent ──
        research_required = required & set(RESEARCH_AGENT.need_keys)
        if research_required:
            keys = sorted(research_required)
            entries.append(self._entry(
                RESEARCH_AGENT, True,
                f"the goal requires {', '.join(keys)} intelligence — academic and IP sources "
                f"are the right place to look",
                order=1 if "research" in research_required else 2,
                need_keys=keys,
            ))
        else:
            reachable = conditional & set(RESEARCH_AGENT.need_keys)
            entries.append(self._entry(
                RESEARCH_AGENT, False,
                (
                    f"no research or patent need is required by the goal; "
                    f"{', '.join(sorted(reachable))} stays conditional and will only be "
                    f"pursued if the evidence calls for it"
                    if reachable else
                    "the goal does not call for academic or IP sources"
                ),
                order=9,
            ))

        # ── Competitive agent ──
        # Market needs alone are not enough to justify company monitoring: a purely
        # academic goal like "recent research on multi-agent RL" trips generic
        # recency/news hints, and running a competitor sweep on it only adds noise.
        # It is selected when the goal is actually about companies or is
        # market-led rather than research-led.
        competitive_required = required & set(COMPETITIVE_AGENT.need_keys)
        research_led = bool(required & {"research", "patent"})
        market_led = bool(required & {"news", "web"}) and not research_led
        company_scoped = "competitor" in required or bool(state.competitors)

        if competitive_required and (company_scoped or market_led):
            keys = sorted(competitive_required)
            if company_scoped:
                reason = (
                    f"the goal names companies to track"
                    f"{': ' + ', '.join(state.competitors[:3]) if state.competitors else ''}, "
                    f"so {', '.join(keys)} intelligence is required"
                )
            else:
                reason = (
                    f"the goal is market-led rather than academic — it requires "
                    f"{', '.join(keys)} intelligence"
                )
            entries.append(self._entry(
                COMPETITIVE_AGENT, True, reason,
                # Companies first: what they did shapes what is worth reading.
                order=1 if state.competitors else 2,
                need_keys=keys,
            ))
        elif competitive_required and research_led:
            entries.append(self._entry(
                COMPETITIVE_AGENT, False,
                (
                    f"the goal is research-led and names no companies — "
                    f"{', '.join(sorted(competitive_required))} coverage was implied only by "
                    f"generic recency wording, so a competitor sweep would add noise"
                ),
                order=9,
            ))
        else:
            entries.append(self._entry(
                COMPETITIVE_AGENT, False,
                (
                    "the goal is academic rather than commercial — no companies named and no "
                    "market need required, so competitor monitoring would add noise"
                ),
                order=9,
            ))

        return entries

    def _ensure_need(self, host: "InsightPulseAgent", key: str, reason: str) -> None:
        """Add an information need the original plan did not contain.

        A follow-up task is pointless if the specialist has no need to satisfy — its
        decision engine would find no candidate action. When the orchestrator decides
        mid-run that a new angle matters, it revises the plan and says so. That plan
        revision is itself evidence of observation-driven reasoning.
        """
        from .state import InformationNeed

        state = host.state
        if state.plan.need(key) is not None:
            existing = state.plan.need(key)
            if existing is not None and not existing.required:
                existing.required = True
                existing.reason = reason
                state.plan.revisions.append(f"promoted '{key}' to required: {reason}")
                host.logger.orchestration(
                    f"Plan revised — '{key}' promoted to required",
                    reason,
                    agent=self.profile.key,
                )
            return

        state.plan.needs.append(
            InformationNeed(key=key, reason=reason, required=True, min_items=1)
        )
        state.plan.revisions.append(f"added '{key}' need: {reason}")
        host.logger.orchestration(
            f"Plan revised — '{key}' need added",
            reason,
            agent=self.profile.key,
        )

    def _entry(
        self,
        profile: AgentProfile,
        selected: bool,
        reason: str,
        order: int,
        need_keys: list[str] | None = None,
    ) -> ExecutionPlanEntry:
        return ExecutionPlanEntry(
            agent=profile.key,
            selected=selected,
            reason=reason,
            order=order,
            need_keys=need_keys or [],
            allowed_tools=list(profile.tool_names) if selected else [],
        )

    # ─────────────────────────────────────────────────────────
    # 3. DELEGATE
    # ─────────────────────────────────────────────────────────
    def _task(
        self,
        host: "InsightPulseAgent",
        entry: ExecutionPlanEntry,
        *,
        kind: str = "primary",
        brief: str = "",
        reason: str = "",
    ) -> AgentTask:
        state = host.state
        profile = self.specialists[entry.agent].profile

        if not brief:
            if entry.agent == RESEARCH_AGENT.key:
                brief = (
                    f"Find recent research, methods and technical developments related to "
                    f"{_topics(state)}."
                )
                if "patent" in entry.need_keys:
                    brief += " Include patent filings that protect this technology."
            else:
                brief = (
                    f"Find recent company activity related to {_topics(state)} — "
                    f"announcements, launches, funding, partnerships and shipped code"
                )
                brief += (
                    f" for {', '.join(state.competitors[:4])}." if state.competitors else "."
                )

        return AgentTask(
            run_id=state.run_id,
            from_agent=self.profile.key,
            to_agent=entry.agent,
            task=brief,
            reason=reason or entry.reason,
            kind=kind,  # type: ignore[arg-type]
            allowed_tools=list(profile.tool_names),
            need_keys=list(entry.need_keys) or list(profile.need_keys),
            max_iterations=3 if kind == "primary" else 1,
            context={
                "user_goal": state.user_goal,
                "keywords": list(state.keywords),
                "competitors": list(state.competitors),
                "previous_observations": [o.summary for o in state.observations[-3:]],
            },
        )

    async def _delegate(
        self, host: "InsightPulseAgent", ctx: ToolContext, task: AgentTask
    ) -> AgentReport:
        state = host.state
        logger = host.logger

        state.agent_messages.append(task.to_dict())
        state.pending_tasks.append({"agent": task.to_agent, "task": task.task})

        specialist = self.specialists[task.to_agent]
        logger.speaking_as(self.profile.key)
        logger.delegation(
            f"{specialist.profile.icon} {specialist.profile.name} assigned",
            f"{task.task} — {task.reason}",
            agent=self.profile.key,
            to_agent=task.to_agent,
            allowed_tools=task.allowed_tools,
            kind=task.kind,
        )

        report = await specialist.execute(host, task, ctx)

        state.pending_tasks = [
            t for t in state.pending_tasks if t.get("agent") != task.to_agent
        ]
        state.agent_messages.append(report.to_dict())

        # A follow-up report augments the primary one rather than replacing it.
        existing = self.reports.get(task.to_agent)
        if existing is not None and task.kind == "follow_up":
            merged = _merge_reports(existing, report)
            self.reports[task.to_agent] = merged
            state.agent_reports = [
                merged.to_dict() if r.get("from_agent") == task.to_agent else r
                for r in state.agent_reports
            ]
        else:
            self.reports[task.to_agent] = report
            state.agent_reports.append(report.to_dict())

        logger.speaking_as(self.profile.key)
        logger.log(
            "observation",
            f"{specialist.profile.name} reported {report.findings_count} finding(s)",
            report.reasoning_summary,
            agent=self.profile.key,
            from_agent=task.to_agent,
            status=report.status,
            coverage=report.coverage,
            confidence=round(report.confidence, 2),
            tools_used=report.tools_used,
        )
        return report

    # ─────────────────────────────────────────────────────────
    # 4. REVIEW
    # ─────────────────────────────────────────────────────────
    def _review(self, host: "InsightPulseAgent", report: AgentReport) -> None:
        """State what the orchestrator concluded from this report."""
        logger = host.logger
        logger.speaking_as(self.profile.key)

        if report.status == "failed":
            logger.orchestration(
                f"{_short(report.from_agent)} failed — continuing without it",
                "Its findings are missing from the briefing; the remaining agents still report.",
                agent=self.profile.key,
            )
            return

        if report.coverage in {"simulated", "unavailable"}:
            logger.warning(
                f"{_short(report.from_agent)} coverage is {report.coverage}",
                "Findings from this agent are labelled accordingly and carry lower confidence.",
                agent=self.profile.key,
                coverage=report.coverage,
            )

        if report.recommended_next_step:
            logger.orchestration(
                f"{_short(report.from_agent)} recommends a next step",
                report.recommended_next_step,
                agent=self.profile.key,
            )

    # ─────────────────────────────────────────────────────────
    # 5. COLLABORATE — one agent's result creates work for another
    # ─────────────────────────────────────────────────────────
    async def _collaborate(
        self, host: "InsightPulseAgent", ctx: ToolContext, plan: list[ExecutionPlanEntry]
    ) -> None:
        state = host.state
        logger = host.logger
        logger.speaking_as(self.profile.key)

        competitive = self.reports.get(COMPETITIVE_AGENT.key)
        research = self.reports.get(RESEARCH_AGENT.key)

        # ── Competitive found a technical claim → ask Research to validate it ──
        if competitive and not research and state.budget_left() > 0:
            technical = set(competitive.signals) & {"patent", "benchmark", "launch"}
            if technical:
                trigger = sorted(technical)[0]
                subject = _lead_subject(state, COMPETITIVE_AGENT.key) or _topics(state)
                entry = self._entry(
                    RESEARCH_AGENT, True,
                    "cross-agent validation requested by the orchestrator",
                    order=3,
                    need_keys=["research"],
                )
                # The original plan had no research need, so give the specialist one
                # to satisfy — otherwise it has nothing it is allowed to act on.
                self._ensure_need(
                    host, "research",
                    f"validating the '{trigger}' signal reported by the Competitive "
                    f"Intelligence Agent",
                )
                task = self._task(
                    host, entry, kind="follow_up",
                    brief=(
                        f"Check whether published research supports the technology behind: "
                        f"{subject}."
                    ),
                    reason=(
                        f"the Competitive Intelligence Agent detected a '{trigger}' signal, so "
                        f"the underlying technical claim needs independent validation"
                    ),
                )
                self._record_collaboration(
                    host,
                    kind="follow_up",
                    initiator=self.profile.key,
                    participants=[COMPETITIVE_AGENT.key, RESEARCH_AGENT.key],
                    summary="Research validation requested for a competitor technology claim",
                    detail=(
                        f"The Competitive Intelligence Agent surfaced a '{trigger}' signal. The "
                        f"orchestrator asked the Research Intelligence Agent to check the "
                        f"published record before the claim is prioritized."
                    ),
                    evidence=[subject] if subject else [],
                )
                await self._delegate(host, ctx, task)
                research = self.reports.get(RESEARCH_AGENT.key)

        # ── Research found strong work on tracked companies' turf → ask Competitive ──
        if (
            research
            and not competitive
            and state.competitors
            and state.budget_left() > 0
            and research.findings_count > 0
        ):
            entry = self._entry(
                COMPETITIVE_AGENT, True,
                "cross-agent validation requested by the orchestrator",
                order=3,
                need_keys=["web"],
            )
            self._ensure_need(
                host, "web",
                "checking whether the tracked companies have commercialised the research "
                "direction the Research Intelligence Agent identified",
            )
            task = self._task(
                host, entry, kind="follow_up",
                brief=(
                    f"Check whether {', '.join(state.competitors[:3])} have commercialised the "
                    f"research direction identified for {_topics(state)}."
                ),
                reason=(
                    "the Research Intelligence Agent found relevant work and the goal names "
                    "companies, so commercial follow-through is worth checking"
                ),
            )
            self._record_collaboration(
                host,
                kind="follow_up",
                initiator=self.profile.key,
                participants=[RESEARCH_AGENT.key, COMPETITIVE_AGENT.key],
                summary="Commercial follow-through requested for a research direction",
                detail=(
                    "The Research Intelligence Agent identified relevant published work. The "
                    "orchestrator asked the Competitive Intelligence Agent whether the tracked "
                    "companies are shipping it."
                ),
                evidence=research.key_developments[:2],
            )
            await self._delegate(host, ctx, task)

    # ─────────────────────────────────────────────────────────
    # 6. MERGE + cross-validate
    # ─────────────────────────────────────────────────────────
    def _merge(self, host: "InsightPulseAgent") -> None:
        state = host.state
        logger = host.logger
        logger.speaking_as(self.profile.key)

        by_agent: dict[str, list[FindingRecord]] = {}
        for f in state.findings:
            by_agent.setdefault(f.discovered_by or "unattributed", []).append(f)

        if len(by_agent) < 2:
            state.merged_note = "single agent contributed — nothing to cross-validate"
            return

        agents = [a for a in by_agent if a != "unattributed"]
        corroborated: list[tuple[FindingRecord, FindingRecord]] = []

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                for a in by_agent[agents[i]]:
                    for b in by_agent[agents[j]]:
                        if _describes_same_development(a, b):
                            corroborated.append((a, b))

        if not corroborated:
            logger.orchestration(
                "No duplicate developments across agents",
                f"{', '.join(_short(a) for a in agents)} reported on distinct events, so no "
                f"finding was merged. Checking for topical support instead.",
                agent=self.profile.key,
            )
            self._cross_support(host, by_agent)
            return

        boosted = 0
        for a, b in corroborated:
            for primary, other in ((a, b), (b, a)):
                agent = other.discovered_by
                if agent and agent not in primary.corroborated_by:
                    primary.corroborated_by.append(agent)
                state.corroborated_finding_ids.add(primary.id)

            # Independent corroboration is real evidence — raise both scores, and
            # more when both sides are live rather than simulated.
            live_bonus = 0.10 if (not a.simulated and not b.simulated) else 0.04
            for finding in (a, b):
                before = finding.relevance
                finding.relevance = round(min(1.0, finding.relevance + live_bonus), 3)
                if finding.relevance > before:
                    boosted += 1

            self._record_collaboration(
                host,
                kind="corroboration",
                initiator=self.profile.key,
                participants=[a.discovered_by, b.discovered_by],
                summary=f"Cross-validated: {_trim(a.title, 70)}",
                detail=(
                    f"{_short(a.discovered_by)} ({a.provider}) and "
                    f"{_short(b.discovered_by)} ({b.provider}) independently reported the same "
                    f"development. Confidence raised by {live_bonus:.2f}; both source links "
                    f"retained."
                ),
                evidence=[u for u in (a.url, b.url) if u],
                confidence_delta=live_bonus,
            )

        logger.collaboration(
            f"{len(corroborated)} development(s) cross-validated across agents",
            f"Independent agents corroborated the same news; confidence raised on "
            f"{boosted} finding(s) and every source link kept.",
            agent=self.profile.key,
            corroborated=len(corroborated),
        )

    def _cross_support(
        self, host: "InsightPulseAgent", by_agent: dict[str, list[FindingRecord]]
    ) -> None:
        """Link a competitive signal to research on the same technology.

        Weaker than corroboration — these are not the same event — but it is a real
        analytical link: it answers "is there published work behind the claim this
        company is making?" No confidence boost is applied, only the linkage.
        """
        commercial = by_agent.get(COMPETITIVE_AGENT.key) or []
        academic = by_agent.get(RESEARCH_AGENT.key) or []
        if not commercial or not academic:
            return

        flagged = [f for f in commercial if set(f.signals) & STRATEGIC_SIGNALS]
        flagged.sort(key=lambda f: -f.relevance)

        # A single shared word is meaningless in general, but a shared *tracked topic*
        # term is exactly the link we are looking for.
        topic_tokens: set[str] = set()
        for term in [*host.state.keywords, *host.state.tracking_topics]:
            topic_tokens |= _tokens(term)

        links = 0
        for claim in flagged[:3]:
            claim_tokens = _tokens(f"{claim.title} {claim.summary}")
            best: FindingRecord | None = None
            best_shared: set[str] = set()
            for paper in academic:
                shared = claim_tokens & _tokens(f"{paper.title} {paper.summary}")
                if len(shared) > len(best_shared):
                    best, best_shared = paper, shared
            if best is None:
                continue
            if len(best_shared) < 2 and not (best_shared & topic_tokens):
                continue

            links += 1
            host.state.corroborated_finding_ids.add(claim.id)
            if RESEARCH_AGENT.key not in claim.corroborated_by:
                claim.corroborated_by.append(RESEARCH_AGENT.key)
            if COMPETITIVE_AGENT.key not in best.corroborated_by:
                best.corroborated_by.append(COMPETITIVE_AGENT.key)

            self._record_collaboration(
                host,
                kind="handoff",
                initiator=self.profile.key,
                participants=[COMPETITIVE_AGENT.key, RESEARCH_AGENT.key],
                summary=f"Technical backing found for: {_trim(claim.title, 66)}",
                detail=(
                    f"The Competitive Intelligence Agent reported this development; the "
                    f"Research Intelligence Agent independently found related published work "
                    f"(\"{_trim(best.title, 70)}\"). Shared technical terms: "
                    f"{', '.join(sorted(best_shared)[:4])}. Linked as supporting context, not "
                    f"as the same event."
                ),
                evidence=[u for u in (claim.url, best.url) if u],
            )

        if links:
            host.logger.collaboration(
                f"{links} competitive signal(s) linked to supporting research",
                "The orchestrator cross-checked each company claim against the research "
                "agent's findings and attached the technical context it found.",
                agent=self.profile.key,
                links=links,
            )

    def _record_collaboration(
        self,
        host: "InsightPulseAgent",
        *,
        kind: str,
        initiator: str,
        participants: list[str],
        summary: str,
        detail: str,
        evidence: list[str] | None = None,
        confidence_delta: float = 0.0,
    ) -> None:
        event = CollaborationEvent(
            run_id=host.state.run_id,
            kind=kind,  # type: ignore[arg-type]
            initiator=initiator,
            participants=[p for p in participants if p],
            summary=summary,
            detail=detail,
            evidence=evidence or [],
            confidence_delta=confidence_delta,
        )
        host.state.collaboration_events.append(event.to_dict())
        if kind != "corroboration":  # corroboration is summarised in one line later
            host.logger.speaking_as(self.profile.key)
            host.logger.collaboration(summary, detail, agent=self.profile.key, kind=kind)

    # ─────────────────────────────────────────────────────────
    # 7. Contributions summary (for the UI, API and report)
    # ─────────────────────────────────────────────────────────
    def contributions(self, host: "InsightPulseAgent") -> list[dict[str, Any]]:
        state = host.state
        out: list[dict[str, Any]] = []

        for key, report in self.reports.items():
            card = report.public()
            card["corroborated"] = sum(
                1 for f in state.findings_by_agent(key) if f.corroborated_by
            )
            out.append(card)

        counts = {}
        for i in state.final_insights:
            counts[i.get("priority", "MEDIUM")] = counts.get(i.get("priority", "MEDIUM"), 0) + 1

        merges = [e for e in state.collaboration_events if e["kind"] == "corroboration"]
        follow_ups = [e for e in state.collaboration_events if e["kind"] == "follow_up"]

        bullets = [
            f"Selected {len([e for e in state.execution_plan if e.get('selected')])} of "
            f"{len(state.execution_plan)} specialists from the goal",
            f"Issued {len(follow_ups)} cross-agent follow-up task(s)",
            f"Cross-validated {len(merges)} development(s) reported by more than one agent",
            f"Prioritized {counts.get('HIGH', 0)} high-priority insight(s) from "
            f"{len(state.findings)} finding(s)",
        ]
        out.append({
            "agent": self.profile.key,
            "name": self.profile.name,
            "icon": self.profile.icon,
            "accent": self.profile.accent,
            "responsibility": self.profile.responsibility,
            "status": "completed",
            "tools_used": [],
            "sources_checked": [],
            "findings_count": 0,
            "coverage": "live",
            "confidence": round(
                sum(r.confidence for r in self.reports.values()) / max(1, len(self.reports)), 3
            ),
            "summary": "; ".join(bullets) + ".",
            "bullets": bullets,
            "observations": [],
        })
        return out


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _short(agent_key: str) -> str:
    return {
        "research_agent": "Research Intelligence Agent",
        "competitive_agent": "Competitive Intelligence Agent",
        "orchestrator": "Orchestrator",
    }.get(agent_key, agent_key or "agent")


def _topics(state: Any) -> str:
    terms = state.keywords or state.tracking_topics
    return ", ".join(terms[:3]) if terms else (state.user_goal or "the tracked topic")


def _trim(text: str, n: int) -> str:
    t = str(text or "")
    return t if len(t) <= n else t[: n - 1] + "…"


def _lead_subject(state: Any, agent_key: str) -> str:
    items = [f for f in state.findings_by_agent(agent_key) if set(f.signals) & STRATEGIC_SIGNALS]
    if not items:
        items = state.findings_by_agent(agent_key)
    items.sort(key=lambda f: -f.relevance)
    return _trim(items[0].title, 110) if items else ""


def _tokens(text: str) -> set[str]:
    return {
        w for w in str(text or "").lower().replace("-", " ").replace(":", " ").split()
        if len(w) > 3 and w.isalpha() and w not in _STOP
    }


def _describes_same_development(a: FindingRecord, b: FindingRecord) -> bool:
    """True when two findings from different agents are about one event.

    Requires a shared anchor (same company, or the same strategic signal) *and*
    meaningful title overlap — either alone produces false positives.
    """
    if a.id == b.id:
        return False

    same_company = bool(a.competitor) and a.competitor.lower() == (b.competitor or "").lower()
    shared_signal = bool(set(a.signals) & set(b.signals) & STRATEGIC_SIGNALS)
    if not (same_company or shared_signal):
        return False

    ta, tb = _tokens(a.title), _tokens(b.title)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= _TITLE_OVERLAP


def _merge_reports(primary: AgentReport, extra: AgentReport) -> AgentReport:
    """Fold a follow-up report into the agent's primary report."""
    primary.findings_count += extra.findings_count
    primary.relevant_count += extra.relevant_count
    primary.finding_ids.extend(extra.finding_ids)
    for attr in (
        "sources_checked", "tools_used", "observations", "signals",
        "research_trends", "key_developments", "competitors_analyzed",
        "market_signals", "errors",
    ):
        merged = list(dict.fromkeys([*getattr(primary, attr), *getattr(extra, attr)]))
        setattr(primary, attr, merged)
    for entry in extra.degraded_providers:
        if entry not in primary.degraded_providers:
            primary.degraded_providers.append(entry)
    primary.duration_ms += extra.duration_ms
    primary.confidence = max(primary.confidence, extra.confidence)
    if extra.status == "failed" and primary.status == "completed":
        primary.status = "partial"
    primary.reasoning_summary = (
        f"{primary.reasoning_summary} Follow-up: {extra.reasoning_summary}"
    ).strip()
    return primary
