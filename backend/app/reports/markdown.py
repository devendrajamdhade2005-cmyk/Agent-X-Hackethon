"""Markdown export — the same report as portable plain text."""

from __future__ import annotations

from typing import Any

BADGE = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def render_markdown(report: dict[str, Any]) -> str:
    st = report.get("summary_stats") or {}
    L: list[str] = []

    L += [
        "# InsightPulse AI — Intelligence Report",
        "",
        "*Autonomous Research & Competitor Intelligence*",
        "",
        f"**Tracking goal:** {report.get('tracking_goal') or '—'}",
        "",
        f"| | |", "|---|---|",
        f"| Generated | {report.get('generated_at_display')} |",
        f"| Report ID | `{report.get('report_id')}` |",
        f"| Run ID | `{report.get('run_id')}` |",
        f"| Status | {report.get('status')} |",
        f"| Reasoning engine | {report.get('reasoner')} |",
    ]
    if report.get("keywords"):
        L.append(f"| Keywords | {', '.join(report['keywords'])} |")
    if report.get("competitors"):
        L.append(f"| Competitors tracked | {', '.join(report['competitors'])} |")
    L.append("")

    if report.get("simulated"):
        L += [
            "> **DATA NOTICE** — Some findings were produced in simulation mode because live "
            "source access was unavailable. Simulated items are labelled and are not verified "
            "real-world data.",
            "",
        ]

    # 01
    L += ["---", "", "## 01 · Executive Summary", "", report.get("executive_summary") or "", ""]
    L += [
        "| Metric | Value |", "|---|---|",
        f"| 🔴 High priority | {st.get('high_priority_count', 0)} |",
        f"| 🟡 Medium priority | {st.get('medium_priority_count', 0)} |",
        f"| 🟢 Low priority | {st.get('low_priority_count', 0)} |",
        f"| Total findings | {st.get('total_findings', 0)} |",
        f"| Relevant findings | {st.get('relevant_findings', 0)} |",
        f"| Tools used | {st.get('tools_used_count', 0)} |",
        "",
    ]
    if report.get("analyst_summary"):
        L += ["### Analyst summary", "", report["analyst_summary"], ""]

    # 02
    L += ["---", "", "## 02 · Prioritized Insights", ""]
    insights = report.get("insights") or []
    if not insights:
        L += ["_No findings cleared the relevance threshold for this goal._", ""]
    for n, i in enumerate(insights, 1):
        pri = i.get("priority", "MEDIUM")
        L.append(f"### {BADGE.get(pri, '•')} {pri} PRIORITY — {i.get('title')}")
        L.append("")
        meta = [f"**Category:** {i.get('category_label')}"]
        if i.get("date"):
            meta.append(f"**Date:** {i['date']}")
        if i.get("competitor"):
            meta.append(f"**Company:** {i['competitor']}")
        if i.get("relevance_score") is not None:
            meta.append(f"**Relevance:** {i['relevance_score']}")
        if i.get("simulated"):
            meta.append("**⚠ simulated data**")
        L += [" · ".join(meta), ""]
        L += [f"**What happened**  \n{i.get('what_happened')}", ""]
        if i.get("summary") and i.get("summary") != i.get("what_happened"):
            L += [f"**Summary**  \n{i.get('summary')}", ""]
        L += [f"**Why it matters**  \n{i.get('why_it_matters')}", ""]
        L += [f"**Recommended action**  \n{i.get('recommended_action')}", ""]
        src = i.get("source_name") or ""
        L += [f"**Source:** {src}" + (f" — <{i['source_url']}>" if i.get("source_url") else ""), ""]
        del n

    # 03 — agent contributions
    ac = report.get("agent_contributions") or {}
    if ac.get("specialists"):
        L += ["---", "", "## 03 · Agent Contributions", "", ac.get("architecture", ""), "",
              "### Orchestrator's agent-selection decisions", ""]
        for e in [*(ac.get("selected") or []), *(ac.get("not_selected") or [])]:
            verdict = "**SELECTED**" if e.get("selected") else "*not selected*"
            L.append(f"- {e.get('icon','')} **{e.get('name')}** — {verdict}  \n  {e.get('reason')}")
        L.append("")
        for a in ac["specialists"]:
            L += [f"### {a.get('icon','')} {a.get('name')}", "",
                  f"*{a.get('responsibility')}*", "",
                  f"| Status | Coverage | Findings | Relevant | Confidence | Cross-validated |",
                  "|---|---|---|---|---|---|",
                  f"| {str(a.get('status','')).upper()} | {str(a.get('coverage','')).upper()} "
                  f"| {a.get('findings_count',0)} | {a.get('relevant_count',0)} "
                  f"| {round(float(a.get('confidence') or 0)*100)}% | {a.get('corroborated',0)} |",
                  "",
                  f"**Tools used:** {', '.join(a.get('tools_used') or []) or 'none'}  ",
                  f"**Providers:** {', '.join(a.get('sources_checked') or []) or 'none'}", "",
                  a.get("summary", ""), ""]
            for key, label in (("research_trends","Recurring themes"),
                               ("key_developments","Key developments"),
                               ("competitors_analyzed","Companies analysed"),
                               ("market_signals","Market signals")):
                if a.get(key):
                    L.append(f"- **{label}:** {', '.join(str(x) for x in a[key][:5])}")
            L.append("")
        orch = ac.get("orchestrator")
        if orch:
            L += [f"### {orch.get('icon','')} {orch.get('name')}", ""]
            for b in orch.get("bullets") or []:
                L.append(f"- {b}")
            L.append("")
        L += [f"### Collaboration events ({ac.get('collaboration_count',0)})", ""]
        if not ac.get("collaboration_events"):
            L.append("_No cross-agent collaboration was required for this goal._")
        for e in (ac.get("collaboration_events") or [])[:8]:
            L.append(f"- **[{str(e.get('kind','')).replace('_',' ').upper()}]** "
                     f"{e.get('summary')}  \n  {e.get('detail')}")
        L.append("")
        revisions = ac.get("plan_revisions") or []
        if len(revisions) > 1:
            L += ["**Plan revisions during the run**", ""]
            for r0 in revisions[1:]:
                L.append(f"- {r0}")
            L.append("")

    # 04 — context & memory
    cm = report.get("context_memory") or {}
    if cm.get("available"):
        tc = cm.get("task_context") or {}
        w = cm.get("working") or {}
        change = cm.get("change") or {}
        cons = cm.get("consolidation") or {}
        L += ["---", "", "## 04 · Context & Memory", "",
              f"Working memory reached version **{w.get('version', 0)}** across "
              f"{w.get('updates', 0)} update(s), retaining {w.get('fact_count', 0)} fact(s) "
              f"of which {w.get('important_fact_count', 0)} were important.", "",
              "**This report is based on**", ""]
        L.append("- Current intelligence gathered in this run")
        if cm.get("used_historical_context"):
            L.append("- Relevant historical context retrieved from previous monitoring")
        L += ["", "### Task context", ""]
        L += [f"- **Topics:** {', '.join(tc.get('topics') or []) or 'none detected'}"]
        if tc.get("competitors"):
            L.append(f"- **Tracked companies:** {', '.join(tc['competitors'])}")
        if tc.get("domains"):
            L.append(f"- **Intelligence domains:** {', '.join(tc['domains'])}")
        L.append(f"- **Time scope:** {tc.get('time_scope', 'unspecified')}")
        if tc.get("constraints"):
            L.append(f"- **Constraints:** {'; '.join(tc['constraints'])}")
        if tc.get("continuation"):
            L.append("- **Continuation** of earlier monitoring")
        L += ["", "### Execution plan state", ""]
        for step in cm.get("plan_steps") or []:
            ref = f" — {step.get('reference')}" if step.get("reference") else ""
            L.append(f"- **{step.get('name')}**: {str(step.get('status','')).replace('_',' ')}{ref}")
        if cm.get("shared_context"):
            L += ["", "### Context shared between agents", ""]
            for sh in cm["shared_context"]:
                L.append(
                    f"- {sh.get('icon','')} **{sh.get('agent')}** received "
                    f"{sh.get('facts', 0)} finding(s) from {', '.join(sh.get('from') or [])}  "
                )
                L.append(f"  Context: {', '.join(sh.get('received') or [])}")
                if sh.get("focus"):
                    L.append(f"  Search focus carried over: {', '.join(sh['focus'])}")
                for why in sh.get("withheld") or []:
                    L.append(f"  _Withheld: {why}_")
        if cm.get("retained_facts"):
            L += ["", "### Findings retained in working memory", ""]
            for fact in cm["retained_facts"]:
                sim = " · SIMULATED" if fact.get("simulated") else ""
                L.append(f"- **[{fact.get('importance')}]** {fact.get('text')}  \n  "
                         f"_{fact.get('agent')}{sim}_")
        L += ["", "### Long-term memory", ""]
        if cm.get("retrieved"):
            for m0 in cm["retrieved"]:
                rel = f" · relevance {m0.get('relevance')}" if m0.get("relevance") else ""
                L.append(f"- **{m0.get('type')}** — {m0.get('summary')}  \n  "
                         f"_from run {m0.get('from_run')}{rel}_")
        else:
            L.append(f"_No relevant previous context was found for this goal "
                     f"({cm.get('retrieval_status')}). This report is based on current "
                     f"intelligence only._")
        L.append("")
        if change.get("compared"):
            L.append(f"**Detected change:** {change.get('verdict')} — {change.get('detail')}")
        else:
            L.append("_No historical baseline was available, so no change comparison "
                     "was made._")
        L += ["", f"Consolidated **{cons.get('stored', 0)}** new item(s) for future "
                  f"monitoring; store holds {cm.get('store_total', 0)} item(s).", ""]
        if w.get("compressions"):
            L += [f"Context compression ran {w['compressions']} time(s), folding "
                  f"{w.get('compressed_count', 0)} lower-importance fact(s) into a summary "
                  f"while keeping important facts verbatim.", ""]

    # 05
    ex = report.get("execution_summary") or {}
    m = ex.get("metrics") or {}
    L += ["---", "", "## 05 · Agent Execution Summary", "", f"`{ex.get('loop')}`", ""]
    L += [
        "| Metric | Value |", "|---|---|",
        f"| Iterations | {m.get('iterations', 0)} of {m.get('max_iterations', 0)} |",
        f"| Tool calls | {m.get('tool_calls', 0)} |",
        f"| Duration | {m.get('duration_seconds', 0)}s |",
        f"| Sources checked | {m.get('sources_checked', 0)} |",
        f"| Relevant findings | {m.get('relevant_findings', 0)} |",
        f"| Duplicates suppressed | {m.get('duplicates_suppressed', 0)} |",
        f"| Errors handled | {m.get('errors_handled', 0)} |",
        "",
        "### Execution trail",
        "",
    ]
    for idx, step in enumerate(ex.get("steps") or [], 1):
        L.append(f"{idx}. **{step.get('stage')}**"
                 + (f"  \n   {step.get('detail')}" if step.get("detail") else ""))
    L.append("")

    # 06
    groups = report.get("findings_by_category") or []
    if groups:
        L += ["---", "", "## 06 · Detailed Findings", ""]
        for g in groups:
            L += [f"### {g.get('label')} ({g.get('count')} finding(s))", ""]
            for f in g.get("items", []):
                bits = [x for x in (
                    f.get("date"), f.get("organization"), f.get("provider_label"),
                    f"relevance {f['relevance']}" if f.get("relevance") is not None else None,
                    "⚠ simulated" if f.get("simulated") else None,
                ) if x]
                line = f"- **{f.get('title')}**"
                if bits:
                    line += f"  \n  _{' · '.join(str(b) for b in bits)}_"
                if f.get("description"):
                    line += f"  \n  {f['description']}"
                if f.get("url"):
                    line += f"  \n  <{f['url']}>"
                L.append(line)
            L.append("")

    # 07
    src = report.get("sources") or {}
    L += ["---", "", "## 07 · Sources & Coverage", "",
          "Every provider queried, who operates it, the endpoint used, and the real "
          "publication domains the findings came from.", "",
          "| Provider | Status | Operator | Access | Endpoint | Findings | Domains harvested |",
          "|---|---|---|---|---|---|---|"]
    for x in src.get("sources_used") or []:
        o = x.get("origin") or {}
        status = "LIVE" if x.get("live") else "SIMULATED"
        doms = ", ".join(f"{h} ({n})" for h, n in (x.get("domains") or [])) or "—"
        L.append(
            f"| {x.get('name')} | {status} | {o.get('operator','?')} | {o.get('auth','?')} "
            f"| `{o.get('endpoint','n/a')}` | {x.get('findings',0)} | {doms} |")
    if src.get("domains"):
        L += ["", f"**All publication domains ({src.get('domain_count',0)} distinct)**", ""]
        L.append(", ".join(f"`{h}` ({n})" for h, n in src["domains"][:40]))
    L += ["", "**Tools called**", ""]
    for t in src.get("tools_used") or [{"name": "—", "findings": 0}]:
        L.append(f"- {t.get('name')} — {t.get('findings')} finding(s)")
    if src.get("coverage"):
        L += ["", "**Coverage by category**", ""]
        for c in src["coverage"]:
            L.append(f"- {c.get('category')}: {c.get('count')}")
    if src.get("degraded"):
        L += ["", "**Providers unavailable during this run**", ""]
        for d in src["degraded"]:
            L.append(f"- {d.get('provider')} — {d.get('reason')}")
    L.append("")

    # 08
    if report.get("caveats"):
        L += ["---", "", "## 08 · Limitations & Caveats", ""]
        for c in report["caveats"]:
            L += [f"**{c.get('title')}**  \n{c.get('body')}", ""]

    L += [
        "---",
        "",
        f"*Report `{report.get('report_id')}` · Run `{report.get('run_id')}` · "
        f"generated {report.get('generated_at_display')} by InsightPulse AI. "
        f"Generated automatically by an autonomous agent — verify material claims against "
        f"the original sources before acting.*",
        "",
    ]
    return "\n".join(L)
