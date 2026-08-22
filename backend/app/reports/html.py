"""Render a report as a standalone, print-ready HTML document.

Deliberately light-themed and self-contained: this is the artefact a user reads
on screen and prints to A4, so it must not inherit the dark product UI, and it
must not depend on any external asset.
"""

from __future__ import annotations

from html import escape
from typing import Any

PRIORITY_STYLE = {
    "HIGH": ("#b42318", "#fef3f2", "#fecdca"),
    "MEDIUM": ("#b54708", "#fffaeb", "#fedf89"),
    "LOW": ("#067647", "#ecfdf3", "#abefc6"),
}

CATEGORY_ICON = {
    "patent": "◈",
    "research": "◇",
    "competitor": "◆",
    "news": "▣",
    "web": "◉",
}


def _e(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def render_html(report: dict[str, Any], *, embedded: bool = False) -> str:
    """`embedded=True` drops the print button (used inside the preview iframe)."""
    stats = report.get("summary_stats") or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_e(report.get('report_id'))} — InsightPulse Intelligence Report</title>
<!-- Every source link opens in a new tab. Without this, a click inside the
     preview iframe navigates the iframe itself, and most publishers send
     X-Frame-Options: DENY, so the reader just sees "refused to connect". -->
<base target="_blank" />
<style>{_CSS}</style>
</head>
<body>
{"" if embedded else _print_bar(report)}
<article class="doc">
  {_cover(report, stats)}
  {_exec_summary(report, stats)}
  {_insights(report)}
  {_agents(report)}
  {_memory(report)}
  {_execution(report)}
  {_findings(report)}
  {_sources(report)}
  {_caveats(report)}
  {_footer(report)}
</article>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
def _print_bar(report: dict[str, Any]) -> str:
    return f"""<div class="printbar">
  <span>Report {_e(report.get('report_id'))}</span>
  <button type="button" onclick="window.print()">Print / Save as PDF</button>
</div>"""


def _cover(report: dict[str, Any], stats: dict[str, Any]) -> str:
    goal = _e(report.get("tracking_goal"))
    keywords = report.get("keywords") or []
    competitors = report.get("competitors") or []

    chips = ""
    if keywords:
        chips += f"""<div class="metarow"><span class="metalabel">Keywords</span>
          <span class="chips">{"".join(f'<span class="chip">{_e(k)}</span>' for k in keywords)}</span></div>"""
    if competitors:
        chips += f"""<div class="metarow"><span class="metalabel">Competitors tracked</span>
          <span class="chips">{"".join(f'<span class="chip chip-co">{_e(c)}</span>' for c in competitors)}</span></div>"""

    sim = ""
    if report.get("simulated"):
        sim = """<div class="notice">
          <strong>Data notice</strong> Some findings in this report were produced in
          simulation mode and are labelled <em>simulated</em>. They are not verified
          real-world data.</div>"""

    return f"""<header class="cover">
  <div class="brand">
    <span class="mark">IP</span>
    <span class="brandtext">
      <strong>InsightPulse AI</strong>
      <small>Autonomous Research &amp; Competitor Intelligence</small>
    </span>
  </div>

  <h1>Intelligence Report</h1>

  <div class="goalbox">
    <span class="metalabel">Tracking goal</span>
    <p class="goal">{goal or "—"}</p>
    {chips}
  </div>

  <div class="idgrid">
    <div><span class="metalabel">Generated</span><b>{_e(report.get('generated_at_display'))}</b></div>
    <div><span class="metalabel">Report ID</span><b class="mono">{_e(report.get('report_id'))}</b></div>
    <div><span class="metalabel">Run ID</span><b class="mono">{_e(report.get('run_id'))}</b></div>
    <div><span class="metalabel">Status</span><b class="ok">{_e(report.get('status'))}</b></div>
  </div>
  {sim}
</header>"""


def _exec_summary(report: dict[str, Any], stats: dict[str, Any]) -> str:
    tiles = [
        ("High priority", stats.get("high_priority_count", 0), "high"),
        ("Medium priority", stats.get("medium_priority_count", 0), "med"),
        ("Low priority", stats.get("low_priority_count", 0), "low"),
        ("Total findings", stats.get("total_findings", 0), ""),
        ("Tools used", stats.get("tools_used_count", 0), ""),
    ]
    tile_html = "".join(
        f'<div class="tile {cls}"><b>{_e(v)}</b><span>{_e(label)}</span></div>'
        for label, v, cls in tiles
    )

    analyst = report.get("analyst_summary") or ""
    analyst_html = ""
    if analyst:
        paras = "".join(f"<p>{_e(p)}</p>" for p in analyst.split("\n\n") if p.strip())
        analyst_html = f'<div class="analyst"><span class="metalabel">Analyst summary</span>{paras}</div>'

    return f"""<section class="sec">
  <h2><span class="num">01</span> Executive Summary</h2>
  <p class="lede">{_e(report.get('executive_summary'))}</p>
  <div class="tiles">{tile_html}</div>
  {analyst_html}
</section>"""


def _insights(report: dict[str, Any]) -> str:
    insights = report.get("insights") or []
    if not insights:
        return """<section class="sec"><h2><span class="num">02</span> Prioritized Insights</h2>
        <p class="empty">No findings cleared the relevance threshold for this goal.</p></section>"""

    cards = []
    for n, i in enumerate(insights, 1):
        pri = i.get("priority", "MEDIUM")
        fg, bg, border = PRIORITY_STYLE.get(pri, PRIORITY_STYLE["MEDIUM"])
        icon = CATEGORY_ICON.get(i.get("category", ""), "▣")

        bits = [f'<span class="cat">{icon} {_e(i.get("category_label"))}</span>']
        if i.get("date"):
            bits.append(f'<span>{_e(i["date"])}</span>')
        if i.get("competitor"):
            bits.append(f'<span class="co">{_e(i["competitor"])}</span>')
        if i.get("relevance_score") is not None:
            bits.append(f'<span>relevance {_e(i["relevance_score"])}</span>')
        if i.get("simulated"):
            bits.append('<span class="simtag">simulated</span>')

        # Lead with the actual publisher, then the provider that surfaced it.
        host = _host(i.get("source_url") or "")
        src = (
            f'<b>{_e(host)}</b> — retrieved via {_e(i.get("source_name"))}'
            if host else _e(i.get("source_name"))
        )
        if i.get("source_url"):
            src += f'<br><a href="{_e(i["source_url"])}">{_e(i["source_url"])}</a>'
        elif i.get("simulated"):
            src += ' · <span class="simtag">simulated — no live URL</span>'

        cards.append(f"""<article class="card">
  <div class="cardtop">
    <span class="badge" style="color:{fg};background:{bg};border-color:{border}">{_e(pri)} PRIORITY</span>
    <span class="cardno">#{n}</span>
  </div>
  <h3>{_e(i.get('title'))}</h3>
  <div class="cardmeta">{" · ".join(bits)}</div>

  <div class="field"><span class="flabel">What happened</span><p>{_e(i.get('what_happened'))}</p></div>
  {f'<div class="field"><span class="flabel">Summary</span><p>{_e(i.get("summary"))}</p></div>' if i.get('summary') and i.get('summary') != i.get('what_happened') else ''}
  <div class="field why"><span class="flabel">Why it matters</span><p>{_e(i.get('why_it_matters'))}</p></div>
  <div class="field action"><span class="flabel">Recommended action</span><p>{_e(i.get('recommended_action'))}</p></div>
  <div class="src"><span class="flabel">Source</span><p>{src}</p></div>
</article>""")

    return f"""<section class="sec">
  <h2><span class="num">02</span> Prioritized Insights</h2>
  <p class="sub">{len(insights)} insight(s), ordered by priority then relevance.</p>
  {"".join(cards)}
</section>"""


_COVERAGE_TAG = {
    "live": ('<span class="livetag">LIVE</span>', ""),
    "partial": ('<span class="parttag">PARTIAL COVERAGE</span>', ""),
    "simulated": ('<span class="simtag">SIMULATED</span>', ""),
    "unavailable": ('<span class="unavailtag">UNAVAILABLE</span>', ""),
}


def _agents(report: dict[str, Any]) -> str:
    ac = report.get("agent_contributions") or {}
    specialists = ac.get("specialists") or []
    if not specialists:
        return ""

    orchestrator = ac.get("orchestrator") or {}
    events = ac.get("collaboration_events") or []

    cards = []
    for a in specialists:
        cov, _ = _COVERAGE_TAG.get(a.get("coverage", "live"), ("", ""))
        extras = []
        if a.get("research_trends"):
            extras.append(("Recurring themes", ", ".join(a["research_trends"][:4])))
        if a.get("key_developments"):
            extras.append(("Key developments", "; ".join(
                _shorten_text(d, 70) for d in a["key_developments"][:2])))
        if a.get("competitors_analyzed"):
            extras.append(("Companies analysed", ", ".join(a["competitors_analyzed"][:5])))
        if a.get("market_signals"):
            extras.append(("Market signals", ", ".join(a["market_signals"][:4])))
        if a.get("degraded_providers"):
            extras.append(("Degraded providers", ", ".join(
                d.get("provider", "") for d in a["degraded_providers"])))

        cards.append(f"""<div class="agentcard">
  <div class="agenthead">
    <span class="agentname">{_e(a.get('icon'))} {_e(a.get('name'))}</span>
    <span class="agentstate">{_e(str(a.get('status','')).upper())} {cov}</span>
  </div>
  <p class="agentresp">{_e(a.get('responsibility'))}</p>
  <div class="agentstats">
    <span><b>{_e(a.get('findings_count'))}</b> findings</span>
    <span><b>{_e(a.get('relevant_count'))}</b> relevant</span>
    <span><b>{_e(round(float(a.get('confidence') or 0) * 100))}%</b> confidence</span>
    <span><b>{_e(a.get('corroborated', 0))}</b> cross-validated</span>
  </div>
  <p class="agentmeta"><b>Tools used:</b> {_e(', '.join(a.get('tools_used') or []) or 'none')}
     &nbsp;·&nbsp; <b>Providers:</b> {_e(', '.join(a.get('sources_checked') or []) or 'none')}</p>
  <p class="agentsummary">{_e(a.get('summary'))}</p>
  {"".join(f'<p class="agentextra"><b>{_e(k)}:</b> {_e(v)}</p>' for k, v in extras)}
</div>""")

    plan_rows = "".join(
        f"""<li class="{'psel' if p.get('selected') else 'pskip'}">
          <b>{_e(p.get('icon'))} {_e(p.get('name'))}</b>
          <span class="pverdict">{'SELECTED' if p.get('selected') else 'NOT SELECTED'}</span>
          <span class="preason">{_e(p.get('reason'))}</span></li>"""
        for p in [*(ac.get("selected") or []), *(ac.get("not_selected") or [])]
    )

    event_rows = "".join(
        f"""<li><span class="ekind">{_e(str(e.get('kind','')).replace('_',' ').upper())}</span>
          <b>{_e(e.get('summary'))}</b>
          <span class="edetail">{_e(e.get('detail'))}</span></li>"""
        for e in events[:8]
    ) or '<li><span class="nodata">No cross-agent collaboration was required for this goal.</span></li>'

    revisions = ac.get("plan_revisions") or []
    rev_html = ""
    if len(revisions) > 1:  # the first entry is always "initial plan"
        rev_html = f"""<div class="revisions">
          <span class="metalabel">Plan revisions during the run</span>
          <ul>{"".join(f'<li>{_e(r)}</li>' for r in revisions[1:])}</ul>
        </div>"""

    return f"""<section class="sec">
  <h2><span class="num">03</span> Agent Contributions</h2>
  <p class="sub">{_e(ac.get('architecture'))}</p>

  <div class="planbox">
    <span class="metalabel">Orchestrator's agent-selection decisions</span>
    <ul class="planlist">{plan_rows}</ul>
  </div>

  <div class="agentgrid">{"".join(cards)}</div>

  {f'''<div class="orchcard">
    <div class="agenthead">
      <span class="agentname">{_e(orchestrator.get('icon'))} {_e(orchestrator.get('name'))}</span>
      <span class="agentstate">COORDINATOR</span>
    </div>
    <ul class="orchbullets">
      {"".join(f'<li>{_e(b)}</li>' for b in (orchestrator.get('bullets') or []))}
    </ul>
  </div>''' if orchestrator else ""}

  <div class="collabbox">
    <span class="metalabel">Collaboration events ({_e(ac.get('collaboration_count'))})</span>
    <ul class="collablist">{event_rows}</ul>
  </div>
  {rev_html}
</section>"""


def _memory(report: dict[str, Any]) -> str:
    """04 — Context & Memory.

    Built imperatively rather than as one large f-string: every claim here is
    conditional on something having actually happened, and nested conditionals
    inside a template are where "3 memories retrieved" appears on a run that
    retrieved none.
    """
    cm = report.get("context_memory") or {}
    if not cm.get("available"):
        return ""

    tc = cm.get("task_context") or {}
    w = cm.get("working") or {}
    change = cm.get("change") or {}
    cons = cm.get("consolidation") or {}

    # What the narrative is allowed to claim it drew on.
    basis = ["Current intelligence gathered in this run"]
    if cm.get("used_historical_context"):
        basis.append("Relevant historical context retrieved from previous monitoring")
    basis_html = "".join(f"<li>{_e(b)}</li>" for b in basis)

    chips = "".join(
        f'<span class="memchip">{_e(v)}</span>'
        for v in [*tc.get("topics", []), *tc.get("competitors", []), *tc.get("domains", [])]
    ) or '<span class="memnone">no explicit topic detected</span>'

    scope_bits = [f"Time scope: {_e(tc.get('time_scope'))}"]
    if tc.get("constraints"):
        scope_bits.append("Constraints: " + _e("; ".join(tc["constraints"])))
    if tc.get("continuation"):
        scope_bits.append("continuation of earlier monitoring")
    scope = " · ".join(scope_bits)

    steps = []
    for step in cm.get("plan_steps") or []:
        ref = f" ({_e(step.get('reference'))})" if step.get("reference") else ""
        status = _e((step.get("status") or "").replace("_", " "))
        steps.append(f"<li><b>{_e(step.get('name'))}</b> — {status}{ref}</li>")
    steps_html = "".join(steps)

    shared = []
    for sh in cm.get("shared_context") or []:
        parts = [
            f"<b>{_e(sh.get('icon'))} {_e(sh.get('agent'))}</b> received "
            f"{_e(sh.get('facts'))} finding(s) from {_e(', '.join(sh.get('from') or []))}.",
            f"<span class=\"memsub\">Context: {_e(', '.join(sh.get('received') or []))}</span>",
        ]
        if sh.get("focus"):
            parts.append(
                "<span class=\"memsub\">Search focus carried over: "
                f"{_e(', '.join(sh['focus']))}</span>"
            )
        for why in sh.get("withheld") or []:
            parts.append(f'<span class="memomit">Withheld: {_e(why)}</span>')
        shared.append("<li>" + "".join(parts) + "</li>")
    shared_html = ""
    if shared:
        shared_html = (
            '<div class="planbox"><span class="metalabel">Context shared between agents'
            '</span><ul class="memlist">' + "".join(shared) + "</ul></div>"
        )

    facts = []
    for fact in cm.get("retained_facts") or []:
        sim = " · SIMULATED" if fact.get("simulated") else ""
        facts.append(
            f'<li><span class="memimp">{_e(fact.get("importance"))}</span> '
            f'{_e(fact.get("text"))}'
            f'<span class="memsub">{_e(fact.get("agent"))}{sim}</span></li>'
        )
    facts_html = ""
    if facts:
        facts_html = (
            '<div class="planbox"><span class="metalabel">Findings retained in working '
            'memory</span><ul class="memlist">' + "".join(facts) + "</ul></div>"
        )

    retrieved = cm.get("retrieved") or []
    if retrieved:
        rows = []
        for m in retrieved:
            rel = f" · relevance {_e(m.get('relevance'))}" if m.get("relevance") else ""
            rows.append(
                f'<li><span class="memtype">{_e(m.get("type"))}</span> '
                f'{_e(m.get("summary"))}'
                f'<span class="memsub">from run {_e(m.get("from_run"))}{rel}</span></li>'
            )
        retrieved_html = '<ul class="memlist">' + "".join(rows) + "</ul>"
    else:
        retrieved_html = (
            '<p class="memnone">No relevant previous context was found for this goal '
            f'({_e(cm.get("retrieval_status"))}). This report is based on current '
            "intelligence only.</p>"
        )

    if change.get("compared"):
        change_html = (
            f'<p class="memchange"><b>Detected change:</b> {_e(change.get("verdict"))} — '
            f'{_e(change.get("detail"))}</p>'
        )
    else:
        change_html = (
            '<p class="memnone">No historical baseline was available, so no change '
            "comparison was made.</p>"
        )

    cons_bits = [f"Consolidated {_e(cons.get('stored', 0))} new item(s) for future monitoring"]
    if cons.get("refreshed"):
        cons_bits.append(f"refreshed {_e(cons.get('refreshed'))}")
    if cons.get("rejected"):
        cons_bits.append(f"rejected {_e(cons.get('rejected'))} as not durable")
    cons_line = ", ".join(cons_bits) + "."
    if cm.get("store_total") is not None:
        cons_line += f" Store holds {_e(cm.get('store_total'))} item(s)."

    compression_html = ""
    if w.get("compressions"):
        compression_html = (
            f'<p class="memsub">Context compression ran {_e(w.get("compressions"))} time(s), '
            f'folding {_e(w.get("compressed_count"))} lower-importance fact(s) into a summary '
            "while keeping important facts verbatim.</p>"
        )

    return f"""<section class="sec">
  <h2><span class="num">04</span> Context &amp; Memory</h2>
  <p class="sub">
    Working memory reached version {_e(w.get('version'))} across {_e(w.get('updates'))}
    update(s), retaining {_e(w.get('fact_count'))} fact(s) of which
    {_e(w.get('important_fact_count'))} were important. Task context was extracted by the
    {_e(tc.get('author'))} reader.
  </p>
  <div class="planbox">
    <span class="metalabel">This report is based on</span>
    <ul class="planlist">{basis_html}</ul>
  </div>
  <div class="memgrid">
    <div>
      <span class="metalabel">Task context</span>
      <div class="memchips">{chips}</div>
      <p class="memsub">{scope}</p>
    </div>
    <div>
      <span class="metalabel">Execution plan state</span>
      <ul class="memlist">{steps_html}</ul>
    </div>
  </div>
  {shared_html}
  {facts_html}
  <div class="collabbox">
    <span class="metalabel">Long-term memory</span>
    {retrieved_html}
    {change_html}
    <p class="memsub">{cons_line}</p>
  </div>
  {compression_html}
</section>"""


def _shorten_text(text: str, n: int) -> str:
    t = str(text or "")
    return t if len(t) <= n else t[: n - 1] + "…"


def _execution(report: dict[str, Any]) -> str:
    ex = report.get("execution_summary") or {}
    steps = ex.get("steps") or []
    m = ex.get("metrics") or {}

    rows = "".join(
        f"""<li><span class="stage">{_e(s.get('stage'))}</span>
        {f'<span class="sdetail">{_e(s.get("detail"))}</span>' if s.get("detail") else ""}</li>"""
        for s in steps
    )

    metrics = [
        ("Iterations", f"{m.get('iterations', 0)} of {m.get('max_iterations', 0)}"),
        ("Tool calls", m.get("tool_calls", 0)),
        ("Duration", f"{m.get('duration_seconds', 0)}s"),
        ("Sources checked", m.get("sources_checked", 0)),
        ("Relevant findings", m.get("relevant_findings", 0)),
        ("Duplicates suppressed", m.get("duplicates_suppressed", 0)),
        ("Errors handled", m.get("errors_handled", 0)),
        ("Reasoning engine", m.get("reasoner", "—")),
    ]
    mhtml = "".join(
        f'<div class="mrow"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in metrics
    )

    return f"""<section class="sec">
  <h2><span class="num">05</span> Agent Execution Summary</h2>
  <p class="loop">{_e(ex.get('loop'))}</p>
  <div class="exwrap">
    <ol class="trail">{rows}</ol>
    <aside class="metrics">{mhtml}</aside>
  </div>
</section>"""


def _findings(report: dict[str, Any]) -> str:
    groups = report.get("findings_by_category") or []
    if not groups:
        return ""

    blocks = []
    for g in groups:
        icon = CATEGORY_ICON.get(g.get("category", ""), "▣")
        rows = []
        for f in g.get("items", []):
            meta = []
            if f.get("date"):
                meta.append(_e(f["date"]))
            if f.get("organization"):
                meta.append(_e(f["organization"]))
            host = _host(f.get("url") or "")
            if host:
                meta.append(f"<b>{_e(host)}</b>")
            if f.get("provider_label"):
                meta.append(f"via {_e(f['provider_label'])}")
            if f.get("relevance") is not None:
                meta.append(f'relevance {_e(f["relevance"])}')
            if f.get("simulated"):
                meta.append('<span class="simtag">simulated</span>')

            link = (
                f'<a href="{_e(f["url"])}">{_e(_shorten(f["url"]))}</a>' if f.get("url") else "—"
            )
            rows.append(f"""<tr>
  <td><b>{_e(f.get('title'))}</b>
      {f'<p class="fdesc">{_e(f.get("description"))}</p>' if f.get("description") else ""}
      <p class="fmeta">{" · ".join(meta)}</p></td>
  <td class="fsrc">{link}</td>
</tr>""")

        blocks.append(f"""<div class="fgroup">
  <h3>{icon} {_e(g.get('label'))} <span class="gcount">{_e(g.get('count'))} finding(s)</span></h3>
  <table class="ftable"><tbody>{"".join(rows)}</tbody></table>
</div>""")

    return f"""<section class="sec">
  <h2><span class="num">06</span> Detailed Findings</h2>
  <p class="sub">Every item the agent collected and judged relevant, grouped by source category.</p>
  {"".join(blocks)}
</section>"""


def _sources(report: dict[str, Any]) -> str:
    s = report.get("sources") or {}
    used = s.get("sources_used") or []
    tools = s.get("tools_used") or []
    coverage = s.get("coverage") or []
    degraded = s.get("degraded") or []
    domains = s.get("domains") or []

    # Full audit table: operator, endpoint, auth and the real domains harvested.
    rows = []
    for x in used:
        origin = x.get("origin") or {}
        state = (
            '<span class="livetag">LIVE</span>' if x.get("live")
            else '<span class="simtag">SIMULATED</span>'
        )
        doms = x.get("domains") or []
        dom_html = (
            " ".join(f'<code>{_e(h)}</code><span class="dcount">{n}</span>' for h, n in doms)
            if doms else '<span class="nodata">no live URLs captured</span>'
        )
        rows.append(f"""<tr>
  <td>
    <b>{_e(x.get('name'))}</b> {state}
    <p class="pmeta">{_e(origin.get('what') or '')}</p>
    <p class="pmeta">Operated by {_e(origin.get('operator') or 'unknown')} ·
       access: {_e(origin.get('auth') or 'unknown')}</p>
    <p class="pendpoint"><code>{_e(origin.get('endpoint') or 'n/a')}</code></p>
  </td>
  <td class="pdomains">
    <span class="metalabel">Domains harvested</span>
    <div>{dom_html}</div>
  </td>
  <td class="pcount"><b>{_e(x.get('findings'))}</b><span>findings</span></td>
</tr>""")

    tool_html = "".join(
        f'<li>{_e(t.get("name"))} <span class="gcount">{_e(t.get("findings"))} finding(s)</span></li>'
        for t in tools
    ) or "<li>—</li>"

    cov_html = "".join(
        f'<div class="mrow"><span>{_e(c.get("category"))}</span><b>{_e(c.get("count"))}</b></div>'
        for c in coverage
    )

    dom_summary = ""
    if domains:
        chips = " ".join(
            f'<span class="domchip"><code>{_e(h)}</code>{n}</span>' for h, n in domains[:28]
        )
        dom_summary = f"""<div class="domsum">
          <span class="metalabel">All publication domains in this report
            ({_e(s.get('domain_count'))} distinct)</span>
          <div class="domchips">{chips}</div>
        </div>"""

    deg_html = ""
    if degraded:
        items = "".join(
            f'<li>{_e(d.get("provider"))} — {_e(d.get("reason"))}</li>' for d in degraded
        )
        deg_html = f"""<div class="degraded">
          <span class="metalabel">Providers unavailable during this run</span>
          <ul>{items}</ul></div>"""

    return f"""<section class="sec">
  <h2><span class="num">07</span> Sources &amp; Coverage</h2>
  <p class="sub">Every provider queried, who operates it, the exact endpoint used, and the
     real publication domains the findings came from.</p>

  <table class="ptable"><tbody>{"".join(rows) or '<tr><td>—</td></tr>'}</tbody></table>

  {dom_summary}

  <div class="twocol" style="margin-top:16px">
    <div><span class="metalabel">Tools called by the agent</span><ul class="plain">{tool_html}</ul></div>
    {f'<div><span class="metalabel">Coverage by category</span>{cov_html}</div>' if cov_html else "<div></div>"}
  </div>
  {deg_html}
</section>"""


def _caveats(report: dict[str, Any]) -> str:
    caveats = report.get("caveats") or []
    if not caveats:
        return ""
    items = "".join(
        f'<div class="caveat"><b>{_e(c.get("title"))}</b><p>{_e(c.get("body"))}</p></div>'
        for c in caveats
    )
    return f"""<section class="sec">
  <h2><span class="num">08</span> Limitations &amp; Caveats</h2>
  {items}
</section>"""


def _footer(report: dict[str, Any]) -> str:
    return f"""<footer class="docfoot">
  <div>
    <b>InsightPulse AI</b> — Autonomous Research &amp; Competitor Intelligence<br />
    Report {_e(report.get('report_id'))} · Run {_e(report.get('run_id'))} ·
    Generated {_e(report.get('generated_at_display'))}
  </div>
  <p class="disclaimer">
    Generated automatically by an autonomous agent from the sources listed in section 05.
    Verify material claims against the original sources before acting.
  </p>
</footer>"""


def _shorten(url: str, limit: int = 58) -> str:
    text = str(url or "").replace("https://", "").replace("http://", "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _host(url: str) -> str:
    """Publication domain, e.g. 'wsj.com' — the source a reader actually cares about."""
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(url)).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


# ─────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box}
:root{
  --ink:#101828; --ink2:#475467; --ink3:#667085;
  --line:#e4e7ec; --line2:#f2f4f7;
  --accent:#5b4bd6; --accent-bg:#f4f3ff;
  --high:#b42318; --med:#b54708; --low:#067647;
}
html,body{margin:0;padding:0;background:#f2f4f7;color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
  font-size:11pt;line-height:1.62;-webkit-print-color-adjust:exact;print-color-adjust:exact}
h1,h2,h3{margin:0;line-height:1.25;letter-spacing:-.01em}
p{margin:0}
a{color:var(--accent);word-break:break-word}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}

.printbar{position:sticky;top:0;z-index:5;display:flex;align-items:center;
  justify-content:space-between;gap:12px;padding:10px 18px;background:#101828;color:#fff;
  font-size:10.5pt}
.printbar button{font:inherit;font-weight:600;cursor:pointer;padding:7px 14px;border:none;
  border-radius:7px;background:#5b4bd6;color:#fff}
.printbar button:hover{filter:brightness(1.1)}

.doc{max-width:210mm;margin:20px auto;background:#fff;padding:22mm 20mm;
  box-shadow:0 2px 18px rgba(16,24,40,.10)}

/* cover */
.cover{padding-bottom:20px;border-bottom:2px solid var(--ink)}
.brand{display:flex;align-items:center;gap:11px;margin-bottom:26px}
.mark{width:34px;height:34px;border-radius:8px;background:var(--accent);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12pt;
  letter-spacing:-.02em}
.brandtext{display:flex;flex-direction:column}
.brandtext strong{font-size:11.5pt}
.brandtext small{color:var(--ink3);font-size:8.5pt;letter-spacing:.03em;text-transform:uppercase}
.cover h1{font-size:27pt;font-weight:680;margin-bottom:20px}

.metalabel{display:block;font-size:7.6pt;font-weight:700;letter-spacing:.10em;
  text-transform:uppercase;color:var(--ink3);margin-bottom:4px}
.goalbox{background:var(--accent-bg);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:13px 16px;margin-bottom:18px}
.goal{font-size:13pt;font-weight:560;line-height:1.45}
.metarow{margin-top:10px}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{display:inline-block;font-size:8.8pt;padding:2px 9px;border-radius:999px;
  background:#fff;border:1px solid #d9d6fe;color:#4a3fbf}
.chip-co{background:#101828;border-color:#101828;color:#fff}

.idgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.idgrid b{font-size:9.8pt;font-weight:600}
.idgrid .ok{color:var(--low)}

.notice{margin-top:16px;background:#fffaeb;border:1px solid #fedf89;border-radius:8px;
  padding:10px 13px;font-size:9.5pt;color:#93370d}
.notice strong{display:block;margin-bottom:2px}

/* sections */
.sec{margin-top:30px;page-break-inside:auto}
.sec h2{font-size:14pt;padding-bottom:8px;border-bottom:1px solid var(--line);
  margin-bottom:14px;display:flex;align-items:baseline;gap:10px}
.num{font-size:9pt;font-weight:700;color:var(--accent);letter-spacing:.08em}
.sub{font-size:9.5pt;color:var(--ink3);margin-bottom:14px}
.lede{font-size:11.5pt;line-height:1.68;color:var(--ink2)}
.empty{font-size:10pt;color:var(--ink3);font-style:italic}

.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-top:18px}
.tile{background:var(--line2);border-radius:8px;padding:11px 8px;text-align:center}
.tile b{display:block;font-size:17pt;font-weight:680;line-height:1.2}
.tile span{font-size:7.6pt;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3)}
.tile.high b{color:var(--high)} .tile.med b{color:var(--med)} .tile.low b{color:var(--low)}

.analyst{margin-top:18px;padding-top:14px;border-top:1px solid var(--line2)}
.analyst p{font-size:10pt;color:var(--ink2);margin-bottom:8px}

/* insight cards */
.card{border:1px solid var(--line);border-radius:10px;padding:15px 17px;margin-bottom:13px;
  page-break-inside:avoid;break-inside:avoid}
.cardtop{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}
.badge{font-size:7.8pt;font-weight:750;letter-spacing:.09em;padding:3px 10px;
  border-radius:999px;border:1px solid}
.cardno{font-size:8.5pt;color:#98a2b3;font-weight:600}
.card h3{font-size:12.5pt;margin-bottom:7px}
.cardmeta{font-size:8.8pt;color:var(--ink3);margin-bottom:12px}
.cardmeta .cat{font-weight:650;color:var(--ink2)}
.cardmeta .co{background:#101828;color:#fff;padding:1px 7px;border-radius:999px}
.simtag{background:#fffaeb;border:1px solid #fedf89;color:#93370d;padding:1px 6px;
  border-radius:4px;font-size:7.8pt;font-weight:650}
.livetag{background:#ecfdf3;border:1px solid #abefc6;color:#067647;padding:1px 6px;
  border-radius:4px;font-size:7.8pt;font-weight:650}

.flabel{display:block;font-size:7.4pt;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink3);margin-bottom:3px}
.field{margin-bottom:10px}
.field p{font-size:10pt;color:var(--ink2)}
.field.why{background:var(--line2);border-radius:7px;padding:10px 12px}
.field.action{border-left:2px solid var(--accent);padding-left:11px}
.field.action p{color:var(--ink);font-weight:520}
.src{padding-top:9px;border-top:1px solid var(--line2)}
.src p{font-size:9pt;color:var(--ink3)}

/* execution */
.loop{font-size:9.2pt;font-weight:600;color:var(--accent);background:var(--accent-bg);
  border-radius:7px;padding:9px 12px;margin-bottom:16px;text-align:center;letter-spacing:.01em}
.exwrap{display:grid;grid-template-columns:1fr 178px;gap:20px;align-items:start}
.trail{list-style:none;margin:0;padding:0;counter-reset:t}
.trail li{position:relative;padding:0 0 13px 24px;counter-increment:t;page-break-inside:avoid}
.trail li::before{content:counter(t);position:absolute;left:0;top:1px;width:16px;height:16px;
  border-radius:50%;background:var(--accent);color:#fff;font-size:7.6pt;font-weight:700;
  display:flex;align-items:center;justify-content:center}
.trail li:not(:last-child)::after{content:"";position:absolute;left:7.5px;top:19px;bottom:2px;
  width:1px;background:var(--line)}
.stage{display:block;font-size:9.8pt;font-weight:640}
.sdetail{display:block;font-size:9pt;color:var(--ink3);margin-top:1px}
.metrics{background:var(--line2);border-radius:9px;padding:12px 14px}
.mrow{display:flex;justify-content:space-between;gap:10px;font-size:9pt;
  padding:4px 0;border-bottom:1px solid var(--line)}
.mrow:last-child{border-bottom:none}
.mrow span{color:var(--ink3)}
.mrow b{font-weight:650;text-align:right}

/* findings */
.fgroup{margin-bottom:20px}
.fgroup h3{font-size:11pt;margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid var(--line2)}
.gcount{font-size:8.5pt;font-weight:500;color:var(--ink3)}
.ftable{width:100%;border-collapse:collapse}
.ftable td{padding:9px 0;border-bottom:1px solid var(--line2);vertical-align:top;
  page-break-inside:avoid}
.ftable td b{font-size:10pt;font-weight:600}
.fdesc{font-size:9.2pt;color:var(--ink2);margin-top:2px}
.fmeta{font-size:8.6pt;color:var(--ink3);margin-top:3px}
.fsrc{width:34%;padding-left:14px !important;font-size:8.8pt;text-align:right}

/* agent contributions */
.memgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0}
.memchips{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0}
.memchip{font-size:9.5pt;font-weight:600;padding:2px 7px;border:1px solid #d8dbe6;border-radius:9px}
.memlist{margin:6px 0 0;padding-left:16px}
.memlist li{font-size:9.5pt;margin-bottom:5px;line-height:1.45}
.memsub{display:block;font-size:8.5pt;color:#6b7280;margin-top:2px}
.memomit{display:block;font-size:8.5pt;color:#9aa1ad;margin-top:2px}
.memtype{font-size:8pt;font-weight:700;text-transform:uppercase;color:#8b5cf6;margin-right:5px}
.memimp{font-size:8pt;font-weight:700;color:#0f9b6c;margin-right:5px}
.memnone{font-size:9.5pt;color:#6b7280;margin:6px 0}
.memchange{font-size:9.5pt;margin:8px 0 0}
.planbox{background:var(--line2);border-radius:8px;padding:12px 14px;margin-bottom:14px}
.planlist{list-style:none;margin:6px 0 0;padding:0}
.planlist li{padding:6px 0;border-bottom:1px solid var(--line);font-size:9.4pt}
.planlist li:last-child{border-bottom:none}
.planlist b{font-size:10pt;margin-right:8px}
.pverdict{font-size:7.4pt;font-weight:750;letter-spacing:.06em;padding:1px 7px;border-radius:3px}
.psel .pverdict{background:#ecfdf3;color:#067647}
.pskip .pverdict{background:#f2f4f7;color:#667085}
.preason{display:block;color:var(--ink3);font-size:8.8pt;margin-top:2px}

.agentgrid{display:grid;gap:11px;margin-bottom:14px}
.agentcard,.orchcard{border:1px solid var(--line);border-radius:9px;padding:13px 15px;
  page-break-inside:avoid;break-inside:avoid}
.orchcard{background:#f4f3ff;border-color:#d9d6fe;margin-bottom:14px}
.agenthead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:5px}
.agentname{font-size:11pt;font-weight:680}
.agentstate{font-size:7.6pt;font-weight:750;letter-spacing:.06em;color:var(--ink3)}
.agentresp{font-size:8.8pt;color:var(--ink3);margin-bottom:8px}
.agentstats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:7px;font-size:8.6pt;color:var(--ink3)}
.agentstats b{font-size:12pt;color:var(--ink);margin-right:3px}
.agentmeta{font-size:8.6pt;color:var(--ink3);margin-bottom:6px}
.agentsummary{font-size:9.4pt;color:var(--ink2)}
.agentextra{font-size:8.8pt;color:var(--ink3);margin-top:4px}
.orchbullets{margin:4px 0 0;padding-left:18px;font-size:9.2pt;color:var(--ink2)}
.orchbullets li{padding:1px 0}

.collabbox{background:var(--line2);border-radius:8px;padding:12px 14px}
.collablist{list-style:none;margin:6px 0 0;padding:0}
.collablist li{padding:7px 0;border-bottom:1px solid var(--line);font-size:9.4pt}
.collablist li:last-child{border-bottom:none}
.ekind{display:inline-block;font-size:7.2pt;font-weight:750;letter-spacing:.06em;
  background:#eef2ff;color:#4a3fbf;padding:1px 6px;border-radius:3px;margin-right:7px}
.edetail{display:block;color:var(--ink3);font-size:8.8pt;margin-top:2px}
.revisions{margin-top:12px;background:#fffaeb;border:1px solid #fedf89;border-radius:8px;padding:11px 13px}
.revisions ul{margin:5px 0 0;padding-left:18px;font-size:9pt;color:#93370d}
.parttag{background:#fffaeb;border:1px solid #fedf89;color:#93370d;padding:1px 6px;
  border-radius:4px;font-size:7.8pt;font-weight:650}
.unavailtag{background:#fef3f2;border:1px solid #fecdca;color:#b42318;padding:1px 6px;
  border-radius:4px;font-size:7.8pt;font-weight:650}

/* provenance table */
.ptable{width:100%;border-collapse:collapse;margin-top:6px}
.ptable td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top;page-break-inside:avoid}
.ptable td:first-child{width:44%;padding-left:0}
.ptable b{font-size:10pt}
.pmeta{font-size:8.8pt;color:var(--ink3);margin-top:3px}
.pendpoint{margin-top:5px}
.pendpoint code,.pdomains code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:8pt;background:var(--line2);padding:1px 5px;border-radius:3px;color:var(--ink2);
  word-break:break-all}
.pdomains{width:40%}
.pdomains div{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px;align-items:center}
.dcount{font-size:7.6pt;color:var(--ink3);margin-right:6px}
.nodata{font-size:8.4pt;color:var(--ink3);font-style:italic}
.pcount{text-align:right;padding-right:0 !important;white-space:nowrap}
.pcount b{display:block;font-size:14pt;font-weight:680}
.pcount span{font-size:7.4pt;text-transform:uppercase;letter-spacing:.05em;color:var(--ink3)}
.domsum{margin-top:16px;background:var(--line2);border-radius:8px;padding:12px 14px}
.domchips{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px}
.domchip{display:inline-flex;align-items:center;gap:4px;background:#fff;border:1px solid var(--line);
  border-radius:5px;padding:2px 6px;font-size:8pt;color:var(--ink3)}

/* sources */
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:22px}
ul.plain{list-style:none;margin:6px 0 0;padding:0}
ul.plain li{font-size:9.6pt;padding:4px 0;border-bottom:1px solid var(--line2);color:var(--ink2)}
.coverage{margin-top:18px;background:var(--line2);border-radius:8px;padding:12px 14px}
.degraded{margin-top:14px;background:#fffaeb;border:1px solid #fedf89;border-radius:8px;padding:11px 13px}
.degraded ul{margin:6px 0 0;padding-left:18px;font-size:9.2pt;color:#93370d}

/* caveats */
.caveat{border-left:2px solid var(--line);padding-left:13px;margin-bottom:13px;page-break-inside:avoid}
.caveat b{font-size:10pt}
.caveat p{font-size:9.4pt;color:var(--ink2);margin-top:2px}

/* footer */
.docfoot{margin-top:34px;padding-top:14px;border-top:2px solid var(--ink);
  font-size:8.6pt;color:var(--ink3)}
.docfoot b{color:var(--ink)}
.disclaimer{margin-top:8px;font-style:italic}

/* print */
@page{size:A4;margin:15mm 14mm 17mm}
@media print{
  html,body{background:#fff}
  .printbar{display:none}
  .doc{margin:0;padding:0;box-shadow:none;max-width:none}
  .sec{page-break-before:auto}
  .cover{page-break-after:avoid}
  a{color:var(--ink);text-decoration:none}
}
@media (max-width:760px){
  .doc{padding:16px;margin:0}
  .idgrid{grid-template-columns:1fr 1fr}
  .tiles{grid-template-columns:repeat(3,1fr)}
  .exwrap,.twocol{grid-template-columns:1fr}
  .fsrc{width:auto;text-align:left;padding-left:0 !important}
  .ftable td{display:block;border-bottom:none}
  .ftable tr{display:block;border-bottom:1px solid var(--line2);padding:6px 0}
}
"""
