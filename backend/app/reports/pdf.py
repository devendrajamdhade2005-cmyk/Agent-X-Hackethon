"""Server-side PDF generation with ReportLab.

A real PDF is produced on the server rather than relying on the browser's print
dialog, so `Download PDF` yields an actual file. Light theme, A4, page numbers
and a running footer carrying the report ID.
"""

from __future__ import annotations

import io
from html import escape
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#101828")
INK2 = colors.HexColor("#475467")
INK3 = colors.HexColor("#667085")
LINE = colors.HexColor("#e4e7ec")
LINE2 = colors.HexColor("#f2f4f7")
ACCENT = colors.HexColor("#5b4bd6")
ACCENT_BG = colors.HexColor("#f4f3ff")
HIGH = colors.HexColor("#b42318")
MED = colors.HexColor("#b54708")
LOW = colors.HexColor("#067647")
WARN_BG = colors.HexColor("#fffaeb")
WARN_LINE = colors.HexColor("#fedf89")

PRIORITY_COLOR = {"HIGH": HIGH, "MEDIUM": MED, "LOW": LOW}
PRIORITY_BG = {
    "HIGH": colors.HexColor("#fef3f2"),
    "MEDIUM": colors.HexColor("#fffaeb"),
    "LOW": colors.HexColor("#ecfdf3"),
}


def _t(value: Any) -> str:
    """Escape for ReportLab's mini-HTML."""
    return escape(str(value if value is not None else ""), quote=False)


def _hex(color: colors.Color) -> str:
    """ReportLab's hexval() yields '0xb42318'; its inline markup needs '#b42318'."""
    return "#" + color.hexval()[2:]


# ─────────────────────────────────────────────────────────────
# styles
# ─────────────────────────────────────────────────────────────
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    mk = lambda **kw: ParagraphStyle(parent=base, **kw)  # noqa: E731

    return {
        "brand": mk(name="brand", fontName="Helvetica-Bold", fontSize=12, textColor=INK,
                    leading=14, spaceAfter=1),
        "brandsub": mk(name="brandsub", fontName="Helvetica", fontSize=7.5, textColor=INK3,
                       leading=10, spaceAfter=18),
        "h1": mk(name="h1", fontName="Helvetica-Bold", fontSize=25, textColor=INK,
                 leading=29, spaceAfter=16),
        "h2": mk(name="h2", fontName="Helvetica-Bold", fontSize=13.5, textColor=INK,
                 leading=17, spaceBefore=4, spaceAfter=6),
        "h3": mk(name="h3", fontName="Helvetica-Bold", fontSize=11, textColor=INK,
                 leading=14, spaceAfter=4),
        "label": mk(name="label", fontName="Helvetica-Bold", fontSize=6.8, textColor=INK3,
                    leading=9, spaceAfter=2),
        "body": mk(name="body", fontName="Helvetica", fontSize=9.2, textColor=INK2,
                   leading=13.4, alignment=TA_LEFT),
        "lede": mk(name="lede", fontName="Helvetica", fontSize=10.4, textColor=INK2,
                   leading=15.4, spaceAfter=4),
        "goal": mk(name="goal", fontName="Helvetica-Bold", fontSize=11.5, textColor=INK,
                   leading=15.5),
        "meta": mk(name="meta", fontName="Helvetica", fontSize=8, textColor=INK3, leading=11),
        "small": mk(name="small", fontName="Helvetica", fontSize=8.2, textColor=INK3, leading=11.4),
        "cardtitle": mk(name="cardtitle", fontName="Helvetica-Bold", fontSize=11, textColor=INK,
                        leading=14, spaceAfter=2),
        "action": mk(name="action", fontName="Helvetica-Bold", fontSize=9.2, textColor=INK,
                     leading=13.2),
        "stage": mk(name="stage", fontName="Helvetica-Bold", fontSize=9, textColor=INK, leading=12),
        "sub": mk(name="sub", fontName="Helvetica", fontSize=8.6, textColor=INK3, leading=12,
                  spaceAfter=8),
        "loop": mk(name="loop", fontName="Helvetica-Bold", fontSize=8.6, textColor=ACCENT,
                   leading=12, alignment=1),
        "mono": mk(name="mono", fontName="Courier", fontSize=8.4, textColor=INK, leading=11),
    }


# ─────────────────────────────────────────────────────────────
# document shell (header rule + footer with page numbers)
# ─────────────────────────────────────────────────────────────
class _Doc(BaseDocTemplate):
    def __init__(self, buf: io.BytesIO, report: dict[str, Any]) -> None:
        super().__init__(
            buf, pagesize=A4,
            leftMargin=17 * mm, rightMargin=17 * mm,
            topMargin=16 * mm, bottomMargin=18 * mm,
            title=f"InsightPulse Intelligence Report {report.get('report_id', '')}",
            author="InsightPulse AI",
            subject=str(report.get("tracking_goal", ""))[:200],
        )
        self.report = report
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height, id="body",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc) -> None:  # noqa: ANN001
        canvas.saveState()
        w, _h = A4
        rid = str(self.report.get("report_id", ""))

        # Running header from page 2 onward.
        if doc.page > 1:
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.setFillColor(INK3)
            canvas.drawString(doc.leftMargin, A4[1] - 11 * mm, "INSIGHTPULSE AI — INTELLIGENCE REPORT")
            canvas.drawRightString(w - doc.rightMargin, A4[1] - 11 * mm, rid)
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(doc.leftMargin, A4[1] - 13 * mm, w - doc.rightMargin, A4[1] - 13 * mm)

        # Footer: report id + page number.
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 13 * mm, w - doc.rightMargin, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(INK3)
        canvas.drawString(doc.leftMargin, 9 * mm, f"Report {rid}")
        canvas.drawCentredString(w / 2, 9 * mm, "Generated by an autonomous agent — verify before acting")
        canvas.drawRightString(w - doc.rightMargin, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()


# ─────────────────────────────────────────────────────────────
# builders
# ─────────────────────────────────────────────────────────────
def render_pdf(report: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = _Doc(buf, report)
    s = _styles()
    width = doc.width
    flow: list[Any] = []

    _cover(flow, report, s, width)
    _exec(flow, report, s, width)
    _insights(flow, report, s, width)
    _agents(flow, report, s, width)
    _memory(flow, report, s, width)
    _execution(flow, report, s, width)
    _findings(flow, report, s, width)
    _sources(flow, report, s, width)
    _caveats(flow, report, s, width)

    doc.build(flow)
    return buf.getvalue()


def _rule(width: float, color=LINE, thickness: float = 0.6) -> Table:
    t = Table([[""]], colWidths=[width], rowHeights=[thickness])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)]))
    return t


def _heading(flow: list[Any], num: str, title: str, s: dict, width: float) -> None:
    flow.append(Spacer(1, 14))
    flow.append(Paragraph(
        f'<font color="#5b4bd6" size="8"><b>{num}</b></font>  {_t(title)}', s["h2"]))
    flow.append(_rule(width))
    flow.append(Spacer(1, 8))


def _cover(flow: list[Any], r: dict, s: dict, width: float) -> None:
    flow.append(Paragraph("InsightPulse AI", s["brand"]))
    flow.append(Paragraph("AUTONOMOUS RESEARCH &amp; COMPETITOR INTELLIGENCE", s["brandsub"]))
    flow.append(Paragraph("Intelligence Report", s["h1"]))

    goal = Table(
        [[Paragraph("TRACKING GOAL", s["label"])],
         [Paragraph(_t(r.get("tracking_goal") or "—"), s["goal"])]],
        colWidths=[width],
    )
    goal.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
    ]))
    flow.append(goal)

    for label, values in (("KEYWORDS", r.get("keywords")), ("COMPETITORS TRACKED", r.get("competitors"))):
        if values:
            flow.append(Spacer(1, 7))
            flow.append(Paragraph(label, s["label"]))
            flow.append(Paragraph(_t(", ".join(values)), s["body"]))

    flow.append(Spacer(1, 14))
    cells = [
        ("GENERATED", r.get("generated_at_display")),
        ("REPORT ID", r.get("report_id")),
        ("RUN ID", r.get("run_id")),
        ("STATUS", r.get("status")),
    ]
    grid = Table(
        [[Paragraph(k, s["label"]) for k, _ in cells],
         [Paragraph(f"<b>{_t(v)}</b>", s["body"]) for _, v in cells]],
        colWidths=[width / 4] * 4,
    )
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(grid)

    if r.get("simulated"):
        flow.append(Spacer(1, 12))
        flow.append(_callout(
            "DATA NOTICE",
            "Some findings in this report were produced in simulation mode because live "
            "source access was unavailable. Simulated items are labelled throughout and are "
            "not verified real-world data.",
            s, width))

    flow.append(Spacer(1, 10))
    flow.append(_rule(width, INK, 1.4))


def _callout(title: str, body: str, s: dict, width: float) -> Table:
    t = Table(
        [[Paragraph(f"<b>{_t(title)}</b>", s["small"])],
         [Paragraph(_t(body), s["small"])]],
        colWidths=[width],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, WARN_LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (0, 0), 7),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
    ]))
    return t


def _exec(flow: list[Any], r: dict, s: dict, width: float) -> None:
    st = r.get("summary_stats") or {}
    _heading(flow, "01", "Executive Summary", s, width)
    flow.append(Paragraph(_t(r.get("executive_summary")), s["lede"]))
    flow.append(Spacer(1, 12))

    tiles = [
        ("High priority", st.get("high_priority_count", 0), HIGH),
        ("Medium priority", st.get("medium_priority_count", 0), MED),
        ("Low priority", st.get("low_priority_count", 0), LOW),
        ("Total findings", st.get("total_findings", 0), INK),
        ("Tools used", st.get("tools_used_count", 0), INK),
    ]
    cw = width / len(tiles)
    row_v = [Paragraph(
        f'<para alignment="center"><font size="15" color="{_hex(c)}"><b>{v}</b></font></para>',
        s["body"]) for _, v, c in tiles]
    row_l = [Paragraph(f'<para alignment="center">{_t(k.upper())}</para>', s["label"])
             for k, _, _ in tiles]

    t = Table([row_v, row_l], colWidths=[cw] * len(tiles))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LINE2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("INNERGRID", (0, 0), (-1, -1), 3, colors.white),
    ]))
    flow.append(t)

    analyst = (r.get("analyst_summary") or "").strip()
    if analyst:
        flow.append(Spacer(1, 12))
        flow.append(Paragraph("ANALYST SUMMARY", s["label"]))
        for para in analyst.split("\n\n"):
            if para.strip():
                flow.append(Paragraph(_t(para.strip()), s["body"]))
                flow.append(Spacer(1, 5))


def _insights(flow: list[Any], r: dict, s: dict, width: float) -> None:
    insights = r.get("insights") or []
    _heading(flow, "02", "Prioritized Insights", s, width)
    if not insights:
        flow.append(Paragraph(
            "<i>No findings cleared the relevance threshold for this goal.</i>", s["body"]))
        return

    flow.append(Paragraph(
        f"{len(insights)} insight(s), ordered by priority then relevance.", s["sub"]))

    for n, i in enumerate(insights, 1):
        pri = i.get("priority", "MEDIUM")
        col = PRIORITY_COLOR.get(pri, MED)

        meta = [f'<b>{_t(i.get("category_label"))}</b>']
        if i.get("date"):
            meta.append(_t(i["date"]))
        if i.get("competitor"):
            meta.append(_t(i["competitor"]))
        if i.get("relevance_score") is not None:
            meta.append(f'relevance {_t(i["relevance_score"])}')
        if i.get("simulated"):
            meta.append('<font color="#93370d"><b>SIMULATED</b></font>')

        inner: list[Any] = [
            Table([[
                Paragraph(f'<font color="{_hex(col)}" size="7.4"><b>{_t(pri)} PRIORITY</b></font>',
                          s["body"]),
                Paragraph(f'<para alignment="right"><font color="#98a2b3" size="7.6">#{n}</font></para>',
                          s["body"]),
            ]], colWidths=[width * 0.72, width * 0.28 - 20]),
            Paragraph(_t(i.get("title")), s["cardtitle"]),
            Paragraph(" &middot; ".join(meta), s["meta"]),
            Spacer(1, 6),
            Paragraph("WHAT HAPPENED", s["label"]),
            Paragraph(_t(i.get("what_happened")), s["body"]),
            Spacer(1, 5),
            Paragraph("WHY IT MATTERS", s["label"]),
            Paragraph(_t(i.get("why_it_matters")), s["body"]),
            Spacer(1, 5),
            Paragraph("RECOMMENDED ACTION", s["label"]),
            Paragraph(_t(i.get("recommended_action")), s["action"]),
            Spacer(1, 5),
            Paragraph("SOURCE", s["label"]),
        ]
        # Lead with the publisher, then the provider that surfaced it.
        host = _host(i.get("source_url") or "")
        src = (f'<b>{_t(host)}</b> — retrieved via {_t(i.get("source_name"))}'
               if host else _t(i.get("source_name")))
        if i.get("source_url"):
            url = i["source_url"]
            inner.append(Paragraph(
                f'{src}<br/><link href="{escape(url, quote=True)}" color="#5b4bd6">'
                f'{_t(_shorten(url, 78))}</link>', s["small"]))
        elif i.get("simulated"):
            inner.append(Paragraph(
                f'{src} · <font color="#93370d">simulated — no live URL</font>', s["small"]))
        else:
            inner.append(Paragraph(src, s["small"]))

        card = Table([[inner]], colWidths=[width])
        card.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("LINEBEFORE", (0, 0), (0, -1), 2.4, col),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        flow.append(KeepTogether(card))
        flow.append(Spacer(1, 9))


def _agents(flow: list[Any], r: dict, s: dict, width: float) -> None:
    """Per-agent contributions — the auditable record of multi-agent collaboration."""
    ac = r.get("agent_contributions") or {}
    specialists = ac.get("specialists") or []
    if not specialists:
        return

    _heading(flow, "03", "Agent Contributions", s, width)
    flow.append(Paragraph(_t(ac.get("architecture")), s["sub"]))

    # orchestrator's selection decisions
    flow.append(Paragraph("ORCHESTRATOR'S AGENT-SELECTION DECISIONS", s["label"]))
    flow.append(Spacer(1, 3))
    for entry in [*(ac.get("selected") or []), *(ac.get("not_selected") or [])]:
        verdict = "SELECTED" if entry.get("selected") else "NOT SELECTED"
        colour = "#067647" if entry.get("selected") else "#667085"
        row = Table([[[
            Paragraph(
                f'<b>{_t(entry.get("name"))}</b>  '
                f'<font size="7" color="{colour}"><b>{verdict}</b></font>', s["body"]),
            Paragraph(_t(entry.get("reason")), s["small"]),
        ]]], colWidths=[width])
        row.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        flow.append(KeepTogether(row))

    flow.append(Spacer(1, 10))

    for a in specialists:
        cov = str(a.get("coverage", "live")).upper()
        cov_colour = {"LIVE": "#067647", "PARTIAL": "#b54708",
                      "SIMULATED": "#93370d", "UNAVAILABLE": "#b42318"}.get(cov, "#667085")
        stats = (
            f'<b>{a.get("findings_count", 0)}</b> findings &nbsp; '
            f'<b>{a.get("relevant_count", 0)}</b> relevant &nbsp; '
            f'<b>{round(float(a.get("confidence") or 0) * 100)}%</b> confidence &nbsp; '
            f'<b>{a.get("corroborated", 0)}</b> cross-validated'
        )
        block: list[Any] = [
            Paragraph(
                f'<b>{_t(a.get("name"))}</b>  '
                f'<font size="7" color="{cov_colour}"><b>{_t(str(a.get("status","")).upper())}'
                f' · {cov}</b></font>', s["body"]),
            Paragraph(_t(a.get("responsibility")), s["small"]),
            Paragraph(stats, s["meta"]),
            Paragraph(
                f'<b>Tools:</b> {_t(", ".join(a.get("tools_used") or []) or "none")} &nbsp;·&nbsp; '
                f'<b>Providers:</b> {_t(", ".join(a.get("sources_checked") or []) or "none")}',
                s["meta"]),
            Paragraph(_t(a.get("summary")), s["small"]),
        ]
        for key, label in (("research_trends", "Recurring themes"),
                           ("competitors_analyzed", "Companies analysed"),
                           ("market_signals", "Market signals")):
            if a.get(key):
                block.append(Paragraph(
                    f'<b>{label}:</b> {_t(", ".join(a[key][:5]))}', s["meta"]))

        card = Table([[block]], colWidths=[width])
        card.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(KeepTogether(card))
        flow.append(Spacer(1, 8))

    orch = ac.get("orchestrator")
    if orch:
        bullets = [Paragraph(f"• {_t(b)}", s["small"]) for b in (orch.get("bullets") or [])]
        card = Table([[[
            Paragraph(f'<b>{_t(orch.get("name"))}</b>  '
                      f'<font size="7" color="#5b4bd6"><b>COORDINATOR</b></font>', s["body"]),
            *bullets,
        ]]], colWidths=[width])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT_BG),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(KeepTogether(card))
        flow.append(Spacer(1, 10))

    events = ac.get("collaboration_events") or []
    flow.append(Paragraph(
        f"COLLABORATION EVENTS ({ac.get('collaboration_count', 0)})", s["label"]))
    flow.append(Spacer(1, 3))
    if not events:
        flow.append(Paragraph(
            "<i>No cross-agent collaboration was required for this goal.</i>", s["small"]))
    for e in events[:8]:
        row = Table([[[
            Paragraph(
                f'<font size="7" color="#4a3fbf"><b>'
                f'{_t(str(e.get("kind","")).replace("_"," ").upper())}</b></font>  '
                f'<b>{_t(e.get("summary"))}</b>', s["small"]),
            Paragraph(_t(e.get("detail")), s["meta"]),
        ]]], colWidths=[width])
        row.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        flow.append(KeepTogether(row))

    revisions = ac.get("plan_revisions") or []
    if len(revisions) > 1:
        flow.append(Spacer(1, 9))
        flow.append(_callout(
            "PLAN REVISIONS DURING THE RUN",
            "; ".join(revisions[1:]), s, width))


def _memory(flow: list[Any], r: dict, s: dict, width: float) -> None:
    """04 — Context & Memory."""
    cm = r.get("context_memory") or {}
    if not cm.get("available"):
        return

    tc = cm.get("task_context") or {}
    w = cm.get("working") or {}
    change = cm.get("change") or {}
    cons = cm.get("consolidation") or {}

    _heading(flow, "04", "Context & Memory", s, width)
    flow.append(Paragraph(
        f"Working memory reached version <b>{_t(w.get('version'))}</b> across "
        f"{_t(w.get('updates'))} update(s), retaining {_t(w.get('fact_count'))} fact(s) "
        f"of which {_t(w.get('important_fact_count'))} were important. Task context was "
        f"extracted by the {_t(tc.get('author'))} reader.", s["body"]))
    flow.append(Spacer(1, 9))

    basis = ["Current intelligence gathered in this run"]
    if cm.get("used_historical_context"):
        basis.append("Relevant historical context retrieved from previous monitoring")
    flow.append(_callout("THIS REPORT IS BASED ON", "; ".join(basis), s, width))
    flow.append(Spacer(1, 9))

    # Task context
    bits = [f"<b>Topics:</b> {_t(', '.join(tc.get('topics') or []) or 'none detected')}"]
    if tc.get("competitors"):
        bits.append(f"<b>Tracked companies:</b> {_t(', '.join(tc['competitors']))}")
    if tc.get("domains"):
        bits.append(f"<b>Domains:</b> {_t(', '.join(tc['domains']))}")
    bits.append(f"<b>Time scope:</b> {_t(tc.get('time_scope'))}")
    if tc.get("constraints"):
        bits.append(f"<b>Constraints:</b> {_t('; '.join(tc['constraints']))}")
    if tc.get("continuation"):
        bits.append("<b>Continuation</b> of earlier monitoring")
    flow.append(Paragraph("TASK CONTEXT", s["label"]))
    for bit in bits:
        flow.append(Paragraph(bit, s["small"]))
    flow.append(Spacer(1, 8))

    # Plan state
    steps = cm.get("plan_steps") or []
    if steps:
        flow.append(Paragraph("EXECUTION PLAN STATE", s["label"]))
        rows = [["Step", "Status", "Result"]]
        for step in steps:
            rows.append([
                _t(step.get("name")),
                _t(str(step.get("status", "")).replace("_", " ").upper()),
                _t(step.get("reference") or "—"),
            ])
        table = Table(rows, colWidths=[width * 0.46, width * 0.22, width * 0.32])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        flow.append(table)
        flow.append(Spacer(1, 9))

    # Cross-agent context sharing — the core evidence for this section.
    shared = cm.get("shared_context") or []
    if shared:
        flow.append(Paragraph("CONTEXT SHARED BETWEEN AGENTS", s["label"]))
        for sh in shared:
            flow.append(Paragraph(
                f"<b>{_t(sh.get('agent'))}</b> received {_t(sh.get('facts'))} finding(s) "
                f"from {_t(', '.join(sh.get('from') or []))}.", s["small"]))
            flow.append(Paragraph(
                f"Context: {_t(', '.join(sh.get('received') or []))}", s["meta"]))
            if sh.get("focus"):
                flow.append(Paragraph(
                    f"Search focus carried over: {_t(', '.join(sh['focus']))}", s["meta"]))
            for why in sh.get("withheld") or []:
                flow.append(Paragraph(f"Withheld: {_t(why)}", s["meta"]))
            flow.append(Spacer(1, 4))
        flow.append(Spacer(1, 5))

    # Retained facts
    facts = cm.get("retained_facts") or []
    if facts:
        flow.append(Paragraph("FINDINGS RETAINED IN WORKING MEMORY", s["label"]))
        for fact in facts[:6]:
            sim = " · SIMULATED" if fact.get("simulated") else ""
            flow.append(Paragraph(
                f"[{_t(fact.get('importance'))}] {_t(fact.get('text'))}", s["small"]))
            flow.append(Paragraph(f"{_t(fact.get('agent'))}{sim}", s["meta"]))
        flow.append(Spacer(1, 9))

    # Long-term memory
    flow.append(Paragraph("LONG-TERM MEMORY", s["label"]))
    retrieved = cm.get("retrieved") or []
    if retrieved:
        for m in retrieved:
            rel = f" · relevance {_t(m.get('relevance'))}" if m.get("relevance") else ""
            flow.append(Paragraph(
                f"<b>{_t(m.get('type'))}</b> — {_t(m.get('summary'))}", s["small"]))
            flow.append(Paragraph(f"from run {_t(m.get('from_run'))}{rel}", s["meta"]))
    else:
        flow.append(Paragraph(
            f"No relevant previous context was found for this goal "
            f"({_t(cm.get('retrieval_status'))}). This report is based on current "
            f"intelligence only.", s["small"]))
    flow.append(Spacer(1, 6))

    if change.get("compared"):
        flow.append(Paragraph(
            f"<b>Detected change:</b> {_t(change.get('verdict'))} — "
            f"{_t(change.get('detail'))}", s["small"]))
    else:
        flow.append(Paragraph(
            "No historical baseline was available, so no change comparison was made.",
            s["small"]))

    line = (f"Consolidated {_t(cons.get('stored', 0))} new item(s) for future monitoring")
    if cons.get("refreshed"):
        line += f", refreshed {_t(cons.get('refreshed'))}"
    if cons.get("rejected"):
        line += f", rejected {_t(cons.get('rejected'))} as not durable"
    line += "."
    if cm.get("store_total") is not None:
        line += f" Store holds {_t(cm.get('store_total'))} item(s)."
    flow.append(Paragraph(line, s["meta"]))

    if w.get("compressions"):
        flow.append(Paragraph(
            f"Context compression ran {_t(w.get('compressions'))} time(s), folding "
            f"{_t(w.get('compressed_count'))} lower-importance fact(s) into a summary "
            f"while keeping important facts verbatim.", s["meta"]))


def _execution(flow: list[Any], r: dict, s: dict, width: float) -> None:
    ex = r.get("execution_summary") or {}
    _heading(flow, "05", "Agent Execution Summary", s, width)

    loop = Table([[Paragraph(_t(ex.get("loop")), s["loop"])]], colWidths=[width])
    loop.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    flow.append(loop)
    flow.append(Spacer(1, 11))

    m = ex.get("metrics") or {}
    rows = [
        ("Iterations", f"{m.get('iterations', 0)} of {m.get('max_iterations', 0)}"),
        ("Tool calls", m.get("tool_calls", 0)),
        ("Duration", f"{m.get('duration_seconds', 0)}s"),
        ("Sources checked", m.get("sources_checked", 0)),
        ("Relevant findings", m.get("relevant_findings", 0)),
        ("Duplicates suppressed", m.get("duplicates_suppressed", 0)),
        ("Errors handled", m.get("errors_handled", 0)),
        ("Reasoning engine", m.get("reasoner", "—")),
    ]
    mt = Table(
        [[Paragraph(_t(k), s["small"]), Paragraph(f'<para alignment="right"><b>{_t(v)}</b></para>', s["small"])]
         for k, v in rows],
        colWidths=[width * 0.62, width * 0.38 - 18],
    )
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LINE2),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(mt)
    flow.append(Spacer(1, 12))

    flow.append(Paragraph("EXECUTION TRAIL", s["label"]))
    flow.append(Spacer(1, 4))
    items = []
    for step in ex.get("steps") or []:
        parts = [Paragraph(_t(step.get("stage")), s["stage"])]
        if step.get("detail"):
            parts.append(Paragraph(_t(step["detail"]), s["small"]))
        items.append(ListItem(parts, leftIndent=14, spaceAfter=6))
    if items:
        flow.append(ListFlowable(
            items, bulletType="1", bulletFontName="Helvetica-Bold",
            bulletFontSize=7.5, bulletColor=ACCENT, leftIndent=15,
        ))


def _findings(flow: list[Any], r: dict, s: dict, width: float) -> None:
    groups = r.get("findings_by_category") or []
    if not groups:
        return
    _heading(flow, "06", "Detailed Findings", s, width)
    flow.append(Paragraph(
        "Every item the agent collected, grouped by source category.", s["sub"]))

    for g in groups:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            f'{_t(g.get("label"))}  <font color="#667085" size="8">'
            f'{_t(g.get("count"))} finding(s)</font>', s["h3"]))
        flow.append(_rule(width, LINE2))
        flow.append(Spacer(1, 4))

        for f in g.get("items", []):
            meta = [x for x in (
                f.get("date"), f.get("organization"), f.get("provider_label"),
                f'relevance {f["relevance"]}' if f.get("relevance") is not None else None,
                "SIMULATED" if f.get("simulated") else None,
            ) if x]
            block: list[Any] = [
                Paragraph(f'<b>{_t(f.get("title"))}</b>', s["body"]),
            ]
            if f.get("description"):
                block.append(Paragraph(_t(f["description"]), s["small"]))
            if meta:
                block.append(Paragraph(
                    f'<font color="#667085">{_t(" · ".join(str(m) for m in meta))}</font>', s["meta"]))
            if f.get("url"):
                block.append(Paragraph(
                    f'<link href="{escape(f["url"], quote=True)}" color="#5b4bd6">'
                    f'{_t(_shorten(f["url"]))}</link>', s["meta"]))

            row = Table([[block]], colWidths=[width])
            row.setStyle(TableStyle([
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE2),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            flow.append(KeepTogether(row))


def _sources(flow: list[Any], r: dict, s: dict, width: float) -> None:
    src = r.get("sources") or {}
    _heading(flow, "07", "Sources & Coverage", s, width)

    def col(title: str, lines: list[str]) -> list[Any]:
        out: list[Any] = [Paragraph(title, s["label"]), Spacer(1, 3)]
        out.extend(Paragraph(line, s["small"]) for line in (lines or ["—"]))
        return out

    flow.append(Paragraph(
        "Every provider queried, who operates it, the endpoint used, and the real "
        "publication domains the findings came from.", s["sub"]))

    # Full provenance table — one row per provider.
    for x in src.get("sources_used") or []:
        origin = x.get("origin") or {}
        state = ('<font color="#067647"><b>LIVE</b></font>' if x.get("live")
                 else '<font color="#93370d"><b>SIMULATED</b></font>')
        doms = x.get("domains") or []
        dom_text = (", ".join(f"{h} ({n})" for h, n in doms) if doms
                    else "no live URLs captured")

        left = [
            Paragraph(f'<b>{_t(x.get("name"))}</b>  {state}', s["body"]),
            Paragraph(_t(origin.get("what") or ""), s["small"]),
            Paragraph(
                f'Operated by {_t(origin.get("operator") or "unknown")} · '
                f'access: {_t(origin.get("auth") or "unknown")}', s["meta"]),
            Paragraph(f'<font face="Courier" size="7.4">{_t(origin.get("endpoint") or "n/a")}</font>',
                      s["meta"]),
            Paragraph(f'<b>Domains:</b> {_t(dom_text)}', s["meta"]),
        ]
        row = Table(
            [[left, Paragraph(
                f'<para alignment="right"><b>{_t(x.get("findings"))}</b><br/>'
                f'<font size="6.6" color="#667085">FINDINGS</font></para>', s["body"])]],
            colWidths=[width * 0.82, width * 0.18],
        )
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        flow.append(KeepTogether(row))

    domains = src.get("domains") or []
    if domains:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph(
            f"ALL PUBLICATION DOMAINS ({src.get('domain_count', 0)} DISTINCT)", s["label"]))
        flow.append(Paragraph(
            _t(", ".join(f"{h} ({n})" for h, n in domains[:40])), s["small"]))

    tools = [
        f'{_t(t.get("name"))}  <font color="#667085">{_t(t.get("findings"))} finding(s)</font>'
        for t in src.get("tools_used") or []
    ]
    flow.append(Spacer(1, 12))
    for line in col("TOOLS CALLED BY THE AGENT", tools):
        flow.append(line)

    coverage = src.get("coverage") or []
    if coverage:
        flow.append(Spacer(1, 11))
        flow.append(Paragraph("COVERAGE BY CATEGORY", s["label"]))
        flow.append(Spacer(1, 3))
        ct = Table(
            [[Paragraph(_t(c.get("category")), s["small"]),
              Paragraph(f'<para alignment="right"><b>{_t(c.get("count"))}</b></para>', s["small"])]
             for c in coverage],
            colWidths=[width * 0.7, width * 0.3 - 18],
        )
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LINE2),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(ct)

    degraded = src.get("degraded") or []
    if degraded:
        flow.append(Spacer(1, 11))
        flow.append(_callout(
            "PROVIDERS UNAVAILABLE DURING THIS RUN",
            "; ".join(f'{d.get("provider")}: {d.get("reason")}' for d in degraded),
            s, width))


def _caveats(flow: list[Any], r: dict, s: dict, width: float) -> None:
    caveats = r.get("caveats") or []
    if not caveats:
        return
    _heading(flow, "08", "Limitations & Caveats", s, width)
    for c in caveats:
        block = Table(
            [[[Paragraph(f'<b>{_t(c.get("title"))}</b>', s["body"]),
               Paragraph(_t(c.get("body")), s["small"])]]],
            colWidths=[width],
        )
        block.setStyle(TableStyle([
            ("LINEBEFORE", (0, 0), (0, -1), 1.6, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(KeepTogether(block))


def _shorten(url: str, limit: int = 60) -> str:
    text = str(url or "").replace("https://", "").replace("http://", "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _host(url: str) -> str:
    """Publication domain, e.g. 'wsj.com'."""
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(url)).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
