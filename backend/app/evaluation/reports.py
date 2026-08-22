"""Evaluation report export.

Projects a stored suite result into a document. Deliberately does not build a second
reporting engine: Markdown is rendered here (the evaluation report has a different
shape from an intelligence briefing), and the existing HTML shell/PDF path is reused
for presentation by handing the rendered sections to the shared renderers.

Everything printed comes from the stored suite. Where a metric was unavailable the
report says so and why, rather than printing a number.
"""

from __future__ import annotations

from typing import Any

from . import metrics as M
from . import human, store


def build_evaluation_report(suite: dict[str, Any]) -> dict[str, Any]:
    """Structured evaluation report payload (also the JSON export)."""
    aggregate = suite.get("aggregate") or {}
    counts = suite.get("counts") or {}
    runs = suite.get("runs") or []
    primary = [r for r in runs if r.get("system") == "insightpulse"]

    failures = [
        {
            "case_id": r.get("case_id"),
            "case_name": r.get("case_name"),
            "scenario_type": r.get("scenario_type"),
            "outcome": r.get("outcome"),
            "gate_failures": r.get("gate_failures"),
            "reasons": r.get("outcome_reasons"),
        }
        for r in primary if r.get("outcome") in {"PARTIAL", "FAIL", "ERROR"}
    ]

    uncertainty_cases = [
        {
            "case_id": r.get("case_id"),
            "verdict": ((r.get("metrics") or {}).get(M.UNCERTAINTY_HANDLING) or {})
                       .get("details", {}).get("verdict"),
            "score": ((r.get("metrics") or {}).get(M.UNCERTAINTY_HANDLING) or {}).get("value"),
        }
        for r in primary
        if ((r.get("metrics") or {}).get(M.UNCERTAINTY_HANDLING) or {}).get("available")
    ]

    recovery_cases = [
        {
            "case_id": r.get("case_id"),
            "injected": d.get("injected_failures"),
            "recovered": d.get("recovered_failures"),
            "fallbacks": d.get("fallback_used"),
            "rate": ((r.get("metrics") or {}).get(M.RECOVERY_RATE) or {}).get("value"),
        }
        for r in primary
        if ((r.get("metrics") or {}).get(M.RECOVERY_RATE) or {}).get("available")
        and (d := ((r.get("metrics") or {}).get(M.RECOVERY_RATE) or {}).get("details") or {})
    ]

    human_reviews = {
        rid: human.aggregate(rid)
        for r in primary
        if (rid := r.get("evaluation_run_id")) and store.human_reviews(rid)
    }

    return {
        "report_type": "evaluation",
        "suite_id": suite.get("suite_id"),
        "generated_from": suite.get("completed_at") or suite.get("started_at"),
        "mode": suite.get("mode"),
        "executive_summary": _executive_summary(suite, aggregate, counts),
        "methodology": {
            "metrics": M.catalogue_dicts(),
            "thresholds": suite.get("thresholds") or {},
            "outcome_rules": _outcome_rules(),
            "determinism": (suite.get("provenance") or {}).get("simulation_mode"),
        },
        "scenario_coverage": suite.get("scenario_matrix") or {},
        "results": aggregate,
        "counts": counts,
        "case_results": [
            {
                "case_id": r.get("case_id"),
                "case_name": r.get("case_name"),
                "scenario_type": r.get("scenario_type"),
                "outcome": r.get("outcome"),
                "accuracy": _v(r, M.ACCURACY),
                "groundedness": _v(r, M.GROUNDEDNESS),
                "hallucination": _v(r, M.HALLUCINATION_RATE),
                "recovery": _v(r, M.RECOVERY_RATE),
                "latency_ms": _v(r, M.LATENCY),
                "task_completion": _v(r, M.TASK_COMPLETION),
            }
            for r in primary
        ],
        "baseline_comparison": suite.get("baseline_comparison") or {},
        "reliability": suite.get("reliability") or {},
        "consistency": suite.get("consistency") or {},
        "human_review": human_reviews or {"note": "no human review submitted yet"},
        "failures": failures or [],
        "uncertainty_cases": uncertainty_cases,
        "recovery_cases": recovery_cases,
        "regression": suite.get("regression") or {},
        "history": store.history()[:10],
        "recommendations": _recommendations(aggregate, failures, suite),
        "provenance": suite.get("provenance") or {},
    }


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    """Markdown export."""
    a = report.get("results") or {}
    lines: list[str] = [
        "# InsightPulse — Evaluation Report",
        "",
        f"**Suite:** `{report.get('suite_id')}`  ",
        f"**Mode:** {report.get('mode')}  ",
        f"**Generated:** {report.get('generated_from')}",
        "",
        "## Executive summary",
        "",
        report.get("executive_summary", ""),
        "",
        "## Results",
        "",
        "| Metric | Value | Unit | Direction |",
        "|---|---|---|---|",
    ]
    for name, entry in a.items():
        if name == "overall_score" or not isinstance(entry, dict):
            continue
        if entry.get("available"):
            direction = "higher is better" if entry.get("higher_is_better") else "lower is better"
            lines.append(
                f"| {name} | {_fmt(entry.get('value'), entry.get('unit'))} | "
                f"{entry.get('unit')} | {direction} |"
            )
        else:
            lines.append(f"| {name} | not measurable | — | {entry.get('unavailable_reason','')} |")
    overall = a.get("overall_score")
    if isinstance(overall, (int, float)):
        lines += ["", f"**Overall score:** {overall:.1%}"]

    lines += ["", "## Scenario coverage", "",
              "| Scenario | Total | Pass | Partial | Fail | Score |", "|---|---|---|---|---|---|"]
    for scenario, b in (report.get("scenario_coverage") or {}).items():
        if not b.get("total"):
            continue
        lines.append(
            f"| {scenario} | {b['total']} | {b['passed']} | {b['partial']} | "
            f"{b['failed']} | {b['score']:.0%} |"
        )

    lines += ["", "## Case results", "",
              "| Case | Scenario | Outcome | Accuracy | Grounded | Halluc. | Recovery | Latency |",
              "|---|---|---|---|---|---|---|---|"]
    for c in report.get("case_results") or []:
        lines.append(
            f"| {c['case_id']} | {c['scenario_type']} | {c['outcome']} | "
            f"{_pct(c['accuracy'])} | {_pct(c['groundedness'])} | {_pct(c['hallucination'])} | "
            f"{_pct(c['recovery'])} | {_ms(c['latency_ms'])} |"
        )

    for system, comp in (report.get("baseline_comparison") or {}).items():
        lines += ["", f"## Baseline comparison — {system}", ""]
        if comp.get("blocked"):
            lines += [f"> This baseline was unavailable: {comp.get('blocked_reason')}.",
                      f"> {comp.get('blocked_note','')}", ""]
        lines += ["| Metric | Baseline | InsightPulse | Difference | Direction |",
                  "|---|---|---|---|---|"]
        for row in comp.get("rows") or []:
            if row.get("available"):
                lines.append(
                    f"| {row['label']} | {_fmt(row['baseline'], row['unit'])} | "
                    f"{_fmt(row['insightpulse'], row['unit'])} | {row['difference']} | "
                    f"{row.get('direction','')} |"
                )
            else:
                lines.append(f"| {row['label']} | n/a | n/a | — | {row.get('unavailable_reason','')} |")

    failures = report.get("failures") or []
    lines += ["", "## Failures and partial outcomes", ""]
    if failures:
        for f in failures:
            lines.append(f"- **{f['case_id']} ({f['scenario_type']}) → {f['outcome']}** — "
                         f"{'; '.join(f.get('reasons') or [])}")
    else:
        lines.append("No case failed or returned a partial outcome in this suite.")

    reg = report.get("regression") or {}
    lines += ["", "## Regression", ""]
    if reg.get("compared"):
        lines.append(f"Compared against `{reg.get('previous_suite_id')}`.")
        for ch in reg.get("changes") or []:
            if ch.get("direction") in {"improved", "regressed"}:
                lines.append(f"- {ch['metric']}: {ch['previous']} → {ch['current']} ({ch['direction']})")
        if not reg.get("regressions"):
            lines.append("- No regressions detected.")
    else:
        lines.append(reg.get("reason", "No previous suite to compare against."))

    lines += ["", "## Methodology", ""]
    for spec in (report.get("methodology") or {}).get("metrics") or []:
        lines += [f"### {spec['label']}", "",
                  f"- **Definition:** {spec['definition']}",
                  f"- **Formula:** `{spec['formula']}`",
                  f"- **Unit:** {spec['unit']} · **Scope:** {spec['scope']}",
                  f"- **Data source:** {spec['data_source']}", ""]

    recs = report.get("recommendations") or []
    if recs:
        lines += ["## Recommendations", ""] + [f"- {r}" for r in recs]
    return "\n".join(lines)


def render_evaluation_html(report: dict[str, Any]) -> str:
    """Minimal print-ready HTML wrapper around the Markdown projection."""
    body = render_evaluation_markdown(report)
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(str(v.get('value')))}</td></tr>"
        for k, v in (report.get("results") or {}).items()
        if isinstance(v, dict) and v.get("available")
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>InsightPulse — Evaluation Report</title>"
        "<style>body{font-family:-apple-system,system-ui,sans-serif;max-width:900px;"
        "margin:40px auto;line-height:1.6;color:#0f172a}pre{white-space:pre-wrap;"
        "background:#f6f7fb;padding:16px;border-radius:10px;font-size:12.5px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "td,th{border:1px solid #e6e9f2;padding:6px 10px;font-size:13px;text-align:left}"
        "h1{font-size:24px}</style></head><body>"
        f"<h1>InsightPulse — Evaluation Report</h1>"
        f"<p><b>Suite:</b> {_esc(str(report.get('suite_id')))} · "
        f"<b>Mode:</b> {_esc(str(report.get('mode')))}</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
        f"<pre>{_esc(body)}</pre></body></html>"
    )


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _executive_summary(suite: dict[str, Any], aggregate: dict[str, Any], counts: dict[str, Any]) -> str:
    overall = aggregate.get("overall_score")
    parts = [
        f"The suite executed {counts.get('runs', 0)} InsightPulse run(s) across "
        f"{counts.get('cases', 0)} benchmark case(s): {counts.get('pass', 0)} passed, "
        f"{counts.get('partial', 0)} partial, {counts.get('fail', 0)} failed, "
        f"{counts.get('error', 0)} errored."
    ]
    if isinstance(overall, (int, float)):
        parts.append(f"The aggregate quality score is {overall:.1%}.")
    g = aggregate.get(M.GROUNDEDNESS) or {}
    h = aggregate.get(M.HALLUCINATION_RATE) or {}
    if g.get("available") and h.get("available"):
        parts.append(
            f"Groundedness averaged {float(g['value']):.1%} with a hallucination rate of "
            f"{float(h['value']):.1%} over the factual claims examined."
        )
    r = aggregate.get(M.RECOVERY_RATE) or {}
    if r.get("available"):
        parts.append(f"Injected failures were recovered at {float(r['value']):.1%}.")
    sim = (suite.get("provenance") or {}).get("simulation_mode")
    if sim:
        parts.append(
            "All runs used deterministic simulation fixtures, so the benchmark is "
            "repeatable; measured consistency reflects that determinism."
        )
    return " ".join(parts)


def _outcome_rules() -> list[str]:
    return [
        "PASS — every configured quality gate met.",
        "PARTIAL — one or more soft gates missed, no critical breach.",
        "FAIL — a critical gate breached (hallucination above the critical ceiling, "
        "groundedness below the critical floor, an unsupported conclusion asserted, "
        "a required recovery not achieved, or an injected contradiction missed).",
        "ERROR — the execution raised before producing a result.",
    ]


def _recommendations(
    aggregate: dict[str, Any], failures: list[dict[str, Any]], suite: dict[str, Any]
) -> list[str]:
    out: list[str] = []
    for name, target, message in (
        (M.EVIDENCE_QUALITY, 0.70, "Raise evidence quality by adding more independent, "
                                   "higher-credibility providers for the weaker scenarios."),
        (M.TASK_COMPLETION, 0.95, "Some required subtasks were not verifiably completed; "
                                  "review the per-subtask evidence on partial cases."),
        (M.UNCERTAINTY_HANDLING, 0.90, "Calibration could be tightened on cases where "
                                       "confidence and evidence strength diverged."),
    ):
        entry = aggregate.get(name) or {}
        if entry.get("available") and isinstance(entry.get("value"), (int, float)):
            if float(entry["value"]) < target:
                out.append(message)
    for f in failures:
        out.append(
            f"Investigate {f['case_id']} ({f['scenario_type']}): "
            f"{'; '.join(f.get('reasons') or []) or f.get('outcome')}."
        )
    if (suite.get("regression") or {}).get("regressions"):
        out.append("Address the regressions listed above before the next evaluation cycle.")
    if not out:
        out.append("No action required: every measured metric met its configured target.")
    return out


def _v(run: dict[str, Any], metric: str) -> float | None:
    m = (run.get("metrics") or {}).get(metric) or {}
    return m.get("value") if m.get("available") else None


def _fmt(value: Any, unit: str) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{int(value)}ms" if unit == "ms" else f"{float(value):.1%}"


def _pct(value: Any) -> str:
    return f"{float(value):.0%}" if isinstance(value, (int, float)) else "n/a"


def _ms(value: Any) -> str:
    return f"{int(value)}ms" if isinstance(value, (int, float)) else "n/a"


def _esc(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
