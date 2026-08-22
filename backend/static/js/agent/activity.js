/* Agent progress + activity timeline.
 *
 * Two audiences, one data source (the real SSE events):
 *   - a normal user sees a 6-step tracker and one plain-English status line
 *   - a judge opens "How the AI Agent Worked" and sees the full decision trail
 *
 * Nothing here exposes private chain-of-thought: every string rendered is the
 * agent's own user-facing activity-log output.
 */

import { esc, setHTML, setText } from "../core/dom.js";
import { CATEGORY, SIGNAL_LABEL, categoryLabel, toolHuman, toolLabel } from "../core/format.js";

export const STEPS = [
  { key: "goal",     label: "Understanding your goal" },
  { key: "plan",     label: "Creating search plan" },
  { key: "search",   label: "Searching relevant sources" },
  { key: "analyze",  label: "Analyzing findings" },
  { key: "trends",   label: "Detecting trends" },
  { key: "report",   label: "Preparing intelligence" },
];

/** Maps an activity phase to a tracker step. Forward-only, enforced by the caller. */
export function phaseToStep(entry) {
  switch (entry.phase) {
    case "start": case "goal": return "goal";
    case "plan": return "plan";
    case "decision": case "action": case "observation": return "search";
    // A thought before any tool call is still planning.
    case "thought": return entry.iteration ? "search" : "plan";
    case "final": return "analyze";
    case "insight": return entry.data?.priority_counts ? "report" : "trends";
    case "done": return "report";
    default: return null;
  }
}

/** Plain-language status line. Returns "" to leave the current line unchanged. */
export function humanMessage(entry) {
  const d = entry.data || {};
  switch (entry.phase) {
    case "start": return "Reading your tracking goal…";
    case "goal": return "Goal understood. Working out where to look.";
    case "plan": {
      const needs = (d.required_needs || []).map((n) => categoryLabel(n).toLowerCase());
      return needs.length ? `Plan ready — will check ${needs.join(", ")}.` : "Search plan ready.";
    }
    case "decision":
      return d.tool ? `Decided to search ${toolHuman(d.tool)}.` : "Deciding the next action…";
    case "action": return `Searching ${toolHuman(d.tool) || "sources"}…`;
    case "observation": {
      const rel = d.relevant ?? 0;
      const dup = d.duplicates ?? 0;
      if (dup > 0) {
        return `Found ${rel} relevant finding${rel === 1 ? "" : "s"}, skipped ${dup} duplicate${dup === 1 ? "" : "s"}.`;
      }
      return `Found ${rel} relevant finding${rel === 1 ? "" : "s"}.`;
    }
    case "thought": {
      const title = entry.title || "";
      if (/^holding back/i.test(title)) {
        const need = d.need ? categoryLabel(d.need).toLowerCase() : "some sources";
        return `Skipping ${need} for now — only worth searching if the evidence calls for it.`;
      }
      if (/need is now satisfied/i.test(title)) {
        const which = (title.match(/'([^']+)'/) || [])[1];
        return `Got enough on ${which ? categoryLabel(which).toLowerCase() : "that"}. Deciding what's next…`;
      }
      return title || "Comparing new results with what was already collected…";
    }
    case "warning": {
      const t = entry.title || "";
      if (/llm|reasoner|model/i.test(t)) return "";
      if (/simulation/i.test(t)) return "Using demo data for a fast offline run.";
      if (/degraded|provider/i.test(t)) return "One source was unavailable — continuing with the others.";
      if (/iteration limit/i.test(t)) return "Reached the step limit — summarizing what was found.";
      if (/unavailable/i.test(t)) return "A tool is unavailable — continuing with the rest.";
      return "";
    }
    case "error": return "A source failed. Continuing with the rest.";
    case "final": return "Enough information gathered. Prioritizing strategic impact…";
    case "insight":
      return d.priority_counts ? "Writing your intelligence report…" : "Ranking what matters most…";
    case "done": return "Intelligence ready.";
    default: return entry.title || "Working…";
  }
}

/* ── live tracker ───────────────────────────────────────── */
export function renderTracker(container) {
  setHTML(container, STEPS.map((s) => `
    <li data-step="${s.key}">
      <span class="step-dot" aria-hidden="true"></span>
      <span class="step-text" data-label>${esc(s.label)}</span>
    </li>`).join(""));
}

export function advanceTracker(container, stepKey, searchLabel) {
  const idx = STEPS.findIndex((s) => s.key === stepKey);
  if (idx < 0) return;
  STEPS.forEach((s, i) => {
    const li = container.querySelector(`li[data-step="${s.key}"]`);
    if (!li) return;
    const state = i < idx ? "done" : i === idx ? "active" : "";
    if (li.dataset.state !== state) li.dataset.state = state;
  });
  if (searchLabel) {
    const label = container.querySelector('li[data-step="search"] [data-label]');
    if (label) setText(label, searchLabel);
  }
}

export function completeTracker(container) {
  STEPS.forEach((s) => {
    const li = container.querySelector(`li[data-step="${s.key}"]`);
    if (li) li.dataset.state = "done";
  });
}

/** One compact live row appended per event — cheap and incremental. */
export function liveRow(entry) {
  const cls = entry.phase === "warning" ? "warn" : entry.phase === "error" ? "err" : "";
  const title = entry.title && entry.title !== entry.label ? ` — ${esc(entry.title)}` : "";
  return `<div class="tick ${cls}">
    <span aria-hidden="true">${esc(entry.icon)}</span>
    <span><b>${esc(entry.label)}</b>${title}${entry.detail ? `<em>${esc(entry.detail)}</em>` : ""}</span>
    <time>${(entry.elapsed_ms / 1000).toFixed(1)}s</time>
  </div>`;
}

/* ── judge-facing reasoning trail ───────────────────────── */
export function reasoningTrail(result) {
  const st = result.state || {};
  const m = result.metrics || {};
  const plan = st.plan || {};
  const steps = [];

  steps.push({ stage: "Goal understood", detail: st.user_goal || "" });

  const required = (plan.needs || []).filter((n) => n.required).map((n) => n.key);
  steps.push({
    stage: "Plan created",
    detail: `Information needs identified: ${required.join(", ") || "none"}.` +
      (plan.opening_move ? ` ${plan.opening_move}` : ""),
  });

  const held = (plan.needs || []).filter((n) => !n.required).map((n) => n.key);
  if (held.length) {
    steps.push({
      stage: `Deferred: ${held.join(", ")}`,
      detail: "Held back deliberately — searched only if the evidence justified it. " +
              "This is the agent choosing, not a fixed pipeline.",
    });
  }

  (st.tool_calls || []).forEach((call, i) => {
    const decision = (st.decisions || [])[i] || {};
    const obs = (st.observations || []).find((o) => o.iteration === call.iteration);

    steps.push({
      stage: `Decision ${i + 1} → ${toolLabel(call.tool)}`,
      detail: decision.reasoning || call.reasoning || "",
      tone: "decision",
    });
    steps.push({
      stage: `Tool called: ${toolLabel(call.tool)}`,
      detail: describeInput(call.tool_input || {}),
      tone: "action",
    });
    steps.push({
      stage: `Observed ${call.items_returned} result${call.items_returned === 1 ? "" : "s"}`,
      detail: (obs && obs.summary) || call.note || "",
      tone: "observe",
    });
    if (obs) {
      const sig = (obs.signals || []).map((s) => SIGNAL_LABEL[s] || s).join(", ");
      steps.push({
        stage: "Analyzed relevance",
        detail: `${obs.relevant_items} relevant; yield judged "${obs.yield_quality}"` +
                (sig ? `; signals: ${sig}` : "") + ".",
        tone: "analyze",
      });
    }
  });

  steps.push({
    stage: "Decided collection was complete",
    detail: st.stop_reason || st.final_decision || "",
    tone: "decision",
  });
  const c = m.priority_counts || {};
  steps.push({
    stage: `Generated ${m.insights || 0} prioritized insight${m.insights === 1 ? "" : "s"}`,
    detail: `${c.HIGH || 0} high, ${c.MEDIUM || 0} medium, ${c.LOW || 0} low.`,
    tone: "done",
  });

  return `<ol class="trail">${steps.map((s) => `
    <li class="${s.tone ? `t-${s.tone}` : ""}">
      <b>${esc(s.stage)}</b>${s.detail ? `<span>${esc(s.detail)}</span>` : ""}
    </li>`).join("")}</ol>`;
}

function describeInput(input) {
  const bits = [];
  if (input.query) bits.push(`query "${input.query}"`);
  if ((input.keywords || []).length) bits.push(`keywords: ${input.keywords.join(", ")}`);
  if ((input.competitors || []).length) bits.push(`companies: ${input.competitors.join(", ")}`);
  if (input.since_days) bits.push(`window: last ${input.since_days} days`);
  return bits.join(" · ");
}

/* ── technical details (inside the trail) ───────────────── */
export function technicalDetails(result) {
  const m = result.metrics || {};
  const llm = m.llm || {};
  const stats = [
    ["Reasoning steps", `${m.iterations ?? "–"} / ${m.max_iterations ?? "–"}`],
    ["Tool calls", m.tool_calls ?? "–"],
    ["Findings", m.findings_total ?? "–"],
    ["Relevant", m.findings_relevant ?? "–"],
    ["Duplicates cut", m.duplicates_suppressed ?? 0],
    ["Duration", m.duration_ms != null ? `${(m.duration_ms / 1000).toFixed(1)}s` : "–"],
    ["Reasoner", m.reasoner || "–"],
    ["Model calls", llm.calls ?? 0],
    ["Est. cost", `$${llm.cost_usd ?? 0}`],
    ["Errors handled", m.errors ?? 0],
  ];

  const log = (result.activity_log || []).map((e) => `
    <div class="tech-line ${e.phase === "warning" ? "warn" : e.phase === "error" ? "err" : ""}">
      <span aria-hidden="true">${esc(e.icon)}</span>
      <span><b>${esc(e.label)}${e.iteration ? ` · step ${e.iteration}` : ""}${
        e.title && e.title !== e.label ? ` — ${esc(e.title)}` : ""}</b>${
        e.detail ? `<em>${esc(e.detail)}</em>` : ""}</span>
      <time>${(e.elapsed_ms / 1000).toFixed(1)}s</time>
    </div>`).join("");

  const observations = (result.state?.observations || []).map((o) => `
    <div class="tech-line">
      <span aria-hidden="true">👁</span>
      <span><b>step ${o.iteration} · ${esc(o.tool)} · ${esc(o.yield_quality)}</b>
      <em>${esc(o.summary)}</em></span><time></time>
    </div>`).join("") || '<p class="muted small">No observations recorded.</p>';

  return `
  <div class="tech-stats">
    ${stats.map(([k, v]) => `<div class="tech-stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("")}
  </div>
  <div class="tech-tabs" role="tablist">
    <button type="button" class="tech-tab is-on" data-tech="log" role="tab">Activity log</button>
    <button type="button" class="tech-tab" data-tech="obs" role="tab">Observations</button>
  </div>
  <div class="tech-panel" data-tech-panel="log">${log}</div>
  <div class="tech-panel" data-tech-panel="obs" hidden>${observations}</div>`;
}

export const CATEGORY_KEYS = Object.keys(CATEGORY);
