/* Multi-agent execution view.
 *
 * Renders the orchestrator's plan, each specialist's contribution, and the
 * collaboration events — all from the real `execution_plan`, `agents` and
 * `collaboration_events` the backend returns. Nothing here is decorative: if the
 * orchestrator skipped an agent, this panel shows which one and why.
 */

import { esc } from "../core/dom.js";
import { truncate } from "../core/format.js";

const COVERAGE = {
  live:        { label: "LIVE",             tone: "green" },
  partial:     { label: "PARTIAL COVERAGE", tone: "amber" },
  simulated:   { label: "SIMULATED",        tone: "amber" },
  unavailable: { label: "UNAVAILABLE",      tone: "red"   },
};

const STATUS = {
  completed: { label: "COMPLETED", tone: "green" },
  partial:   { label: "PARTIAL",   tone: "amber" },
  degraded:  { label: "DEGRADED",  tone: "amber" },
  skipped:   { label: "SKIPPED",   tone: "slate" },
  failed:    { label: "FAILED",    tone: "red"   },
};

const KIND = {
  follow_up:     { label: "Follow-up task",   icon: "📤" },
  corroboration: { label: "Cross-validated",  icon: "✅" },
  handoff:       { label: "Context linked",   icon: "🔗" },
  merge:         { label: "Merged",           icon: "🧩" },
  gap_fill:      { label: "Gap filled",       icon: "🧭" },
};

/* Status glyph for the pipeline nodes. */
const GLYPH = {
  completed: "✓", partial: "⚠", degraded: "⚠", skipped: "—", failed: "✕",
};

/* Short role labels for the pipeline. The backend `responsibility` is full
 * prose — good in the detail card, too long for a flow node. */
const ROLE = {
  research_agent:    "Research papers, trends & patents",
  competitive_agent: "Competitors & live market intelligence",
  orchestrator:      "Workflow control & agent delegation",
};

/* Annotate a tool with the provider that makes it recognisable. */
const TOOL_NOTE = { web_search: "Tavily" };
const toolLabel = (t) => (TOOL_NOTE[t] ? `${t} (${TOOL_NOTE[t]})` : t);

export function renderMultiAgent(result) {
  const plan = result.execution_plan || [];
  const agents = result.agents || [];
  const events = result.collaboration_events || [];

  if (!agents.length) {
    return `<p class="muted small">No multi-agent data was recorded for this run.</p>`;
  }

  const specialists = agents.filter((a) => a.agent !== "orchestrator");
  const orchestrator = agents.find((a) => a.agent === "orchestrator");
  const selected = plan.filter((p) => p.selected);
  const skipped = plan.filter((p) => !p.selected);

  return `
  <div class="ma">
    <div class="ma-head">
      <span class="eyebrow">Multi-agent execution</span>
      <p class="ma-lead">
        The orchestrator selected <b>${selected.length}</b> of <b>${plan.length}</b>
        specialist${plan.length === 1 ? "" : "s"} from your goal,
        ran ${specialists.length} of them, and recorded
        <b>${events.length}</b> collaboration event${events.length === 1 ? "" : "s"}.
      </p>
    </div>

    ${pipeline(orchestrator, specialists, selected, events, result)}

    ${planBlock(selected, skipped)}

    <div class="ma-cards">
      ${specialists.map(agentCard).join("")}
      ${orchestrator ? orchestratorCard(orchestrator) : ""}
    </div>

    ${collabBlock(events)}
  </div>`;
}

/* ── execution pipeline ─────────────────────────────────────
 * orchestrator → specialists → collaboration → combined report,
 * with an explicit arrow between each stage.
 */
function pipeline(orchestrator, specialists, selected, events, result) {
  const corroborated = result.metrics?.corroborated_findings ?? 0;
  const stages = [];

  if (orchestrator) {
    stages.push(node({
      icon: orchestrator.icon || "🧠",
      name: orchestrator.name,
      accent: orchestrator.accent || "purple",
      status: orchestrator.status,
      meta: [["Decision", `Selected ${selected.length} specialized agent${
        selected.length === 1 ? "" : "s"}`]],
    }));
  }

  if (specialists.length) {
    stages.push(`<div class="ma-lane">${specialists.map((a) => node({
      icon: a.icon || "•",
      name: a.name,
      accent: a.accent || "blue",
      status: a.status,
      meta: [
        ["Role", ROLE[a.agent] || truncate(a.responsibility || "", 72)],
        ["Tools", (a.tools_used || []).map(toolLabel).join(", ") || "none called"],
      ],
    })).join("")}</div>`);
  }

  stages.push(node({
    icon: events.length ? "🤝" : "○",
    name: "Collaboration",
    accent: events.length ? "purple" : "slate",
    meta: events.length
      ? [["", `${events.length} collaboration event${events.length === 1 ? "" : "s"}`],
         ["", `${corroborated} finding${corroborated === 1 ? "" : "s"} corroborated`]]
      : [["", "Not required — one specialist covered this goal"]],
  }));

  stages.push(node({
    icon: "📊",
    name: "Combined Intelligence Report",
    accent: "green",
    meta: [["", `${result.findings?.length ?? 0} findings · ${
      result.insights?.length ?? 0} prioritized insight${
      (result.insights?.length ?? 0) === 1 ? "" : "s"}`]],
  }));

  return `<div class="ma-pipe">${
    stages.join(`<div class="ma-arrow" aria-hidden="true">↓</div>`)}</div>`;
}

function node({ icon, name, accent, status, meta }) {
  const st = status ? STATUS[status] || STATUS.completed : null;
  const badge = st
    ? `<span class="ma-node-status tone-${st.tone}">${
        esc(GLYPH[status] || "✓")} ${esc(titleCase(st.label))}</span>`
    : "";
  return `
  <div class="ma-node accent-${esc(accent)}">
    <div class="ma-node-head">
      <span class="ma-icon">${esc(icon)}</span>
      <span class="ma-node-name">${esc(name)}</span>
      ${badge}
    </div>
    <dl class="ma-node-meta">
      ${meta.map(([k, v]) => `
        ${k ? `<dt>${esc(k)}:</dt>` : `<dt class="is-bullet"></dt>`}
        <dd>${esc(v)}</dd>`).join("")}
    </dl>
  </div>`;
}

const titleCase = (s) => s.charAt(0) + s.slice(1).toLowerCase();

/* ── plan: who was chosen, and who wasn't ───────────────── */
function planBlock(selected, skipped) {
  const row = (p) => `
    <li class="${p.selected ? "is-on" : "is-off"}">
      <span class="pl-agent">${esc(p.icon || "•")} ${esc(p.name)}</span>
      <span class="badge tone-${p.selected ? "green" : "slate"}">
        ${p.selected ? "SELECTED" : "NOT SELECTED"}</span>
      <span class="pl-reason">${esc(p.reason)}</span>
    </li>`;
  return `
  <div class="ma-plan">
    <span class="mini-label">Orchestrator's agent-selection decisions</span>
    <ul>${[...selected, ...skipped].map(row).join("")}</ul>
  </div>`;
}

/* ── per-agent card ─────────────────────────────────────── */
function agentCard(a) {
  const st = STATUS[a.status] || STATUS.completed;
  const cov = COVERAGE[a.coverage] || COVERAGE.live;

  const extras = [];
  if (a.research_trends?.length) extras.push(["Recurring themes", a.research_trends.slice(0, 4).join(", ")]);
  if (a.key_developments?.length) extras.push(["Key developments", a.key_developments.slice(0, 2).map((d) => truncate(d, 64)).join(" · ")]);
  if (a.competitors_analyzed?.length) extras.push(["Companies analysed", a.competitors_analyzed.slice(0, 5).join(", ")]);
  if (a.market_signals?.length) extras.push(["Market signals", a.market_signals.slice(0, 4).join(", ")]);
  if (a.degraded_providers?.length) extras.push(["Degraded providers", a.degraded_providers.map((d) => d.provider).join(", ")]);

  return `
  <article class="ma-card accent-${esc(a.accent || "blue")}">
    <div class="ma-card-head">
      <span class="ma-card-title">${esc(a.icon || "•")} ${esc(a.name)}</span>
      <span class="ma-badges">
        <span class="badge tone-${st.tone}">${esc(st.label)}</span>
        <span class="badge tone-${cov.tone}">${esc(cov.label)}</span>
      </span>
    </div>
    <p class="ma-resp">${esc(a.responsibility)}</p>

    <div class="ma-stats">
      <span><b>${a.findings_count ?? 0}</b>findings</span>
      <span><b>${a.relevant_count ?? 0}</b>relevant</span>
      <span><b>${Math.round((a.confidence || 0) * 100)}%</b>confidence</span>
      <span><b>${a.corroborated ?? 0}</b>cross-validated</span>
    </div>

    ${a.tools_used?.length ? `<div class="ma-tools">
      <span class="mini-label">Tools used</span>
      <div class="feed-tags">${a.tools_used.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
    </div>` : ""}

    ${a.sources_checked?.length ? `<div class="ma-tools">
      <span class="mini-label">Providers queried</span>
      <div class="feed-tags">${a.sources_checked.map((p) => `<span class="tag">${esc(p)}</span>`).join("")}</div>
    </div>` : ""}

    ${a.summary ? `<p class="ma-summary">${esc(a.summary)}</p>` : ""}
    ${extras.map(([k, v]) => `<p class="ma-extra"><b>${esc(k)}:</b> ${esc(v)}</p>`).join("")}
    ${a.errors?.length ? `<p class="ma-extra err"><b>Errors handled:</b> ${esc(a.errors.join("; "))}</p>` : ""}
  </article>`;
}

function orchestratorCard(o) {
  return `
  <article class="ma-card accent-purple is-orchestrator">
    <div class="ma-card-head">
      <span class="ma-card-title">${esc(o.icon || "🧠")} ${esc(o.name)}</span>
      <span class="badge tone-purple">COORDINATOR</span>
    </div>
    <p class="ma-resp">${esc(o.responsibility)}</p>
    <ul class="ma-bullets">
      ${(o.bullets || []).map((b) => `<li>${esc(b)}</li>`).join("")}
    </ul>
  </article>`;
}

/* ── collaboration events ───────────────────────────────── */
function collabBlock(events) {
  if (!events.length) {
    return `<div class="ma-collab">
      <span class="mini-label">Collaboration events</span>
      <p class="muted small">No cross-agent collaboration was required for this goal —
        a single specialist covered it end to end.</p>
    </div>`;
  }
  return `
  <div class="ma-collab">
    <span class="mini-label">Collaboration events (${events.length})</span>
    <ul>
      ${events.map((e) => {
        const k = KIND[e.kind] || { label: e.kind, icon: "•" };
        return `<li>
          <span class="ce-kind">${esc(k.icon)} ${esc(k.label)}</span>
          <span class="ce-body">
            <b>${esc(e.summary)}</b>
            <span>${esc(e.detail)}</span>
            ${e.participants?.length ? `<span class="ce-who">${
              esc(e.participants.map(shortAgent).join(" → "))}</span>` : ""}
          </span>
        </li>`;
      }).join("")}
    </ul>
  </div>`;
}

const shortAgent = (k) => ({
  research_agent: "Research Agent",
  competitive_agent: "Competitive Agent",
  orchestrator: "Orchestrator",
}[k] || k);
