/* Context & Memory view.
 *
 * Everything here is rendered from the `memory` block the backend returns — the
 * working-memory version counter, the plan step statuses, the facts that were
 * retained, which context each agent actually received, and what long-term memory
 * was retrieved or consolidated.
 *
 * There is no placeholder content anywhere in this file. If nothing was retrieved,
 * it says so; a fabricated "3 memories found" would be worse than an empty state,
 * because the whole point of the panel is to be evidence.
 */

import { esc } from "../core/dom.js";
import { truncate } from "../core/format.js";

const IMPORTANCE_TONE = {
  CRITICAL: "red",
  HIGH: "green",
  MEDIUM: "amber",
  LOW: "slate",
};

const STEP_TONE = {
  completed: "green",
  in_progress: "amber",
  pending: "slate",
  skipped: "slate",
  failed: "red",
};

const STEP_GLYPH = {
  completed: "✓",
  in_progress: "◐",
  pending: "○",
  skipped: "—",
  failed: "✕",
};

const AGENT_ICON = {
  research_agent: "🔬",
  competitive_agent: "🏢",
  orchestrator: "🧠",
};

const AGENT_NAME = {
  research_agent: "Research Intelligence Agent",
  competitive_agent: "Competitive Intelligence Agent",
  orchestrator: "Intelligence Orchestrator",
};

/* Which working-memory milestones to show as a progression, and how to decide
 * whether each one actually happened. Read from the real timeline. */
const MILESTONES = [
  ["Task Context", "Captured", (t) => t.some((e) => e.event === "task_context_captured")],
  ["Execution Plan", "Stored", (t) => t.some((e) => e.event === "plan_stored")],
  ["Agent Findings", "Retained", (t) => t.some((e) => e.event === "agent_findings_recorded")],
  ["Shared Context", "Passed between agents", (t, m) => (m.shared_events || 0) > 0],
  ["Cross-Agent Analysis", "Consolidated", (t) => t.some((e) => e.event === "compared_with_baseline" || e.event === "memory_consolidated")],
];

export function renderMemory(result) {
  const mem = result?.memory;
  if (!mem || !mem.available || !mem.working) {
    return `<p class="muted small">No context or memory data was recorded for this run.</p>`;
  }

  const w = mem.working;
  const lt = mem.long_term || {};
  const change = mem.change || {};
  const ctx = w.task_context;

  return `
  <div class="mem">
    <div class="mem-head">
      <span class="eyebrow">Context &amp; memory</span>
      <p class="mem-lead">
        Working memory reached <b>version ${w.version}</b>, retaining
        <b>${w.fact_count}</b> fact${w.fact_count === 1 ? "" : "s"}
        (${w.important_fact_count} important) across
        <b>${(w.plan_steps || []).length}</b> plan step${(w.plan_steps || []).length === 1 ? "" : "s"}.
        ${lt.retrieved_count
          ? `<b>${lt.retrieved_count}</b> relevant memory item${lt.retrieved_count === 1 ? "" : "s"} were retrieved from previous runs.`
          : `No previous context was retrieved for this goal.`}
      </p>
    </div>

    ${taskContextBlock(ctx)}
    ${progressionBlock(w, result)}
    ${planBlock(w)}
    ${agentContextBlock(result)}
    ${factsBlock(w)}
    ${longTermBlock(lt, change)}
    ${compressionBlock(w)}
    ${notesBlock(w, lt)}
  </div>`;
}

/* ── current task context ────────────────────────────────── */
function taskContextBlock(ctx) {
  if (!ctx) return "";
  const chips = [
    ...(ctx.topics || []).map((t) => ({ text: t, cls: "topic" })),
    ...(ctx.domain_labels || []).map((d) => ({ text: d, cls: "domain" })),
    ...(ctx.competitors || []).map((c) => ({ text: c, cls: "company" })),
  ];
  const meta = [];
  if (ctx.time_scope && ctx.time_scope !== "unspecified") meta.push(["Time scope", ctx.time_scope]);
  if ((ctx.constraints || []).length) meta.push(["Constraints", ctx.constraints.join("; ")]);
  if (ctx.continuation) {
    meta.push(["Continuation", ctx.subjectless
      ? "goal refers back to earlier monitoring — subject restored from memory"
      : "goal continues earlier monitoring"]);
  }
  meta.push(["Extracted by", ctx.author || "heuristic"]);

  return `
  <div class="mem-block">
    <span class="mini-label">Current task context</span>
    <div class="mem-chips">
      ${chips.length
        ? chips.map((c) => `<span class="mem-chip is-${c.cls}">${esc(c.text)}</span>`).join("")
        : `<span class="muted small">No explicit topic detected in the goal.</span>`}
    </div>
    <dl class="mem-meta">
      ${meta.map(([k, v]) => `<dt>${esc(k)}:</dt><dd>${esc(v)}</dd>`).join("")}
    </dl>
  </div>`;
}

/* ── working-memory progression ──────────────────────────── */
function progressionBlock(w, result) {
  const timeline = w.timeline || [];
  const sharedEvents = (result?.agents || []).filter(
    (a) => (a.context_shared_from || []).length > 0,
  ).length;
  const rows = MILESTONES.map(([label, done, test], i) => {
    const ok = test(timeline, { shared_events: sharedEvents });
    return `
      <li class="${ok ? "is-done" : "is-pending"}">
        <span class="mem-step-n">${i + 1}</span>
        <span class="mem-step-body">
          <b>${esc(label)}</b>
          <span>${ok ? `✓ ${esc(done)}` : "not reached in this run"}</span>
        </span>
      </li>`;
  }).join("");

  return `
  <div class="mem-block">
    <span class="mini-label">Working memory progression</span>
    <ol class="mem-steps">${rows}</ol>
    <p class="muted small">
      ${timeline.length} memory update${timeline.length === 1 ? "" : "s"} recorded,
      ending at version ${w.version}.
    </p>
  </div>`;
}

/* ── execution plan state ────────────────────────────────── */
function planBlock(w) {
  const steps = w.plan_steps || [];
  if (!steps.length) return "";
  return `
  <div class="mem-block">
    <span class="mini-label">Execution plan held in working memory</span>
    <ul class="mem-plan">
      ${steps.map((s) => {
        const tone = STEP_TONE[s.status] || "slate";
        return `
        <li>
          <span class="mem-plan-glyph tone-${tone}">${esc(STEP_GLYPH[s.status] || "○")}</span>
          <span class="mem-plan-body">
            <b>${esc(s.step_name)}</b>
            ${s.result_reference ? `<span>${esc(s.result_reference)}</span>` : ""}
          </span>
          <span class="badge tone-${tone}">${esc(s.status.replace("_", " ").toUpperCase())}</span>
        </li>`;
      }).join("")}
    </ul>
  </div>`;
}

/* ── what each agent actually received ───────────────────── */
function agentContextBlock(result) {
  const agents = (result?.agents || []).filter((a) => (a.context_received || []).length);
  if (!agents.length) return "";
  return `
  <div class="mem-block">
    <span class="mini-label">Context shared with each agent</span>
    <div class="mem-agents">
      ${agents.map((a) => `
        <article class="mem-agent accent-${esc(a.accent || "blue")}">
          <div class="mem-agent-head">
            <span>${esc(a.icon || AGENT_ICON[a.agent] || "•")} ${esc(a.name || AGENT_NAME[a.agent] || a.agent)}</span>
            ${a.memory_version ? `<span class="mem-ver">memory v${a.memory_version}</span>` : ""}
          </div>
          <ul class="mem-ticks">
            ${(a.context_received || []).map((c) => `<li>✓ ${esc(c)}</li>`).join("")}
          </ul>
          ${(a.context_shared_from || []).length ? `
            <p class="mem-shared">
              🔄 ${a.context_facts} finding${a.context_facts === 1 ? "" : "s"} carried over from
              ${esc((a.context_shared_from || []).map((k) => AGENT_NAME[k] || k).join(", "))}
            </p>` : ""}
          ${(a.context_focus || []).length ? `
            <p class="mem-focus">
              <b>Search focus from context:</b>
              ${(a.context_focus || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}
            </p>` : ""}
          ${(a.context_omitted || []).map((o) => `
            <p class="mem-omit">⊘ ${esc(o.why)}</p>`).join("")}
        </article>`).join("")}
    </div>
  </div>`;
}

/* ── retained facts ──────────────────────────────────────── */
function factsBlock(w) {
  const facts = w.facts || [];
  if (!facts.length) {
    return `
    <div class="mem-block">
      <span class="mini-label">Retained findings</span>
      <p class="muted small">Nothing cleared the importance floor for this run, so no
        findings were promoted into reusable context.</p>
    </div>`;
  }
  return `
  <div class="mem-block">
    <span class="mini-label">Retained findings (${w.fact_count} total, top ${Math.min(facts.length, 6)} shown)</span>
    <ul class="mem-facts">
      ${facts.slice(0, 6).map((f) => `
        <li>
          <span class="badge tone-${IMPORTANCE_TONE[f.importance] || "slate"}">${esc(f.importance)}</span>
          <span class="mem-fact-body">
            <b>${esc(truncate(f.text, 96))}</b>
            <span>
              ${esc(AGENT_NAME[f.source_agent] || f.source_agent || "unattributed")}
              ${f.simulated ? ` · <span class="mem-sim">SIMULATED</span>` : ""}
              ${(f.signals || []).length ? ` · ${esc((f.signals || []).join(", "))}` : ""}
            </span>
          </span>
        </li>`).join("")}
    </ul>
  </div>`;
}

/* ── long-term memory ────────────────────────────────────── */
function longTermBlock(lt, change) {
  const items = lt.retrieved || [];
  const cons = lt.consolidation || {};
  const store = lt.store || {};

  const retrieved = items.length
    ? `<ul class="mem-lt">
        ${items.map((m) => `
          <li>
            <span class="mem-lt-type">${esc(m.type_label || m.memory_type)}</span>
            <span class="mem-lt-body">
              <b>${esc(truncate(m.summary || m.content, 110))}</b>
              <span>
                ${m.source_run_id ? `from run ${esc(m.source_run_id)}` : ""}
                ${m.relevance ? ` · relevance ${Number(m.relevance).toFixed(2)}` : ""}
                ${m.recurrence > 1 ? ` · seen in ${m.recurrence} runs` : ""}
              </span>
            </span>
          </li>`).join("")}
      </ul>`
    : `<p class="muted small">No relevant previous context found for this run
        (${esc(lt.retrieval_status || "not attempted")}). Started with current task context.</p>`;

  const changeLine = change.compared
    ? `<p class="mem-change"><b>Compared with previous monitoring:</b>
        <span class="badge tone-${change.verdict === "TREND ACCELERATING" ? "amber" : "green"}">${esc(change.verdict)}</span>
        ${esc(change.detail || "")}</p>`
    : `<p class="muted small">No historical baseline was available, so no change
        comparison was made.</p>`;

  return `
  <div class="mem-block is-lt">
    <span class="mini-label">Long-term memory</span>
    ${retrieved}
    ${changeLine}
    <p class="mem-cons">
      ${Object.keys(cons).length
        ? `💾 Consolidated <b>${cons.stored ?? 0}</b> new item${(cons.stored ?? 0) === 1 ? "" : "s"}
           for future monitoring${cons.refreshed ? `, refreshed ${cons.refreshed}` : ""}${
             cons.rejected ? `, rejected ${cons.rejected} as not durable` : ""}.
           ${cons.persisted === false ? `<span class="mem-sim">held in process only — the store could not be written</span>` : ""}`
        : `Nothing was consolidated for this run.`}
      ${store.total !== undefined ? ` Store now holds ${store.total} item${store.total === 1 ? "" : "s"}.` : ""}
    </p>
  </div>`;
}

/* ── compression ─────────────────────────────────────────── */
function compressionBlock(w) {
  if (!w.compressions) return "";
  return `
  <div class="mem-block">
    <span class="mini-label">Context compression</span>
    <p class="small">
      ${w.compressions} compression pass${w.compressions === 1 ? "" : "es"} folded
      <b>${w.compressed_count}</b> lower-importance fact${w.compressed_count === 1 ? "" : "s"}
      into a summary. Important facts were kept verbatim.
    </p>
    ${w.narrative_summary ? `<p class="mem-narrative">${esc(w.narrative_summary)}</p>` : ""}
  </div>`;
}

/* ── honest degradation notes ────────────────────────────── */
function notesBlock(w, lt) {
  const notes = [...(w.notes || [])];
  if (lt.store && lt.store.degraded) notes.push(lt.store.degraded);
  if (!notes.length) return "";
  return `
  <div class="mem-block is-warn">
    <span class="mini-label">Memory status notes</span>
    <ul class="mem-notes">${notes.map((n) => `<li>⚠️ ${esc(n)}</li>`).join("")}</ul>
  </div>`;
}
