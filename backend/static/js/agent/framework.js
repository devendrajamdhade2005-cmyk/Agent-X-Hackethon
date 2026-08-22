/* ⚙️ Autonomous Agent Framework view (Task 5 — LangGraph).
 *
 * Self-contained: it drives the /api/agent/graph endpoints, streams the framework
 * events live, animates a compact graph visualisation, and renders a panel that
 * reflects *what actually happened* — plan, parallel execution, failure recovery,
 * conflict resolution, verification, replanning, self-evaluation, checkpoints and
 * the resource budget. Nothing here is decorative: every value comes from the run.
 */

import { $, esc, setHTML, show } from "../core/dom.js";
import * as api from "../core/api.js";

let booted = false;
let controller = null;
const statuses = {};        // node id -> status
const live = { fallbacks: 0, conflicts: 0, replans: 0, verifies: 0, tools: 0 };

/* Node → { fw_event : status } transitions that drive the visualisation. */
const NODES = [
  ["understand", "Understand"],
  ["plan", "Dynamic Planner"],
  ["decompose", "Task Decomposer"],
  ["resource_check", "Resource / Policy"],
  ["dispatch", "Dynamic Router"],
  ["research_agent", "🔬 Research Agent"],
  ["competitive_agent", "🏢 Competitive Agent"],
  ["observer", "Observer"],
  ["conflict_resolution", "Conflict Resolution"],
  ["self_evaluator", "Self-Evaluator"],
  ["verify", "Verification"],
  ["replan", "Replanner"],
  ["finalize", "Final Synthesis"],
  ["memory_update", "Memory Update"],
];
const AGENT_NODES = new Set(["research_agent", "competitive_agent"]);

const EVENT_ICON = {
  planner_started: "🎯", plan_created: "🗂", task_decomposed: "🧩",
  parallel_tasks_started: "⚡", agent_started: "🤖",
  tool_started: "🔧", tool_succeeded: "✓", tool_failed: "⚠️", tool_timeout: "⏱",
  retry_started: "🔄", fallback_started: "↩", fallback_succeeded: "✓",
  evaluation_started: "🧭", evaluation_completed: "🧭",
  conflict_detected: "⚠️", conflict_resolved: "✓",
  verification_started: "🔎", verification_completed: "✓",
  replan_triggered: "↻", checkpoint_saved: "💾",
  budget_constraint_detected: "💰", deadlock_detected: "🛑",
  resource_status: "📊", final_synthesis_started: "📊",
  run_completed: "✅", memory_updated: "🧠",
};

const DEMO_GOAL =
  "Analyze important AI-agent research and competitor developments and determine " +
  "whether current evidence indicates meaningful strategic competitive movement.";

/* ─────────────────────────────────────────────────────────── */
export function initFramework(host) {
  if (booted) return;
  booted = true;

  setHTML(host, `
    <section class="panel fw-panel">
      <div class="panel-head"><div>
        <span class="eyebrow">LangGraph runtime</span>
        <h2>⚙️ Autonomous Agent Framework</h2>
        <p class="panel-sub">A stateful LangGraph StateGraph: dynamic planning, parallel
          agents, checkpointing, failure recovery, conflict resolution, verification,
          self-evaluation and autonomous replanning — shown live as it runs.</p>
      </div></div>

      <div class="fw-controls">
        <label class="fw-field">
          <span>Goal</span>
          <input id="fw-goal" type="text" value="${esc(DEMO_GOAL)}" />
        </label>
        <label class="fw-field fw-narrow">
          <span>Competitors</span>
          <input id="fw-comp" type="text" value="OpenAI, Anthropic" />
        </label>
        <label class="fw-field fw-narrow">
          <span>Scenario</span>
          <select id="fw-scenario">
            <option value="full">Full adversarial</option>
            <option value="tool_failure">Tool failure only</option>
            <option value="conflict">Evidence conflict</option>
            <option value="budget">Budget constraint</option>
          </select>
        </label>
        <div class="fw-buttons">
          <button type="button" class="btn-primary" id="fw-run">▶ Run LangGraph Scan</button>
          <button type="button" class="btn-ghost" id="fw-adv">🧪 Run Adversarial Test</button>
          <button type="button" class="btn-ghost" id="fw-stop" hidden>Stop</button>
        </div>
      </div>
      <p class="fw-status" id="fw-status">Ready. Runs offline in simulation mode — repeatable and safe.</p>

      <div class="fw-grid">
        <div class="fw-viz-wrap">
          <h3 class="fw-h3">Execution graph</h3>
          <div class="fw-viz" id="fw-viz"></div>
          <div class="fw-legend">
            <span><i class="fw-dot is-completed"></i>done</span>
            <span><i class="fw-dot is-running"></i>running</span>
            <span><i class="fw-dot is-recovered"></i>recovered</span>
            <span><i class="fw-dot is-failed"></i>failed</span>
            <span><i class="fw-dot is-pending"></i>pending</span>
          </div>
        </div>
        <div class="fw-live-wrap">
          <h3 class="fw-h3">Live framework events</h3>
          <div class="fw-live" id="fw-live"></div>
        </div>
      </div>

      <div id="fw-report"></div>
    </section>`);

  renderViz();
  $("fw-run").addEventListener("click", () => run("/api/agent/graph/run/stream", false));
  $("fw-adv").addEventListener("click", () => run("/api/agent/graph/adversarial", true));
  $("fw-stop").addEventListener("click", () => controller && controller.abort());
}

/* ─────────────────────────────────────────────────────────── */
function resetRun() {
  for (const [id] of NODES) statuses[id] = "pending";
  live.fallbacks = live.conflicts = live.replans = live.verifies = live.tools = 0;
  setHTML($("fw-live"), "");
  setHTML($("fw-report"), "");
  renderViz();
}

async function run(path, adversarial) {
  resetRun();
  const goal = $("fw-goal").value.trim() || DEMO_GOAL;
  const competitors = $("fw-comp").value.split(",").map((s) => s.trim()).filter(Boolean);
  const scenario = $("fw-scenario").value;
  const payload = {
    goal, competitors, keywords: ["AI agents"],
    simulation_mode: true, adversarial, scenario,
  };

  $("fw-run").disabled = true;
  $("fw-adv").disabled = true;
  show($("fw-stop"), true);
  setText("fw-status", adversarial
    ? `Running adversarial scenario "${scenario}" — tools will fail and evidence will conflict; the graph must recover.`
    : "Running the LangGraph orchestration…");

  controller = new AbortController();
  try {
    const result = await api.runGraphStream(path, payload, onEvent, controller.signal);
    renderReport(result);
    setText("fw-status", `✅ ${result.status} — objective completed autonomously.`);
  } catch (err) {
    if (err.name === "AbortError") setText("fw-status", "Stopped.");
    else setText("fw-status", `Run failed: ${err.message}`);
  } finally {
    controller = null;
    $("fw-run").disabled = false;
    $("fw-adv").disabled = false;
    show($("fw-stop"), false);
  }
}

function onEvent(event) {
  if (event.type !== "activity" || !event.entry) return;
  const e = event.entry;
  const fw = e.data && e.data.fw_event;
  if (!fw) return;
  applyTransition(fw, e);
  appendLive(fw, e);
  renderViz();
}

function applyTransition(fw, e) {
  const set = (id, s) => { if (statuses[id] !== "completed" || s === "completed" || s === "recovered") statuses[id] = s; };
  switch (fw) {
    case "planner_started": set("understand", "running"); break;
    case "plan_created": set("understand", "completed"); set("plan", "completed"); set("decompose", "running"); break;
    case "task_decomposed": set("decompose", "completed"); set("resource_check", "running"); break;
    case "resource_status": set("resource_check", "completed"); break;
    case "budget_constraint_detected": set("resource_check", "completed"); break;
    case "parallel_tasks_started":
      set("dispatch", "completed");
      (e.data.agents || []).forEach((a) => set(a, "running"));
      break;
    case "agent_started": {
      const a = e.data.agent;
      if (a && AGENT_NODES.has(a)) set(a, statuses[a] === "running" ? "completed" : "running");
      break;
    }
    case "tool_failed": case "tool_timeout": { const a = e.data.agent; if (a) set(a, "recovered"); break; }
    case "fallback_succeeded": { live.fallbacks++; const a = e.data.agent; if (a) set(a, "recovered"); break; }
    case "conflict_detected": set("observer", "completed"); set("conflict_resolution", "running"); live.conflicts++; break;
    case "conflict_resolved": set("conflict_resolution", "completed"); break;
    case "evaluation_started": set("observer", "completed"); set("conflict_resolution", "completed"); set("self_evaluator", "running"); break;
    case "evaluation_completed": set("self_evaluator", "completed"); break;
    case "verification_started": set("verify", "running"); break;
    case "verification_completed": set("verify", "completed"); live.verifies++; break;
    case "replan_triggered": set("replan", "completed"); live.replans++; break;
    case "final_synthesis_started": set("finalize", "running"); break;
    case "memory_updated": set("memory_update", "completed"); break;
    case "run_completed":
      set("finalize", "completed"); set("memory_update", "completed");
      for (const [id] of NODES) if (statuses[id] === "pending") statuses[id] = "skipped";
      break;
    default: break;
  }
}

/* ─────────────────────────────────────────────────────────── */
function renderViz() {
  const rows = [];
  for (const [id, label] of NODES) {
    if (id === "competitive_agent") continue; // rendered beside research_agent
    if (id === "research_agent") {
      rows.push(`<div class="fw-branch">
        ${nodeBox("research_agent", "🔬 Research Agent")}
        ${nodeBox("competitive_agent", "🏢 Competitive Agent")}
      </div>`);
      continue;
    }
    rows.push(nodeBox(id, label));
  }
  setHTML($("fw-viz"), rows.join('<div class="fw-arrow">↓</div>'));
}

function nodeBox(id, label) {
  const s = statuses[id] || "pending";
  return `<div class="fw-node is-${s}" data-node="${id}">
    <span class="fw-node-label">${esc(label)}</span>
    <span class="fw-node-badge">${s}</span>
  </div>`;
}

function appendLive(fw, e) {
  const icon = EVENT_ICON[fw] || "•";
  const row = document.createElement("div");
  row.className = "fw-ev" + (/(failed|timeout|conflict_detected|deadlock|budget)/.test(fw) ? " is-warn" : "");
  row.innerHTML = `<span class="fw-ev-i">${icon}</span>
    <span class="fw-ev-b"><b>${esc(e.title || fw)}</b>${e.detail ? `<em>${esc(e.detail)}</em>` : ""}</span>`;
  const box = $("fw-live");
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

/* ─────────────────────────────────────────────────────────── */
function renderReport(result) {
  const f = result.framework || {};
  const ev = f.evaluation || {};
  const res = f.resource || {};
  const hyp = (f.hypotheses || [])[0] || {};
  const conflicts = f.conflicting_evidence || [];

  const card = (title, body) => `<div class="fw-card"><h4>${title}</h4>${body}</div>`;
  const line = (k, v) => `<div class="fw-line"><span>${esc(k)}</span><b>${v}</b></div>`;

  const recovery = (f.fallback_history || []).length
    ? (f.fallback_history || []).map((fb) =>
        `<div class="fw-line"><span>⚠ ${esc(fb.tool)} · ${esc(fb.from)} failed</span><b class="ok">✓ recovered via ${esc(fb.to)}</b></div>`).join("")
    : `<p class="muted small">No tool failures in this run.</p>`;

  const conflictBody = conflicts.length
    ? conflicts.map((c) =>
        `<div class="fw-conflict">
          <div class="fw-line"><span>Subject</span><b>${esc((c.subject || "").slice(0, 60))}</b></div>
          <p class="small">A: ${esc(c.claim_a || "")}<br>B: ${esc(c.claim_b || "")}</p>
          <span class="tag ${c.resolved ? "tag-live" : "tag-sim"}">${esc(c.verdict || (c.resolved ? "RESOLVED" : "UNRESOLVED"))}</span>
          <p class="small muted">${esc(c.detail || "")}</p>
        </div>`).join("")
    : `<p class="muted small">No conflicting evidence detected.</p>`;

  setHTML($("fw-report"), `
    <div class="fw-report-grid">
      ${card("🎯 Planner", line("Plan version", f.plan_version || 1) +
        line("Selected agents", (f.selected_agents || []).join(", ") || "—") +
        line("Runtime", esc(f.runtime || "langgraph")))}
      ${card("⚡ Execution", line("Agents completed", (f.completed_agents || []).length) +
        line("Graph steps", f.graph_steps || 0) +
        line("Tool calls", (f.tool_executions || []).length))}
      ${card("🛠 Failure recovery", recovery)}
      ${card("🔎 Verification", line("Verifications", f.verify_count || 0) +
        line("Status", esc(f.verification_status || "n/a")) +
        line("Independent sources", (f.verification_findings || []).length))}
      ${card("↻ Replanning", line("Replans", f.replan_count || 0) +
        line("Final plan version", f.plan_version || 1))}
      ${card("🧭 Self-evaluation",
        line("Completion", pct(ev.completion_score)) +
        line("Evidence", pct(ev.evidence_score)) +
        line("Confidence", pct(ev.confidence_score != null ? ev.confidence_score : f.overall_confidence)))}
      ${card("💰 Resource budget",
        line("Tool calls", `${res.tool_calls || 0} / ${res.max_tool_calls || "—"}`) +
        line("Est. cost", `$${(res.estimated_cost || 0).toFixed(3)}`) +
        line("Under pressure", f.adversarial && f.adversarial.enabled ? "yes" : "no"))}
      ${card("💾 Checkpoints", (f.checkpoints || []).map((c) =>
        `<div class="fw-line"><span>✓ ${esc(c.label)}</span><b>#${c.n}</b></div>`).join("") || "—")}
      ${card("🔬 Hypothesis", hyp.statement
        ? `<p class="small">${esc(hyp.statement)}</p>
           <span class="tag ${hyp.status === "SUPPORTED" ? "tag-live" : "tag-mixed"}">${esc(hyp.status || "PROPOSED")}</span>
           <span class="small muted"> conf ${pct(hyp.confidence)}</span>`
        : "—")}
      ${card("⚠ Conflict resolution", conflictBody)}
      ${card("✅ Status", line("Outcome", esc(result.status)) +
        line("Deadlock", f.deadlock_detected ? "detected" : "none") +
        `<p class="small muted">${esc(f.termination_reason || "")}</p>`)}
    </div>
    <p class="fw-foot">${(result.insights || []).length} prioritized insight(s) ·
      ${(result.findings || []).length} finding(s) · the classic dashboard tabs above
      also reflect this run.</p>`);
}

function pct(v) {
  if (v == null) return "—";
  return `${Math.round(Number(v) * 100)}%`;
}

function setText(id, t) {
  const el = $(id);
  if (el) el.textContent = t;
}
