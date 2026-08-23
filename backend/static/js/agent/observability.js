/**
 * 🔍 Observability — the trace explorer and the self-improvement loop.
 *
 * Two halves:
 *   1. A trace explorer: every recorded run, its hierarchical span timeline
 *      (run → node → agent → tool → provider), token accounting, and errors.
 *   2. The improvement cycle: inject a controlled failure, diagnose the trace,
 *      apply a bounded policy change, re-run the same scenario, and show the
 *      before/after comparison with the accept or reject verdict.
 *
 * Everything rendered here is read from a real recorded trace. Where a value was
 * not measurable — token usage when the provider reported none, for instance —
 * this shows the recorded reason instead of a number. There are no placeholder
 * metrics anywhere in this view.
 */

import { $, esc, setHTML, show } from "../core/dom.js";
import * as api from "../core/api.js";

let booted = false;
let controller = null;
let state = {
  status: null, traces: [], errors: null, targets: null,
  policy: null, cycles: [], openTrace: null, tree: null,
  diagnosis: null, report: null,
};

// Span kind → accent + glyph. Keeps the timeline readable at a glance.
const KIND_STYLE = {
  run:          ["blue",   "▶"],
  orchestrator: ["blue",   "◆"],
  node:         ["slate",  "▪"],
  agent:        ["purple", "🤖"],
  decision:     ["cyan",   "⚖"],
  llm:          ["pink",   "🧠"],
  tool:         ["orange", "🔧"],
  provider:     ["slate",  "🌐"],
  retry:        ["yellow", "↻"],
  fallback:     ["yellow", "⤵"],
  memory:       ["cyan",   "🧩"],
  evaluation:   ["green",  "📊"],
  verification: ["green",  "✓"],
  synthesis:    ["purple", "✦"],
};

const VERDICT_TONE = {
  IMPROVEMENT_VERIFIED: "green",
  IMPROVEMENT_REJECTED: "red",
  NO_MATERIAL_CHANGE:   "yellow",
  NOT_MEASURABLE:       "slate",
  NO_SAFE_IMPROVEMENT:  "slate",
  NO_DIAGNOSIS:         "slate",
  RERUN_FAILED:         "red",
};

const STAGE_LABELS = [
  ["trace",      "Trace"],
  ["understand", "Understand"],
  ["diagnose",   "Diagnose"],
  ["choose",     "Choose"],
  ["apply",      "Apply"],
  ["rerun",      "Re-run"],
  ["measure",    "Measure"],
  ["verify",     "Verify"],
];

/* ─────────────────────────────────────────────────────────── */
export function initObservability(host) {
  if (booted) { refresh(); return; }
  booted = true;

  setHTML(host, `
    <section class="panel obs-panel">
      <div class="panel-head"><div>
        <span class="eyebrow">Tracing &amp; self-improvement</span>
        <h2>🔍 Observability</h2>
        <p class="panel-sub">End-to-end traces of every agent run — agent, decision,
          prompt, tool, provider, latency, token and error spans. Then the loop that
          matters: inject a controlled failure, diagnose the root cause from the trace,
          apply a bounded runtime change, re-run the same scenario and verify with the
          Task&nbsp;6 evaluators whether the system actually improved.</p>
      </div></div>

      <div class="obs-controls">
        <label class="obs-field">
          <span>Controlled failure target</span>
          <select id="obs-target"></select>
        </label>
        <label class="obs-field obs-narrow">
          <span>Failure type</span>
          <select id="obs-failure"></select>
        </label>
        <label class="obs-field obs-narrow">
          <span>Failures</span>
          <select id="obs-count">
            <option value="1">1</option>
            <option value="2" selected>2</option>
            <option value="3">3</option>
          </select>
        </label>
        <label class="obs-field obs-narrow">
          <span>Target metric</span>
          <select id="obs-metric">
            <option value="duration_ms">Latency (duration_ms)</option>
            <option value="errors">Errors</option>
            <option value="retries">Retries</option>
            <option value="provider_calls">Provider calls</option>
          </select>
        </label>
        <div class="obs-buttons">
          <button type="button" class="btn-primary" id="obs-run">🔬 Run Improvement Cycle</button>
          <button type="button" class="btn-ghost" id="obs-stop" hidden>Stop</button>
          <button type="button" class="btn-mini" id="obs-reset">↺ Reset policy</button>
          <button type="button" class="btn-mini" id="obs-refresh">⟳ Refresh</button>
        </div>
      </div>
      <p class="obs-status" id="obs-status">Loading trace data…</p>

      <div class="obs-stages" id="obs-stages" hidden></div>
      <div id="obs-body"></div>
    </section>`);

  $("obs-run").addEventListener("click", runCycle);
  $("obs-stop").addEventListener("click", () => controller && controller.abort());
  $("obs-reset").addEventListener("click", resetPolicy);
  $("obs-refresh").addEventListener("click", refresh);
  $("obs-body").addEventListener("click", onBodyClick);
  refresh();
}

/* ── data ─────────────────────────────────────────────────── */
async function refresh() {
  setStatus("Loading trace data…");
  const results = await Promise.allSettled([
    api.getObservabilityStatus(),
    api.getTraces(20),
    api.getTraceErrors(10),
    api.getFailureTargets(),
    api.getOptimizationPolicy(),
    api.getImprovementCycles(),
  ]);
  const [status, traces, errors, targets, policy, cycles] = results;

  if (status.status === "fulfilled") state.status = status.value;
  if (traces.status === "fulfilled") state.traces = traces.value.traces || [];
  if (errors.status === "fulfilled") state.errors = errors.value;
  if (targets.status === "fulfilled") state.targets = targets.value;
  if (policy.status === "fulfilled") state.policy = policy.value;
  if (cycles.status === "fulfilled") state.cycles = cycles.value.cycles || [];

  const failed = results.filter((r) => r.status === "rejected");
  if (failed.length === results.length) {
    setStatus(`Could not reach the observability API: ${failed[0].reason?.message || "unknown error"}`);
    setHTML($("obs-body"), emptyState("The backend is unreachable, so no trace data can be shown."));
    return;
  }

  fillSelects();
  setStatus(summaryLine());
  render();
}

function fillSelects() {
  const t = state.targets;
  if (!t) return;
  const targetSel = $("obs-target");
  if (targetSel && !targetSel.options.length) {
    targetSel.innerHTML = (t.targets || []).map((name) =>
      `<option value="${esc(name)}"${name === t.default?.target_source ? " selected" : ""}>${esc(name)}</option>`,
    ).join("");
  }
  const failSel = $("obs-failure");
  if (failSel && !failSel.options.length) {
    failSel.innerHTML = (t.failure_types || []).map((name) =>
      `<option value="${esc(name)}"${name === t.default?.failure_type ? " selected" : ""}>${esc(name.replace(/_/g, " "))}</option>`,
    ).join("");
  }
}

function summaryLine() {
  const s = state.status;
  if (!s) return "Trace data loaded.";
  const stored = s.trace_provider?.traces_stored ?? 0;
  const version = s.policy?.version ?? 0;
  const exportNote = s.external_export?.enabled
    ? "external export on" : "local traces only";
  const armed = (s.armed_failures || []).length;
  return `${stored} trace(s) recorded · runtime policy v${version} · ${exportNote}`
    + (armed ? ` · ${armed} controlled failure(s) armed` : "");
}

function setStatus(text) { const n = $("obs-status"); if (n) n.textContent = text; }

/* ── the improvement cycle ────────────────────────────────── */
async function runCycle() {
  if (controller) return;
  controller = new AbortController();
  show($("obs-stop"), true);
  $("obs-run").disabled = true;
  state.report = null;

  const payload = {
    target_source: $("obs-target").value,
    failure_type: $("obs-failure").value,
    failure_count: Number($("obs-count").value),
    primary_metric: $("obs-metric").value,
    simulation_mode: true,
    validate_with_evaluation: true,
  };

  const live = {};
  renderStages(live);
  show($("obs-stages"), true);
  setStatus(`Running the cycle against ${payload.target_source} (${payload.failure_type})…`);

  try {
    const report = await api.runImprovementStream(payload, (event) => {
      onCycleEvent(event, live);
    }, controller.signal);
    state.report = report;
    // Reload the trace list and policy so both reflect the cycle, then restore the
    // verdict: refresh() sets its own summary line, and the verdict is the thing the
    // user just waited for, so it has to win.
    await refresh();
    setStatus(verdictLine(report));
  } catch (err) {
    if (err.name === "AbortError") setStatus("Cycle stopped.");
    else setStatus(`Cycle failed: ${err.message}`);
  } finally {
    controller = null;
    show($("obs-stop"), false);
    $("obs-run").disabled = false;
  }
}

// SSE event → which stage is now done, and what it reported.
const EVENT_STAGE = {
  cycle_started:         ["trace", "running", "running the baseline with the controlled failure armed"],
  baseline_traced:       ["trace", "done", null],
  trace_analyzed:        ["understand", "done", null],
  root_cause_identified: ["diagnose", "done", null],
  improvement_proposed:  ["choose", "done", null],
  improvement_applied:   ["apply", "done", null],
  rerun_completed:       ["rerun", "done", null],
  metrics_collected:     ["measure", "done", null],
  cycle_completed:       ["verify", "done", null],
};

function onCycleEvent(event, live) {
  const mapped = EVENT_STAGE[event.type];
  if (!mapped) return;
  const [stage, status, note] = mapped;

  // Mark the stage that just reported, and show the next one as running.
  live[stage] = { status, detail: note || describeEvent(event) };
  if (status === "done") {
    const order = STAGE_LABELS.map(([k]) => k);
    const next = order[order.indexOf(stage) + 1];
    if (next && !live[next]) live[next] = { status: "running", detail: "working…" };
  }
  if (event.type === "cycle_completed") {
    live.verify = {
      status: event.verified ? "done" : "rejected",
      detail: `${event.verdict}`,
    };
  }
  renderStages(live);
}

function describeEvent(event) {
  switch (event.type) {
    case "baseline_traced":
      return `${event.spans} spans, ${event.errors} error(s)`;
    case "trace_analyzed":
      return `${event.errors} error(s), ${event.wasted_retries} provider(s) with spent retries`;
    case "root_cause_identified":
      return `${event.root_cause} on ${event.component} (${pct(event.confidence)})`;
    case "improvement_proposed":
      return `${event.improvement_type || "none"} — ${event.parameter || "no parameter"}`;
    case "improvement_applied":
      return `runtime policy v${event.version}`;
    case "rerun_completed":
      return "same scenario re-run";
    case "metrics_collected":
      return event.validated ? "scored by the Task 6 evaluators" : "runtime metrics only";
    default:
      return "done";
  }
}

async function resetPolicy() {
  try {
    const res = await api.resetOptimizationPolicy();
    setStatus(`Runtime policy reset to v${res.version} — the shipped defaults.`);
    await refresh();
  } catch (err) {
    setStatus(`Could not reset the policy: ${err.message}`);
  }
}

/* ── rendering ────────────────────────────────────────────── */
function render() {
  const body = $("obs-body");
  if (!body) return;
  const parts = [];

  if (state.report) parts.push(renderReport(state.report));
  parts.push(renderPolicy());
  parts.push(renderTraceList());
  if (state.tree) parts.push(renderTimeline(state.tree, state.diagnosis));
  parts.push(renderErrors());
  if (state.cycles.length) parts.push(renderCycleHistory());

  setHTML(body, parts.filter(Boolean).join(""));
}

function renderStages(live) {
  const host = $("obs-stages");
  if (!host) return;
  const html = STAGE_LABELS.map(([key, label], i) => {
    const s = live[key];
    const status = s?.status || "pending";
    return `
      <div class="obs-stage is-${esc(status)}">
        <div class="obs-stage-n">${i + 1}</div>
        <div class="obs-stage-b">
          <b>${esc(label)}</b>
          <em>${esc(s?.detail || "waiting")}</em>
        </div>
      </div>`;
  }).join('<div class="obs-stage-arrow">→</div>');
  setHTML(host, html);
}

function renderReport(r) {
  const tone = VERDICT_TONE[r.verdict] || "slate";
  const cmp = r.comparison || {};
  const dx = r.diagnosis || {};
  const plan = r.plan || {};
  const ev = r.evaluation || {};

  const rows = (cmp.rows || []).filter((row) => row.change !== null);
  const changed = rows.filter((row) => row.direction !== "unchanged");
  const shown = changed.length ? changed : rows;

  return `
    <div class="obs-report tone-${esc(tone)}">
      <div class="obs-verdict">
        <span class="obs-verdict-tag">${esc((r.verdict || "").replace(/_/g, " "))}</span>
        <div>
          <b>${r.improvement_verified ? "The change was verified and kept" : "The change was not kept"}</b>
          <em>${esc((r.reasons || [])[0] || "")}</em>
        </div>
      </div>

      <div class="obs-report-grid">
        <div class="obs-card">
          <h4>Root cause</h4>
          <div class="obs-line"><span>Cause</span><b>${esc(dx.root_cause_type || "—")}</b></div>
          <div class="obs-line"><span>Component</span><b>${esc(dx.affected_component || "—")}</b></div>
          <div class="obs-line"><span>Confidence</span><b>${pct(dx.confidence)}</b></div>
          <div class="obs-line"><span>Certain enough to act</span><b class="${dx.uncertain ? "" : "ok"}">${dx.uncertain ? "needs review" : "yes"}</b></div>
          ${(dx.evidence || []).length ? `<ul class="obs-evidence">${
            dx.evidence.map((e) => `<li>${esc(e)}</li>`).join("")
          }</ul>` : ""}
        </div>

        <div class="obs-card">
          <h4>Improvement applied</h4>
          <div class="obs-line"><span>Type</span><b>${esc(plan.improvement_type || "—")}</b></div>
          <div class="obs-line"><span>Parameter</span><b class="mono">${esc(plan.changed_parameter || "—")}</b></div>
          <div class="obs-line"><span>Before → after</span><b>${
            esc(Object.values(plan.current_configuration || {}).join(", ") || "—")
          } → ${esc(Object.values(plan.proposed_configuration || {}).join(", ") || "—")}</b></div>
          <div class="obs-line"><span>Policy version</span><b>v${esc(plan.previous_version ?? 0)} → v${esc(plan.optimization_version ?? 0)}</b></div>
          <div class="obs-line"><span>Status</span><b>${esc(plan.status || "—")}</b></div>
          ${plan.risk ? `<p class="obs-note">Risk: ${esc(plan.risk)}</p>` : ""}
          <p class="obs-note">This changes runtime configuration only — no source file is modified.</p>
        </div>

        <div class="obs-card">
          <h4>Measured impact</h4>
          ${impactLines(dx.impact || {})}
        </div>
      </div>

      <h4 class="obs-h3">Before vs after — same scenario, re-run</h4>
      <div class="obs-table-wrap">
        <table class="obs-table">
          <thead><tr>
            <th>Metric</th><th>Before</th><th>After</th><th>Change</th><th>Direction</th>
          </tr></thead>
          <tbody>${shown.map((row) => `
            <tr class="is-${esc(row.direction)}${row.metric === cmp.primary_metric ? " is-primary" : ""}">
              <td>${esc(row.metric)}${row.metric === cmp.primary_metric ? ' <span class="obs-pill">target</span>' : ""}</td>
              <td>${fmtNum(row.before)}</td>
              <td>${fmtNum(row.after)}</td>
              <td>${row.change > 0 ? "+" : ""}${fmtNum(row.change)}</td>
              <td>${esc(row.direction)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>

      <div class="obs-report-grid">
        <div class="obs-card">
          <h4>Task 6 quality validation</h4>
          <div class="obs-line"><span>Validated by evaluators</span><b>${ev.validated_with_task6 ? "yes" : "no"}</b></div>
          <div class="obs-line"><span>Before outcome</span><b>${esc(ev.before?.outcome || "not measured")}</b></div>
          <div class="obs-line"><span>After outcome</span><b>${esc(ev.after?.outcome || "not measured")}</b></div>
          ${(cmp.quality_regressions || []).length
            ? `<p class="obs-note obs-bad">Quality regressions: ${cmp.quality_regressions.map(esc).join("; ")}</p>`
            : `<p class="obs-note">No quality metric regressed beyond tolerance.</p>`}
          ${ev.before?.error ? `<p class="obs-note obs-bad">${esc(ev.before.error)}</p>` : ""}
        </div>
        <div class="obs-card">
          <h4>Traces compared</h4>
          <div class="obs-line"><span>Before</span><b class="mono">${esc(r.before_trace_id || "—")}</b></div>
          <div class="obs-line"><span>After</span><b class="mono">${esc(r.after_trace_id || "—")}</b></div>
          <div class="obs-line"><span>Scenario</span><b class="mono">${esc(r.scenario || "—")}</b></div>
          <div class="obs-buttons obs-inline">
            ${r.before_trace_id ? `<button type="button" class="btn-mini" data-trace="${esc(r.before_trace_id)}">View before trace</button>` : ""}
            ${r.after_trace_id ? `<button type="button" class="btn-mini" data-trace="${esc(r.after_trace_id)}">View after trace</button>` : ""}
          </div>
        </div>
        <div class="obs-card">
          <h4>Reasoning</h4>
          <ul class="obs-evidence">${
            (r.reasons || []).map((x) => `<li>${esc(x)}</li>`).join("") || "<li>—</li>"
          }</ul>
          ${r.reverted ? '<p class="obs-note obs-bad">The policy change was rolled back automatically.</p>' : ""}
        </div>
      </div>
    </div>`;
}

function impactLines(impact) {
  const out = [];
  const add = (label, value) => out.push(
    `<div class="obs-line"><span>${esc(label)}</span><b>${value}</b></div>`);
  add("Latency added by the fault", `${fmtNum(impact.latency_added_ms)} ms`);
  add("Share of the run", pct(impact.latency_share_of_run));
  add("Retry calls", fmtNum(impact.retry_calls));
  add("Errors", fmtNum(impact.error_count));
  add("Fallbacks", fmtNum(impact.fallback_count));
  // Token and cost overhead are only shown as numbers when the provider actually
  // reported usage; otherwise the recorded reason is shown instead.
  out.push(impact.token_overhead === null || impact.token_overhead === undefined
    ? `<p class="obs-note">Token overhead: not measurable — ${esc(impact.token_overhead_note || "no usage reported")}</p>`
    : `<div class="obs-line"><span>Token overhead</span><b>${fmtNum(impact.token_overhead)}</b></div>`);
  out.push(impact.estimated_cost_change_usd === null || impact.estimated_cost_change_usd === undefined
    ? `<p class="obs-note">Cost change: not measurable — ${esc(impact.estimated_cost_note || "derived from token usage")}</p>`
    : `<div class="obs-line"><span>Cost change</span><b>$${fmtNum(impact.estimated_cost_change_usd)}</b></div>`);
  return out.join("");
}

function renderPolicy() {
  const p = state.policy;
  if (!p) return "";
  const active = p.active || {};
  const retries = Object.entries(active.retry_attempts_by_source || {});
  const timeouts = Object.entries(active.timeout_by_source || {});
  const history = p.history || [];

  return `
    <div class="obs-block">
      <h4 class="obs-h3">Runtime optimization policy — v${esc(p.version ?? 0)}</h4>
      <div class="obs-report-grid">
        <div class="obs-card">
          <h4>Active values</h4>
          ${retries.length
            ? retries.map(([k, v]) => `<div class="obs-line"><span class="mono">retry_attempts[${esc(k)}]</span><b>${esc(v)}</b></div>`).join("")
            : '<p class="obs-note">No retry override in force — every source uses its shipped ceiling.</p>'}
          ${timeouts.map(([k, v]) => `<div class="obs-line"><span class="mono">timeout[${esc(k)}]</span><b>${esc(v)}s</b></div>`).join("")}
          <div class="obs-line"><span>Deduplicate identical tool calls</span><b>${active.dedup_identical_tool_calls ? "on" : "off"}</b></div>
        </div>
        <div class="obs-card">
          <h4>Bounds enforced</h4>
          ${Object.entries(p.bounds || {}).map(([k, v]) =>
            `<div class="obs-line"><span class="mono">${esc(k)}</span><b>${esc(v[0])} – ${esc(v[1])}</b></div>`).join("")}
          <p class="obs-note">${esc(p.note || "")}</p>
        </div>
        <div class="obs-card">
          <h4>Version history</h4>
          ${history.length
            ? history.slice(-6).reverse().map((h) =>
                `<div class="obs-line"><span>v${esc(h.version)}</span><b>${esc(h.reason || "initial defaults")}</b></div>`).join("")
            : '<p class="obs-note">Only the shipped defaults have ever been active.</p>'}
        </div>
      </div>
    </div>`;
}

function renderTraceList() {
  if (!state.traces.length) {
    return emptyState(
      "No traces recorded yet. Run the improvement cycle above, or start any agent "
      + "run from Overview or Framework — every run is traced automatically.");
  }
  return `
    <div class="obs-block">
      <h4 class="obs-h3">Recorded traces</h4>
      <div class="obs-table-wrap">
        <table class="obs-table">
          <thead><tr>
            <th>Trace</th><th>Scenario</th><th>Status</th><th>Spans</th>
            <th>Errors</th><th>Duration</th><th>Policy</th><th>Tokens</th><th></th>
          </tr></thead>
          <tbody>${state.traces.map((t) => `
            <tr${state.openTrace === t.trace_id ? ' class="is-open"' : ""}>
              <td class="mono">${esc(t.trace_id)}</td>
              <td>${esc(t.scenario || "normal")}</td>
              <td><span class="obs-tag is-${esc(t.status || "ok")}">${esc(t.status || "ok")}</span></td>
              <td>${fmtNum(t.span_count)}</td>
              <td>${fmtNum(t.error_count)}</td>
              <td>${fmtNum(t.duration_ms)} ms</td>
              <td>v${esc(t.optimization_version ?? 0)}</td>
              <td>${esc(tokenLabel(t))}${t.partial ? ' <span class="obs-pill">summary</span>' : ""}</td>
              <td><button type="button" class="btn-mini" data-trace="${esc(t.trace_id)}">Inspect</button></td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      ${state.status?.trace_provider?.degraded
        ? `<p class="obs-note obs-bad">Trace store: ${esc(state.status.trace_provider.degraded)}</p>` : ""}
    </div>`;
}

function tokenLabel(t) {
  const tokens = t.token_usage || {};
  if (tokens.status === "measured") return `${tokens.total_tokens ?? 0}`;
  return "unavailable";
}

function renderTimeline(tree, diagnosis) {
  const total = Math.max(1, tree.duration_ms || 1);
  const rows = [];

  const walk = (nodes) => {
    for (const span of nodes) {
      const [accent, glyph] = KIND_STYLE[span.kind] || ["slate", "▪"];
      const width = Math.max(0.6, (span.duration_ms / total) * 100);
      // Offset each bar by where it started, so the timeline shows concurrency
      // (two agents running in parallel appear side by side, not stacked).
      const attrs = span.attributes || {};
      const detail = [
        attrs.provider && `provider ${attrs.provider}`,
        attrs.tool && `tool ${attrs.tool}`,
        attrs.prompt_type && `prompt ${attrs.prompt_type}`,
        Number.isFinite(attrs.result_count) && `${attrs.result_count} result(s)`,
        Number.isFinite(attrs.attempts) && attrs.attempts > 0 && `${attrs.attempts} attempt(s)`,
        Number.isFinite(attrs.retry_wait_ms) && attrs.retry_wait_ms > 0 && `${attrs.retry_wait_ms}ms backoff`,
        attrs.decision && `→ ${attrs.decision}`,
        attrs.providers_failed && `lost ${attrs.providers_failed}`,
      ].filter(Boolean).join(" · ");

      rows.push(`
        <div class="obs-span is-${esc(span.status)}" style="--indent:${span.depth * 14}px">
          <div class="obs-span-label">
            <span class="obs-span-glyph accent-${esc(accent)}">${glyph}</span>
            <b>${esc(span.name)}</b>
            <span class="obs-span-kind">${esc(span.kind)}</span>
            ${span.agent ? `<span class="obs-span-agent">${esc(span.agent)}</span>` : ""}
          </div>
          <div class="obs-span-track">
            <div class="obs-span-bar accent-${esc(accent)}" style="width:${width.toFixed(2)}%"></div>
            <span class="obs-span-ms">${fmtNum(span.duration_ms)} ms</span>
          </div>
          ${detail ? `<div class="obs-span-detail">${esc(detail)}</div>` : ""}
          ${(span.events || []).length ? `<div class="obs-span-events">${
            span.events.map((e) => `<span class="obs-event">${esc(e.name)}</span>`).join("")
          }</div>` : ""}
        </div>`);
      if (span.children?.length) walk(span.children);
    }
  };
  walk(tree.tree || []);

  const tokens = tree.token_usage || {};
  // A restored trace has counts but no span detail. Say so plainly instead of
  // rendering an empty timeline that looks like a bug.
  if (tree.partial && !rows.length) {
    return `
      <div class="obs-block" id="obs-timeline">
        <h4 class="obs-h3">Span timeline — <span class="mono">${esc(tree.trace_id)}</span></h4>
        <div class="obs-meta">
          <span><b>${fmtNum(tree.recorded_span_count)}</b> spans were recorded</span>
          <span><b>${fmtNum(tree.duration_ms)}</b> ms total</span>
          <span>scenario <b>${esc(tree.scenario || "normal")}</b></span>
        </div>
        ${emptyState(tree.partial_reason
          || "Span detail is not retained for this trace. Run the cycle above to produce a full trace.")}
      </div>`;
  }
  return `
    <div class="obs-block" id="obs-timeline">
      <h4 class="obs-h3">Span timeline — <span class="mono">${esc(tree.trace_id)}</span></h4>
      <div class="obs-meta">
        <span><b>${fmtNum(tree.span_count)}</b> spans</span>
        <span><b>${fmtNum(tree.duration_ms)}</b> ms total</span>
        <span>scenario <b>${esc(tree.scenario || "normal")}</b></span>
        <span>policy <b>v${esc(tree.optimization_version ?? 0)}</b></span>
        <span class="${tree.orphan_count ? "obs-bad" : ""}">
          parent/child integrity: <b>${tree.orphan_count ? `${tree.orphan_count} orphan span(s)` : "intact"}</b>
        </span>
        <span>tokens: <b>${tokens.status === "measured"
          ? `${tokens.total_tokens} (${esc(tokens.model || "model")})`
          : "unavailable"}</b></span>
      </div>
      ${tokens.status !== "measured" && tokens.reason
        ? `<p class="obs-note">Token usage was not recorded for this run: ${esc(tokens.reason)}</p>` : ""}
      ${diagnosis ? renderInlineDiagnosis(diagnosis) : ""}
      <div class="obs-timeline">${rows.join("")}</div>
    </div>`;
}

function renderInlineDiagnosis(d) {
  const dx = d.diagnosis || {};
  if (!dx.root_cause_type) return "";
  const tone = dx.root_cause_type === "UNKNOWN" ? "slate"
    : dx.uncertain ? "yellow" : "red";
  return `
    <div class="obs-diagnosis tone-${esc(tone)}">
      <b>Diagnosis: ${esc(dx.root_cause_type)}${dx.affected_component ? ` on ${esc(dx.affected_component)}` : ""}</b>
      <span class="obs-conf">confidence ${pct(dx.confidence)}${dx.uncertain ? " · needs review" : ""}</span>
      <ul class="obs-evidence">${(dx.evidence || []).map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
      ${dx.recommended_improvement ? `<p class="obs-note">${esc(dx.recommended_improvement)}</p>` : ""}
      ${(dx.alternatives || []).length
        ? `<p class="obs-note">Other explanations considered: ${dx.alternatives.map(esc).join(", ")}</p>` : ""}
    </div>`;
}

function renderErrors() {
  const e = state.errors;
  if (!e || !e.count) return "";
  return `
    <div class="obs-block">
      <h4 class="obs-h3">Errors across recent traces</h4>
      <div class="obs-meta">
        <span><b>${fmtNum(e.count)}</b> total</span>
        <span><b>${fmtNum(e.recovered)}</b> recovered</span>
        <span><b>${fmtNum(e.injected)}</b> deliberately injected</span>
      </div>
      <div class="obs-chips">
        ${Object.entries(e.by_category || {}).map(([k, v]) =>
          `<span class="obs-chip">${esc(k)} <b>${v}</b></span>`).join("")}
      </div>
      <div class="obs-table-wrap">
        <table class="obs-table">
          <thead><tr>
            <th>Component</th><th>Type</th><th>HTTP</th><th>Retryable</th>
            <th>Recovery</th><th>Injected</th><th>Message</th>
          </tr></thead>
          <tbody>${(e.errors || []).slice(0, 24).map((err) => `
            <tr>
              <td>${esc(err.provider || err.component || "—")}</td>
              <td>${esc(err.error_type || "—")}</td>
              <td>${esc(err.http_status ?? "—")}</td>
              <td>${err.retryable ? "yes" : "no"}</td>
              <td>${esc(err.recovery_status || "—")}</td>
              <td>${err.injected ? "yes" : "no"}</td>
              <td class="obs-msg">${esc(err.safe_message || "")}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <p class="obs-note">Messages are recorded through a redaction filter, so no
        credential, prompt text or internal reasoning can appear here.</p>
    </div>`;
}

function renderCycleHistory() {
  return `
    <div class="obs-block">
      <h4 class="obs-h3">Previous improvement cycles</h4>
      <div class="obs-table-wrap">
        <table class="obs-table">
          <thead><tr>
            <th>Cycle</th><th>Root cause</th><th>Confidence</th><th>Parameter</th>
            <th>Verdict</th><th>Kept</th>
          </tr></thead>
          <tbody>${state.cycles.map((c) => `
            <tr>
              <td class="mono">${esc(c.cycle_id)}</td>
              <td>${esc(c.root_cause || "—")}</td>
              <td>${pct(c.confidence)}</td>
              <td class="mono">${esc(c.changed_parameter || "—")}</td>
              <td><span class="obs-tag is-${esc(VERDICT_TONE[c.verdict] || "slate")}">${esc((c.verdict || "").replace(/_/g, " "))}</span></td>
              <td>${c.improvement_verified ? "kept" : (c.reverted ? "reverted" : "no")}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>`;
}

/* ── interaction ──────────────────────────────────────────── */
async function onBodyClick(e) {
  const btn = e.target.closest("[data-trace]");
  if (!btn) return;
  const traceId = btn.dataset.trace;
  setStatus(`Loading trace ${traceId}…`);
  try {
    const [tree, diagnosis] = await Promise.all([
      api.getTraceTree(traceId),
      api.getTraceRootCause(traceId).catch(() => null),
    ]);
    state.openTrace = traceId;
    state.tree = tree;
    state.diagnosis = diagnosis;
    render();
    setStatus(summaryLine());
    $("obs-timeline")?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setStatus(`Could not load that trace: ${err.message}`);
  }
}

/* ── helpers ──────────────────────────────────────────────── */
function verdictLine(r) {
  const verified = r.improvement_verified ? "verified and kept" : "not kept";
  return `${(r.verdict || "").replace(/_/g, " ")} — the change was ${verified}. `
    + `${(r.reasons || [])[0] || ""} · ${summaryLine()}`;
}

function pct(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function fmtNum(v) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return esc(String(v));
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(Math.abs(n) < 1 ? 4 : 2);
}

function emptyState(message) {
  return `<div class="obs-empty"><p>${esc(message)}</p></div>`;
}
