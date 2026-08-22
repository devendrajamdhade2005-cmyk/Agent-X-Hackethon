/* 📊 Evaluation dashboard (Task 6).
 *
 * Every number rendered here comes from a stored evaluation result produced by a
 * real agent execution. When a metric was not measurable, this shows the recorded
 * reason instead of a number — and when no suite has run yet it shows an explicit
 * empty state rather than placeholder scores.
 */

import { $, esc, setHTML, show } from "../core/dom.js";
import * as api from "../core/api.js";

let booted = false;
let controller = null;
let state = { metrics: null, runs: [], baseline: {}, history: [], human: null, cases: null };

const SCENARIOS = [
  "NORMAL", "AMBIGUOUS", "ADVERSARIAL", "CONTRADICTORY",
  "INCOMPLETE", "TOOL_FAILURE", "UNSUPPORTED_CONCLUSION",
];

const CARD_METRICS = [
  ["accuracy", "Accuracy", "blue"],
  ["task_completion", "Task completion", "green"],
  ["groundedness", "Groundedness", "cyan"],
  ["hallucination_rate", "Hallucination", "red"],
  ["recovery_rate", "Recovery", "green"],
  ["robustness", "Robustness", "purple"],
  ["evidence_quality", "Evidence quality", "orange"],
  ["uncertainty_handling", "Uncertainty handling", "yellow"],
  ["consistency", "Consistency", "cyan"],
  ["reliability", "Reliability", "green"],
  ["latency", "Median latency", "slate"],
  ["resource_efficiency", "Resource efficiency", "slate"],
];

const HUMAN_DIMS = [
  ["accuracy_score", "Accuracy"],
  ["completion_score", "Task completion"],
  ["evidence_score", "Evidence quality"],
  ["groundedness_score", "Groundedness"],
  ["uncertainty_score", "Uncertainty handling"],
  ["actionability_score", "Actionability"],
  ["overall_score", "Overall quality"],
];

/* ─────────────────────────────────────────────────────────── */
export function initEvaluation(host) {
  if (booted) { refresh(); return; }
  booted = true;

  setHTML(host, `
    <section class="panel ev-panel">
      <div class="panel-head"><div>
        <span class="eyebrow">Quality measurement</span>
        <h2>📊 Evaluation</h2>
        <p class="panel-sub">Automated and human evaluation of the real InsightPulse agent
          across normal, ambiguous, adversarial, contradictory, incomplete, tool-failure
          and unsupported-conclusion scenarios — with repeated runs and baseline comparison.</p>
      </div></div>

      <div class="ev-controls">
        <label class="ev-field">
          <span>Suite</span>
          <select id="ev-mode">
            <option value="demo">Demo suite (representative)</option>
            <option value="full">Full suite (all cases)</option>
            <option value="adversarial">Adversarial suite</option>
          </select>
        </label>
        <label class="ev-field ev-narrow">
          <span>Baseline</span>
          <select id="ev-baseline">
            <option value="1">Include baseline</option>
            <option value="0">Skip baseline</option>
          </select>
        </label>
        <div class="ev-buttons">
          <button type="button" class="btn-primary" id="ev-run">🧪 Run Evaluation Suite</button>
          <button type="button" class="btn-ghost" id="ev-stop" hidden>Stop</button>
          <a class="btn-mini" id="ev-export-md" href="#" target="_blank" rel="noopener">↓ Report (MD)</a>
          <a class="btn-mini" id="ev-export-html" href="#" target="_blank" rel="noopener">↓ Report (HTML)</a>
        </div>
      </div>
      <p class="ev-status" id="ev-status">Loading evaluation data…</p>
      <div class="ev-progress" id="ev-progress" hidden></div>

      <div id="ev-body"></div>
    </section>`);

  $("ev-run").addEventListener("click", runSuite);
  $("ev-stop").addEventListener("click", () => controller && controller.abort());
  $("ev-body").addEventListener("click", onBodyClick);
  refresh();
}

/* ─────────────────────────────────────────────────────────── */
async function refresh() {
  try {
    const [metrics, runs, baseline, history, humanQ, cases] = await Promise.all([
      api.getEvaluationMetrics(), api.getEvaluationRuns(), api.getEvaluationBaseline(),
      api.getEvaluationHistory(), api.getEvaluationHuman(), api.getEvaluationCases(),
    ]);
    state = { metrics, runs: runs.runs || [], baseline: baseline.comparison || {},
              history: history.history || [], regression: history.regression || {},
              human: humanQ, cases };
    render();
  } catch (err) {
    setText("ev-status", `Could not load evaluation data: ${err.message}`);
  }
}

function render() {
  const m = state.metrics || {};
  const suiteId = m.latest_suite_id;
  $("ev-export-md").href = api.evaluationReportUrl("md", suiteId || "");
  $("ev-export-html").href = api.evaluationReportUrl("html", suiteId || "");

  if (!m.has_data) {
    setText("ev-status",
      `${(state.cases?.cases || []).length} benchmark cases loaded across ${SCENARIOS.length} scenario types.`);
    setHTML($("ev-body"), `
      <div class="ev-empty">
        <span class="ev-empty-i">📊</span>
        <h3>No evaluation data yet</h3>
        <p>Run the evaluation suite to measure accuracy, groundedness, hallucination,
           recovery, consistency, latency and resource efficiency against the real agent.</p>
      </div>
      ${casesTable()}
      ${methodology()}`);
    return;
  }

  setText("ev-status",
    `Suite ${suiteId} · ${(m.counts?.runs) || 0} run(s) · ` +
    `${m.counts?.pass || 0} pass / ${m.counts?.partial || 0} partial / ${m.counts?.fail || 0} fail`);

  setHTML($("ev-body"), `
    ${cards()}
    ${scenarioMatrix()}
    ${caseTable()}
    ${baselineView()}
    ${humanView()}
    ${historyView()}
    ${methodology()}`);
}

/* ── metric cards (section 39) ───────────────────────────── */
function cards() {
  const latest = state.metrics?.latest || {};
  const reg = (state.regression?.changes || []).reduce((acc, c) => {
    acc[c.metric] = c; return acc;
  }, {});
  const baselineRows = firstBaselineRows();

  const items = CARD_METRICS.map(([key, label, accent]) => {
    const entry = latest[key] || aggregateFallback(key);
    if (!entry) return "";
    if (!entry.available) {
      return `<article class="ev-card is-na">
        <span class="ev-card-l">${esc(label)}</span>
        <b class="ev-card-v">n/a</b>
        <span class="ev-card-s">${esc(entry.unavailable_reason || "not measurable")}</span>
      </article>`;
    }
    const isMs = entry.unit === "ms";
    const value = isMs ? `${Math.round(entry.value)}ms` : `${(entry.value * 100).toFixed(1)}%`;
    const lower = entry.higher_is_better === false;

    // vs baseline, when the same metric was comparably measured
    const b = baselineRows[key];
    let cmp = "";
    if (b && b.available && typeof b.difference === "number" && b.direction !== "equal") {
      const good = b.direction === "better";
      const delta = isMs ? `${Math.abs(Math.round(b.difference))}ms` : `${Math.abs(b.difference * 100).toFixed(1)}%`;
      cmp = `<span class="ev-delta ${good ? "up" : "down"}">${good ? "↑" : "↓"} ${delta} vs baseline</span>`;
    }
    const r = reg[key];
    let trend = "";
    if (r && r.direction === "improved") trend = `<span class="ev-delta up">improved</span>`;
    else if (r && r.direction === "regressed") trend = `<span class="ev-delta down">regressed</span>`;

    return `<article class="ev-card accent-${accent}${lower ? " is-lower" : ""}">
      <span class="ev-card-l">${esc(label)}${lower ? " <em>(lower is better)</em>" : ""}</span>
      <b class="ev-card-v">${value}</b>
      <span class="ev-card-s">${cmp}${trend}</span>
    </article>`;
  }).join("");

  const overall = state.metrics?.latest?.overall_score;
  return `
    <div class="ev-overall">
      <span>Overall evaluation score</span>
      <b>${typeof overall === "number" ? `${(overall * 100).toFixed(1)}%` : "n/a"}</b>
    </div>
    <div class="ev-cards">${items}</div>`;
}

/** Reliability/consistency live on the suite, not the per-metric aggregate. */
function aggregateFallback(key) {
  if (key !== "consistency" && key !== "reliability") return null;
  const runs = state.runs || [];
  void runs;
  // Pull the first measured value from the suite-level blocks, if present.
  const block = key === "consistency" ? state.metrics?.consistency : state.metrics?.reliability;
  if (!block) {
    return { available: false, unavailable_reason: "requires a repeated-run case" };
  }
  const first = Object.values(block).find((v) => v && typeof v === "object" && v.available);
  return first || { available: false, unavailable_reason: "requires a repeated-run case" };
}

function firstBaselineRows() {
  const systems = Object.values(state.baseline || {});
  const preferred = systems.find((s) => !s.blocked) || systems[0];
  const out = {};
  for (const row of (preferred?.rows || [])) out[row.metric] = row;
  return out;
}

/* ── scenario matrix (section 37) ─────────────────────────── */
function scenarioMatrix() {
  const matrix = state.metrics?.scenario_matrix || {};
  const rows = SCENARIOS.map((s) => {
    const b = matrix[s] || { total: 0, passed: 0, partial: 0, failed: 0, score: 0 };
    const cls = !b.total ? "is-na" : b.failed ? "is-bad" : b.partial ? "is-mid" : "is-good";
    return `<tr class="${cls}">
      <td>${esc(s.replace(/_/g, " "))}</td>
      <td>${b.total || "—"}</td><td>${b.passed || 0}</td>
      <td>${b.partial || 0}</td><td>${b.failed || 0}</td>
      <td>${b.total ? `${(b.score * 100).toFixed(0)}%` : "not run"}</td>
    </tr>`;
  }).join("");
  return `<section class="ev-block"><h3>Scenario coverage</h3>
    <table class="ev-table"><thead><tr>
      <th>Scenario</th><th>Cases</th><th>Pass</th><th>Partial</th><th>Fail</th><th>Score</th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
}

/* ── case table (section 40) ──────────────────────────────── */
function caseTable() {
  const runs = (state.runs || []).filter((r) => r.system === "insightpulse");
  if (!runs.length) return "";
  const rows = runs.map((r) => {
    const m = r.metrics || {};
    return `<tr>
      <td class="mono">${esc(r.case_id)}${r.repeat_index ? ` <em>#${r.repeat_index + 1}</em>` : ""}</td>
      <td>${esc((r.scenario_type || "").replace(/_/g, " "))}</td>
      <td><span class="ev-pill is-${String(r.outcome).toLowerCase()}">${esc(r.outcome)}</span></td>
      <td>${cell(m.accuracy)}</td>
      <td>${cell(m.groundedness)}</td>
      <td>${cell(m.hallucination_rate)}</td>
      <td>${cell(m.recovery_rate)}</td>
      <td>${cell(m.latency)}</td>
      <td><button type="button" class="btn-mini" data-ev-review="${esc(r.evaluation_run_id)}">
        ${r.reviewer_count ? `👤 ${r.reviewer_count}` : "Review"}</button></td>
    </tr>`;
  }).join("");
  return `<section class="ev-block"><h3>Case results</h3>
    <table class="ev-table"><thead><tr>
      <th>Case</th><th>Scenario</th><th>Status</th><th>Accuracy</th><th>Grounded</th>
      <th>Halluc.</th><th>Recovery</th><th>Latency</th><th>Human</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="muted small">“n/a” means the metric was not applicable to that case — the
      recorded reason is shown in the exported report.</p></section>`;
}

function cell(entry) {
  if (!entry) return `<span class="muted">—</span>`;
  if (!entry.available) {
    return `<span class="muted" title="${esc(entry.unavailable_reason || "")}">n/a</span>`;
  }
  return entry.unit === "ms"
    ? `${Math.round(entry.value)}ms`
    : `${(entry.value * 100).toFixed(0)}%`;
}

/* ── baseline comparison (section 41) ─────────────────────── */
function baselineView() {
  const systems = Object.entries(state.baseline || {});
  if (!systems.length) return "";
  const blocks = systems.map(([name, comp]) => {
    const rows = (comp.rows || []).map((row) => {
      if (!row.available) {
        return `<tr class="is-na"><td>${esc(row.label)}</td><td colspan="4" class="muted">
          not comparable — ${esc(row.unavailable_reason || "")}</td></tr>`;
      }
      const isMs = row.unit === "ms";
      const fmt = (v) => (isMs ? `${Math.round(v)}ms` : `${(v * 100).toFixed(1)}%`);
      const dirCls = row.direction === "better" ? "up" : row.direction === "worse" ? "down" : "flat";
      return `<tr>
        <td>${esc(row.label)}${row.higher_is_better === false ? " <em>(lower better)</em>" : ""}</td>
        <td>${fmt(row.baseline)}</td>
        <td><b>${fmt(row.insightpulse)}</b></td>
        <td>${isMs ? `${Math.round(row.difference)}ms` : `${(row.difference * 100).toFixed(1)}%`}</td>
        <td><span class="ev-delta ${dirCls}">${esc(row.direction || "")}</span></td>
      </tr>`;
    }).join("");
    return `<div class="ev-baseline">
      <h4>${esc(name.replace(/_/g, " "))}</h4>
      ${comp.blocked ? `<p class="ev-warn">⚠ This baseline could not produce output:
        ${esc(comp.blocked_reason || "")}. ${esc(comp.blocked_note || "")}</p>` : ""}
      <p class="muted small">Cases compared: ${esc((comp.cases_compared || []).join(", ") || "none")}.
        ${(comp.excluded_cases || []).length
          ? `Excluded: ${esc(comp.excluded_cases.join(", "))} — ${esc(comp.exclusion_reason || "")}`
          : ""}</p>
      <table class="ev-table"><thead><tr>
        <th>Metric</th><th>Baseline</th><th>InsightPulse</th><th>Difference</th><th></th>
      </tr></thead><tbody>${rows}</tbody></table>
    </div>`;
  }).join("");
  return `<section class="ev-block"><h3>📈 Baseline vs InsightPulse</h3>${blocks}</section>`;
}

/* ── human review (section 42) ────────────────────────────── */
function humanView() {
  const h = state.human || {};
  const pending = h.pending || [];
  const completed = h.completed || [];
  return `<section class="ev-block"><h3>👤 Human evaluation</h3>
    <p class="muted small">${completed.length} reviewed · ${pending.length} awaiting review ·
      ${h.review_count || 0} total review(s) submitted.</p>
    <div id="ev-review-host"></div>
    ${completed.length ? `<table class="ev-table"><thead><tr>
        <th>Case</th><th>Scenario</th><th>Automated</th><th>Reviewers</th><th></th>
      </tr></thead><tbody>${completed.map((c) => `<tr>
        <td class="mono">${esc(c.case_id)}</td>
        <td>${esc((c.scenario_type || "").replace(/_/g, " "))}</td>
        <td><span class="ev-pill is-${String(c.outcome).toLowerCase()}">${esc(c.outcome)}</span></td>
        <td>${c.reviewer_count}</td>
        <td><button type="button" class="btn-mini" data-ev-review="${esc(c.evaluation_run_id)}">Open</button></td>
      </tr>`).join("")}</tbody></table>` : ""}
    <p class="muted small">Use the “Review” button in the case table to score a run.</p>
  </section>`;
}

/* ── history + regression (sections 43-44) ────────────────── */
function historyView() {
  const hist = state.history || [];
  const reg = state.regression || {};
  const rows = hist.slice(0, 8).map((h, i) => {
    const overall = (h.aggregate || {}).overall_score;
    return `<tr>
      <td class="mono">${esc(h.suite_id)}</td>
      <td>${esc((h.completed_at || h.started_at || "").slice(0, 16).replace("T", " "))}</td>
      <td>${esc(h.mode || "")}</td>
      <td>${typeof overall === "number" ? `${(overall * 100).toFixed(1)}%` : "n/a"}</td>
      <td>${i === 0 && typeof reg.overall_delta === "number"
            ? `<span class="ev-delta ${reg.overall_delta >= 0 ? "up" : "down"}">
                 ${reg.overall_delta >= 0 ? "+" : ""}${(reg.overall_delta * 100).toFixed(1)}%</span>`
            : ""}</td>
    </tr>`;
  }).join("");

  const regBlock = reg.compared
    ? `<div class="ev-reg">
         <h4>Regression vs ${esc(reg.previous_suite_id || "previous suite")}</h4>
         ${(reg.regressions || []).length
            ? `<ul class="ev-reg-list is-bad">${reg.regressions.map((r) =>
                `<li>▼ ${esc(r.label || r.metric)}: ${r.previous} → ${r.current}</li>`).join("")}</ul>`
            : `<p class="muted small">No regressions detected.</p>`}
         ${(reg.improvements || []).length
            ? `<ul class="ev-reg-list is-good">${reg.improvements.map((r) =>
                `<li>▲ ${esc(r.label || r.metric)}: ${r.previous} → ${r.current}</li>`).join("")}</ul>`
            : ""}
       </div>`
    : `<p class="muted small">${esc(reg.reason || "No previous suite to compare against.")}</p>`;

  return `<section class="ev-block"><h3>Evaluation history</h3>
    ${hist.length ? `<table class="ev-table"><thead><tr>
      <th>Suite</th><th>When</th><th>Mode</th><th>Overall</th><th>Change</th>
    </tr></thead><tbody>${rows}</tbody></table>` : `<p class="muted small">No history yet.</p>`}
    ${regBlock}</section>`;
}

/* ── benchmark cases + methodology (section 46) ───────────── */
function casesTable() {
  const cases = state.cases?.cases || [];
  if (!cases.length) return "";
  return `<section class="ev-block"><h3>Benchmark dataset</h3>
    <table class="ev-table"><thead><tr>
      <th>Case</th><th>Scenario</th><th>Goal</th><th>Difficulty</th><th>Repeats</th>
    </tr></thead><tbody>${cases.map((c) => `<tr>
      <td class="mono">${esc(c.case_id)}</td>
      <td>${esc((c.scenario_type || "").replace(/_/g, " "))}</td>
      <td>${esc(c.user_goal)}</td>
      <td>${esc(c.difficulty)}</td>
      <td>${c.repeat_count}</td>
    </tr>`).join("")}</tbody></table></section>`;
}

function methodology() {
  const specs = state.metrics?.methodology || [];
  if (!specs.length) return "";
  return `<details class="ev-block ev-method"><summary><h3>Metric methodology</h3></summary>
    <table class="ev-table"><thead><tr>
      <th>Metric</th><th>Definition</th><th>Formula</th><th>Data source</th>
    </tr></thead><tbody>${specs.map((s) => `<tr>
      <td><b>${esc(s.label)}</b><br><span class="muted small">${esc(s.unit)} · ${esc(s.scope)} ·
        ${s.higher_is_better ? "higher better" : "lower better"}</span></td>
      <td class="small">${esc(s.definition)}</td>
      <td class="small mono">${esc(s.formula)}</td>
      <td class="small muted">${esc(s.data_source)}</td>
    </tr>`).join("")}</tbody></table></details>`;
}

/* ─────────────────────────────────────────────────────────── */
async function runSuite() {
  const mode = $("ev-mode").value;
  const includeBaseline = $("ev-baseline").value === "1";
  $("ev-run").disabled = true;
  show($("ev-stop"), true);
  show($("ev-progress"), true);
  setHTML($("ev-progress"), "");
  setText("ev-status", `Running the ${mode} evaluation suite against the real agent…`);

  controller = new AbortController();
  try {
    await api.runEvaluationStream(
      { mode, include_baseline: includeBaseline, simulation_mode: true },
      onProgress, controller.signal,
    );
    setText("ev-status", "Evaluation complete. Loading results…");
    await refresh();
  } catch (err) {
    setText("ev-status", err.name === "AbortError" ? "Stopped." : `Evaluation failed: ${err.message}`);
  } finally {
    controller = null;
    $("ev-run").disabled = false;
    show($("ev-stop"), false);
  }
}

function onProgress(event) {
  if (event.type !== "evaluation") return;
  const box = $("ev-progress");
  const e = event.event;
  let line = "";
  if (e === "suite_started") line = `▶ Suite started — ${event.total_cases} case(s), mode ${event.mode}`;
  else if (e === "case_started") line = `• ${event.case_id} ${event.name} (${event.scenario}) ×${event.repeats}`;
  else if (e === "case_result") {
    const g = typeof event.groundedness === "number" ? ` grounded ${(event.groundedness * 100).toFixed(0)}%` : "";
    line = `   ${event.outcome === "PASS" ? "✓" : event.outcome === "FAIL" ? "✕" : "◐"} ${event.case_id} → ${event.outcome}${g}`;
  } else if (e === "baseline_started") line = `▶ Baseline ${event.system} — ${event.cases} case(s)`;
  else if (e === "baseline_result") line = `   · ${event.system} ${event.case_id} → ${event.outcome}`;
  else if (e === "suite_completed") {
    line = `✅ Suite complete — overall ${typeof event.overall === "number" ? `${(event.overall * 100).toFixed(1)}%` : "n/a"}`;
  }
  if (!line) return;
  const row = document.createElement("div");
  row.className = "ev-prog-row" + (line.includes("✕") ? " is-bad" : "");
  row.textContent = line;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

/* ── human review form ───────────────────────────────────── */
async function onBodyClick(e) {
  const btn = e.target.closest("[data-ev-review]");
  if (!btn) return;
  const runId = btn.dataset.evReview;
  try {
    const data = await api.getEvaluationRun(runId);
    openReviewForm(runId, data);
  } catch (err) {
    setText("ev-status", `Could not open the run: ${err.message}`);
  }
}

function openReviewForm(runId, data) {
  const host = $("ev-review-host");
  if (!host) return;
  const run = data.run || {};
  const hv = data.human_vs_automated || {};
  const rows = HUMAN_DIMS.map(([key, label]) => `
    <label class="ev-score">
      <span>${esc(label)}</span>
      <select data-score="${key}">
        ${[1, 2, 3, 4, 5].map((n) => `<option value="${n}" ${n === 3 ? "selected" : ""}>${n}</option>`).join("")}
      </select>
    </label>`).join("");

  const compare = hv.available
    ? `<div class="ev-compare">
         <h5>Automated vs human${hv.disagreement_detected ? " — <span class='ev-warn-i'>disagreement detected</span>" : ""}</h5>
         <table class="ev-table"><thead><tr><th>Dimension</th><th>Automated</th><th>Human</th><th>Gap</th></tr></thead>
         <tbody>${(hv.rows || []).map((r) => `<tr class="${r.disagreement ? "is-mid" : ""}">
           <td>${esc(r.metric)}</td>
           <td>${typeof r.automated === "number" ? `${(r.automated * 100).toFixed(0)}%` : `<span class="muted">n/a</span>`}</td>
           <td>${typeof r.human_normalised === "number" ? `${(r.human_normalised * 100).toFixed(0)}%` : "—"}</td>
           <td>${typeof r.gap === "number" ? `${(r.gap * 100).toFixed(0)}%` : "—"}</td>
         </tr>`).join("")}</tbody></table>
       </div>`
    : "";

  setHTML(host, `
    <div class="ev-review">
      <h4>Review ${esc(run.case_id || runId)} — ${esc(run.case_name || "")}</h4>
      <p class="muted small">Automated outcome:
        <span class="ev-pill is-${String(run.outcome).toLowerCase()}">${esc(run.outcome || "")}</span>
        · ${esc((run.outcome_reasons || []).join("; "))}</p>
      <div class="ev-scores">${rows}</div>
      <label class="ev-field">
        <span>Decision</span>
        <select id="ev-decision">
          <option value="PASS">PASS</option>
          <option value="PARTIAL" selected>PARTIAL</option>
          <option value="FAIL">FAIL</option>
        </select>
      </label>
      <label class="ev-field">
        <span>Comment (optional)</span>
        <input id="ev-comment" type="text" placeholder="What stood out?" />
      </label>
      <div class="ev-buttons">
        <button type="button" class="btn-primary compact" id="ev-submit">Submit review</button>
        <button type="button" class="btn-ghost compact" id="ev-cancel">Cancel</button>
      </div>
      <p class="ev-review-status" id="ev-review-status"></p>
      ${compare}
    </div>`);

  host.scrollIntoView({ behavior: "smooth", block: "center" });
  $("ev-cancel").addEventListener("click", () => setHTML(host, ""));
  $("ev-submit").addEventListener("click", async () => {
    const payload = {
      evaluation_run_id: runId,
      reviewer_id: `reviewer-${Date.now().toString().slice(-4)}`,
      decision: $("ev-decision").value,
      comment: $("ev-comment").value.slice(0, 1200),
    };
    host.querySelectorAll("[data-score]").forEach((sel) => {
      payload[sel.dataset.score] = Number(sel.value);
    });
    try {
      const res = await api.submitHumanReview(payload);
      const agg = res.aggregate || {};
      // Refresh and re-open first, then write the confirmation — otherwise the
      // re-render replaces the form and the message disappears immediately.
      await refresh();
      openReviewForm(runId, { run, human_vs_automated: res.human_vs_automated });
      setText("ev-review-status",
        `Saved. ${agg.reviewer_count} reviewer(s), average overall ${agg.average_overall}` +
        (agg.score_variance ? `, variance ${agg.score_variance}` : ""));
    } catch (err) {
      setText("ev-review-status", `Could not save: ${err.message}`);
    }
  });
}

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}
