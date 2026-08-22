/* ═══════════════════════════════════════════════════════════
   InsightPulse AI — UI logic

   Backend contract is unchanged:
     GET  /api/agent/tools        capabilities + tool catalog
     POST /api/agent/run/stream   SSE: run_started | activity | result | error | done
     POST /api/agent/run          non-streaming fallback

   Performance approach: nothing re-renders the page. Streaming events only
   touch three things — the step tracker, one status line, and an appended log
   row. The report renders once, when the result arrives.
   ═══════════════════════════════════════════════════════════ */

(() => {
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ── element cache ─────────────────────────────────────── */
const ui = {
  status: $("status"), statusText: $("status-text"),
  form: $("run-form"), goal: $("goal"),
  runBtn: $("run-btn"), stopBtn: $("stop-btn"), formError: $("form-error"),
  maxIter: $("max-iterations"), mode: $("mode"),
  progress: $("progress-card"), progressMsg: $("progress-message"),
  barFill: $("bar-fill"), steps: $("steps"),
  liveLog: $("live-log"), liveLogToggle: $("live-log-toggle"),
  report: $("report"), reportMeta: $("report-meta"),
  summaryLead: $("summary-lead"), summaryCounts: $("summary-counts"),
  trendBlock: $("trend-block"), summaryTrend: $("summary-trend"),
  nextBlock: $("next-block"), summaryNext: $("summary-next"),
  fullSummary: $("full-summary"), summaryFullText: $("summary-full-text"),
  topInsight: $("top-insight"), filters: $("filters"), sourceFilters: $("source-filters"),
  grid: $("insight-grid"), emptyNote: $("empty-note"),
  agentStory: $("agent-story"), flow: $("flow"),
  techStats: $("tech-stats"), techLog: $("tech-log"),
  techObs: $("tech-observations"), techFind: $("tech-findings"),
  drawer: $("drawer"), drawerBackdrop: $("drawer-backdrop"), drawerBody: $("drawer-body"),
  modal: $("modal"), modalBackdrop: $("modal-backdrop"),
  modalBody: $("modal-body"), modalTitle: $("modal-title"),
  newScan: $("new-scan-btn"),
};

/* ── state ─────────────────────────────────────────────── */
let controller = null;
let lastResult = null;
const filter = { priority: "all", source: "all" };

const TOOL_LABEL = {
  research_search: "research papers",
  news_search: "industry news",
  competitor_search: "competitor activity",
  patent_search: "patents",
};
const SOURCE_LABEL = {
  research: "Research", news: "News", competitor: "Competitors", patent: "Patents",
};

/* ═══ CHIP INPUTS ═══════════════════════════════════════ */
function makeChipField(boxId, chipsId, inputId, initial) {
  const values = [...initial];
  const chipsEl = $(chipsId), inputEl = $(inputId);

  const render = () => {
    chipsEl.innerHTML = values.map((v, i) =>
      `<span class="chip">${esc(v)}<button type="button" data-i="${i}" aria-label="Remove ${esc(v)}">×</button></span>`
    ).join("");
  };

  const add = (raw) => {
    raw.split(",").map((s) => s.trim()).filter(Boolean).forEach((v) => {
      if (values.length < 10 && !values.some((x) => x.toLowerCase() === v.toLowerCase())) {
        values.push(v.slice(0, 120));
      }
    });
    inputEl.value = "";
    render();
  };

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); add(inputEl.value); }
    else if (e.key === "Backspace" && !inputEl.value && values.length) { values.pop(); render(); }
  });
  inputEl.addEventListener("blur", () => { if (inputEl.value.trim()) add(inputEl.value); });
  chipsEl.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-i]");
    if (btn) { values.splice(Number(btn.dataset.i), 1); render(); }
  });
  $(boxId).addEventListener("click", (e) => { if (e.target === e.currentTarget) inputEl.focus(); });

  render();
  return {
    get: () => [...values],
    set: (list) => { values.length = 0; list.forEach((v) => v && values.push(v)); render(); },
  };
}

const keywords = makeChipField("kw-box", "kw-chips", "kw-input", []);
const competitors = makeChipField("comp-box", "comp-chips", "comp-input", []);

/* ═══ CAPABILITIES (settings only) ══════════════════════ */
let capabilities = null;

async function loadCapabilities() {
  try {
    const res = await fetch("/api/agent/tools");
    if (!res.ok) throw new Error("HTTP " + res.status);
    capabilities = await res.json();
  } catch (err) {
    capabilities = { error: err.message };
    setStatus("error", "Backend offline");
  }
}

function renderSettings() {
  if (!capabilities) { ui.drawerBody.innerHTML = `<p class="muted">Loading…</p>`; return; }
  if (capabilities.error) {
    ui.drawerBody.innerHTML = `<p class="muted">Backend unreachable: ${esc(capabilities.error)}</p>`;
    return;
  }
  const caps = capabilities.capabilities || {};
  const llm = caps.llm || {};
  const keyed = caps.keyed_sources || {};
  const live = Object.entries(keyed).filter(([, v]) => v).map(([k]) => k);
  const sim = Object.entries(keyed).filter(([, v]) => !v).map(([k]) => k);

  ui.drawerBody.innerHTML = `
    <div class="setting-group">
      <h3>Reasoning engine</h3>
      <div class="setting-row"><span>Mode</span><span class="${llm.live ? "dot-ok" : "dot-warn"}">
        ${llm.live ? "AI model" : "built-in reasoner"}</span></div>
      <div class="setting-row"><span>Provider</span><span>${esc(llm.provider || "—")}</span></div>
      <div class="setting-row"><span>Model</span><span>${esc(llm.model || "—")}</span></div>
      ${!llm.live ? `<p class="muted" style="font-size:12px;margin-top:10px">
        No AI credential is active, so the agent uses its deterministic reasoner.
        Planning, tool choice and prioritization all still run.</p>` : ""}
    </div>

    <div class="setting-group">
      <h3>Data sources</h3>
      <div class="setting-row"><span>Keyless &amp; live</span>
        <span>${esc((caps.keyless_sources || []).join(", ") || "—")}</span></div>
      ${live.length ? `<div class="setting-row"><span>Keyed &amp; live</span><span>${esc(live.join(", "))}</span></div>` : ""}
      ${sim.length ? `<div class="setting-row"><span>Demo data</span><span>${esc(sim.join(", "))}</span></div>` : ""}
      <div class="setting-row"><span>Simulation mode</span>
        <span>${capabilities.simulation_mode ? "on" : "off"}</span></div>
    </div>

    <div class="setting-group">
      <h3>Agent tools (${(capabilities.usable || []).length})</h3>
      ${(capabilities.tools || []).map((t) => `
        <div class="tool-row">
          <b>${esc(t.display_name)}</b>
          <p>${esc(t.when_to_use)}</p>
          <div class="provs">
            ${(t.providers_live || []).map((p) => `<span class="tag">${esc(p)}</span>`).join("")}
            ${(t.providers_simulated || []).map((p) => `<span class="tag sim">${esc(p)}</span>`).join("")}
          </div>
        </div>`).join("")}
    </div>`;
}

/* ═══ STATUS + STEPPER ═════════════════════════════════ */
function setStatus(state, text) {
  ui.status.dataset.state = state;
  ui.statusText.textContent = text;
}

const STEP_ORDER = ["goal", "plan", "search", "analyze", "insights"];

function setStep(name, state) {
  const li = ui.steps.querySelector(`li[data-step="${name}"]`);
  if (li && li.dataset.state !== state) li.dataset.state = state;
}

function advanceTo(name, searchLabel) {
  const idx = STEP_ORDER.indexOf(name);
  if (idx < 0) return;
  STEP_ORDER.forEach((s, i) => setStep(s, i < idx ? "done" : i === idx ? "active" : ""));
  ui.barFill.style.width = Math.round(((idx + 0.55) / STEP_ORDER.length) * 100) + "%";
  if (searchLabel) {
    const label = ui.steps.querySelector('li[data-step="search"] [data-label]');
    if (label) label.textContent = searchLabel;
  }
}

function completeSteps() {
  STEP_ORDER.forEach((s) => setStep(s, "done"));
  ui.barFill.style.width = "100%";
}

function resetSteps() {
  STEP_ORDER.forEach((s) => setStep(s, ""));
  ui.barFill.style.width = "6%";
  const label = ui.steps.querySelector('li[data-step="search"] [data-label]');
  if (label) label.textContent = "Searching sources";
}

/* ═══ LIVE EVENT HANDLING ══════════════════════════════ */
/* Plain-language message per phase — the technical text stays in the log. */
function humanMessage(entry) {
  const d = entry.data || {};
  switch (entry.phase) {
    case "start": return "Reading your goal…";
    case "goal": return "Goal understood. Choosing where to look.";
    case "plan": {
      const needs = (d.required_needs || []).map((n) => SOURCE_LABEL[n] || n);
      return needs.length ? `Plan ready — will check ${needs.join(", ").toLowerCase()}.` : "Plan ready.";
    }
    case "decision": return d.tool ? `Decided to search ${TOOL_LABEL[d.tool] || d.tool}.` : "Deciding what to do next…";
    case "action": return `Searching ${TOOL_LABEL[d.tool] || d.tool || "sources"}…`;
    case "observation": {
      const rel = d.relevant ?? 0;
      const dup = d.duplicates ?? 0;
      if (dup > 0) return `Found ${rel} relevant results, skipped ${dup} duplicate${dup === 1 ? "" : "s"}.`;
      return `Found ${rel} relevant result${rel === 1 ? "" : "s"}.`;
    }
    case "thought": return entry.title || "Analyzing what came back…";
    case "warning": return "One source was unavailable — continuing with the others.";
    case "error": return "A source failed. Continuing with the rest.";
    case "final": return "Enough information gathered. Analyzing strategic importance…";
    case "insight": return d.priority_counts ? "Writing your report…" : "Ranking what matters most…";
    case "done": return "Report ready.";
    default: return entry.title || "Working…";
  }
}

function phaseToStep(entry) {
  switch (entry.phase) {
    case "start": case "goal": return "goal";
    case "plan": return "plan";
    case "decision": case "action": case "observation": return "search";
    case "thought": case "final": return "analyze";
    case "insight": case "done": return "insights";
    default: return null;
  }
}

function onActivity(entry) {
  const step = phaseToStep(entry);
  if (step) {
    const label = entry.phase === "action" && entry.data && entry.data.tool
      ? "Searching " + (TOOL_LABEL[entry.data.tool] || entry.data.tool)
      : null;
    advanceTo(step, label);
  }
  // "thought" entries during search shouldn't drag the tracker backwards.
  if (entry.phase === "thought" && entry.iteration) advanceTo("search");

  const msg = humanMessage(entry);
  if (msg) ui.progressMsg.textContent = msg;

  appendLiveRow(entry);
}

function appendLiveRow(entry) {
  const row = document.createElement("div");
  row.className = "row" + (entry.phase === "warning" ? " warn" : entry.phase === "error" ? " err" : "");
  row.innerHTML = `<span>${esc(entry.icon)}</span><span><b>${esc(entry.label)}</b>${
    entry.title && entry.title !== entry.label ? " — " + esc(entry.title) : ""}${
    entry.detail ? "<br>" + esc(entry.detail) : ""}</span>`;
  ui.liveLog.appendChild(row);
  if (!ui.liveLog.hidden) ui.liveLog.scrollTop = ui.liveLog.scrollHeight;
}

/* ═══ RUN ══════════════════════════════════════════════ */
function buildPayload() {
  return {
    goal: ui.goal.value.trim(),
    keywords: keywords.get(),
    competitors: competitors.get(),
    max_iterations: Math.min(25, Math.max(1, Number(ui.maxIter.value) || 10)),
    simulation_mode: ui.mode.value === "sim",
  };
}

function startRunUI() {
  ui.formError.hidden = true;
  ui.report.hidden = true;
  ui.progress.hidden = false;
  ui.liveLog.innerHTML = "";
  resetSteps();
  ui.progressMsg.textContent = "Getting started…";
  ui.runBtn.disabled = true;
  ui.stopBtn.hidden = false;
  setStatus("running", "Agent Running");
}

function endRunUI() {
  ui.runBtn.disabled = false;
  ui.stopBtn.hidden = true;
}

async function run() {
  const payload = buildPayload();
  if (payload.goal.length < 3) {
    ui.formError.textContent = "Tell the agent what to track first.";
    ui.formError.hidden = false;
    return;
  }

  startRunUI();
  controller = new AbortController();

  try {
    const res = await fetch("/api/agent/run/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error("stream unavailable (HTTP " + res.status + ")");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let got = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }

        if (ev.type === "activity" && ev.entry) onActivity(ev.entry);
        else if (ev.type === "result") { got = true; finish(ev.result); }
        else if (ev.type === "error") throw new Error(ev.message || "agent error");
      }
    }
    if (!got) throw new Error("the run ended without a report");
  } catch (err) {
    if (err.name === "AbortError") {
      ui.progress.hidden = true;
      setStatus("ready", "Agent Ready");
    } else {
      await runFallback(payload, err);
    }
  } finally {
    controller = null;
    endRunUI();
  }
}

/* Streaming can be blocked by a proxy — the plain endpoint still works. */
async function runFallback(payload, streamErr) {
  ui.progressMsg.textContent = "Working… (live updates unavailable)";
  try {
    const res = await fetch("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const result = await res.json();
    ui.liveLog.innerHTML = "";
    (result.activity_log || []).forEach(appendLiveRow);
    finish(result);
  } catch (err) {
    setStatus("error", "Run failed");
    ui.progress.hidden = true;
    ui.formError.textContent = `Could not complete the scan: ${err.message}` +
      (streamErr ? ` (stream: ${streamErr.message})` : "");
    ui.formError.hidden = false;
  }
}

function finish(result) {
  lastResult = result;
  completeSteps();
  ui.progressMsg.textContent = "Report ready.";
  setStatus("ready", "Agent Ready");
  renderReport(result);
  ui.progress.hidden = true;
  ui.report.hidden = false;
  ui.report.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ═══ REPORT ═══════════════════════════════════════════ */
function renderReport(result) {
  const insights = result.insights || [];
  const m = result.metrics || {};
  const counts = m.priority_counts || { HIGH: 0, MEDIUM: 0, LOW: 0 };

  ui.reportMeta.textContent =
    `${insights.length} relevant development${insights.length === 1 ? "" : "s"} found` +
    (counts.HIGH ? ` • ${counts.HIGH} high priority` : "");

  renderSummary(result, insights, counts);
  renderTopInsight(insights[0]);
  renderSourceFilters(insights);
  filter.priority = "all"; filter.source = "all";
  syncPills();
  renderCards(insights);
  renderStory(result);
}

function renderSummary(result, insights, counts) {
  if (!insights.length) {
    ui.summaryLead.textContent =
      "No developments cleared the relevance bar this time. Try broader keywords or a longer window.";
    ui.summaryCounts.innerHTML = "";
    ui.trendBlock.hidden = ui.nextBlock.hidden = true;
  } else {
    ui.summaryLead.textContent =
      `We found ${insights.length} relevant development${insights.length === 1 ? "" : "s"}.`;

    const rows = [
      ["HIGH", counts.HIGH, "requires immediate attention"],
      ["MEDIUM", counts.MEDIUM, "worth monitoring"],
      ["LOW", counts.LOW, "low priority"],
    ].filter(([, n]) => n > 0);
    ui.summaryCounts.innerHTML = rows.map(([k, n, label]) =>
      `<li><span class="swatch ${k}"></span>${n} ${esc(label)}</li>`).join("");

    const trend = deriveTrend(result, insights);
    ui.summaryTrend.textContent = trend;
    ui.trendBlock.hidden = !trend;

    const next = insights[0] ? insights[0].recommended_action : "";
    ui.summaryNext.textContent = next;
    ui.nextBlock.hidden = !next;
  }

  const full = (result.summary || "").trim();
  if (full) {
    ui.summaryFullText.innerHTML = full.split("\n\n").map((p) => `<p>${esc(p)}</p>`).join("");
    ui.fullSummary.hidden = false;
    ui.fullSummary.open = false;
  } else {
    ui.fullSummary.hidden = true;
  }
}

/* One readable sentence about the dominant pattern. */
function deriveTrend(result, insights) {
  const sig = (result.metrics && result.metrics.signals_detected) || [];
  const PHRASE = {
    patent: "patent and IP activity",
    launch: "product launches",
    funding: "funding and investment activity",
    partnership: "partnership activity",
    acquisition: "acquisitions",
    regulatory: "regulatory attention",
    benchmark: "benchmark and performance claims",
    hiring: "team and hiring moves",
  };
  const coverage = (result.metrics && result.metrics.coverage) || {};
  const topSource = Object.entries(coverage).sort((a, b) => b[1] - a[1])[0];
  const companies = [...new Set(insights.map((i) => i.competitor).filter(Boolean))];

  const parts = [];
  if (sig.length) {
    parts.push(`Increased ${sig.slice(0, 2).map((s) => PHRASE[s] || s).join(" and ")}`);
  } else if (topSource) {
    parts.push(`Most activity is coming from ${SOURCE_LABEL[topSource[0]] || topSource[0]}`.toLowerCase()
      .replace(/^./, (c) => c.toUpperCase()));
  }
  if (companies.length) parts.push(`concentrated around ${companies.slice(0, 2).join(" and ")}`);
  return parts.length ? parts.join(", ") + "." : "";
}

function renderTopInsight(top) {
  if (!top) { ui.topInsight.hidden = true; return; }
  ui.topInsight.hidden = false;
  ui.topInsight.style.borderLeftColor =
    top.priority === "HIGH" ? "var(--high)" : top.priority === "MEDIUM" ? "var(--med)" : "var(--low)";

  ui.topInsight.innerHTML = `
    <span class="ti-label">Top insight</span>
    <div style="margin-top:8px">${badge(top.priority)}</div>
    <h3>${esc(top.title)}</h3>
    <p class="ti-summary">${esc(top.summary || top.what_happened)}</p>
    <div class="ti-fields">
      <div class="summary-block">
        <span class="block-label">Why it matters</span>
        <p>${esc(top.why_it_matters)}</p>
      </div>
      <div class="summary-block">
        <span class="block-label">Recommended action</span>
        <p>${esc(top.recommended_action)}</p>
      </div>
    </div>
    <div class="ti-actions">
      ${top.source_url ? `<a class="mini-btn" href="${esc(top.source_url)}" target="_blank" rel="noopener">View source ↗</a>` : ""}
      <button type="button" class="mini-btn" data-detail="${esc(top.id)}">View details</button>
    </div>`;
}

const badge = (p) => `<span class="badge ${esc(p)}"><span class="swatch"></span>${esc(p)} priority</span>`;

function renderSourceFilters(insights) {
  const present = [...new Set(insights.map(sourceKey))].filter(Boolean);
  if (present.length < 2) { ui.sourceFilters.innerHTML = ""; ui.filters.hidden = !insights.length; return; }
  ui.filters.hidden = false;
  ui.sourceFilters.innerHTML =
    `<button type="button" class="pill active" data-filter="source" data-value="all">All sources</button>` +
    present.map((s) => `<button type="button" class="pill" data-filter="source" data-value="${esc(s)}">${
      esc(SOURCE_LABEL[s] || s)}</button>`).join("");
}

/* Insight.source looks like "Competitor activity · rss" — take the leading word. */
function sourceKey(insight) {
  const head = String(insight.source || "").split("·")[0].trim().toLowerCase();
  if (head.startsWith("research")) return "research";
  if (head.startsWith("patent")) return "patent";
  if (head.startsWith("competitor")) return "competitor";
  if (head.startsWith("news")) return "news";
  return "news";
}

function renderCards(insights) {
  const visible = insights.filter((i) =>
    (filter.priority === "all" || i.priority === filter.priority) &&
    (filter.source === "all" || sourceKey(i) === filter.source));

  if (!visible.length) {
    ui.grid.innerHTML = "";
    ui.emptyNote.hidden = false;
    ui.emptyNote.textContent = insights.length
      ? "Nothing matches these filters."
      : "No insights were produced. Try broader keywords or demo data mode.";
    return;
  }
  ui.emptyNote.hidden = true;

  ui.grid.innerHTML = visible.map((i) => {
    const key = sourceKey(i);
    return `
    <article class="insight">
      ${badge(i.priority)}
      <h4>${esc(i.title)}</h4>
      <p class="meta">
        <span class="tag ${esc(key)}">${esc(SOURCE_LABEL[key] || key)}</span>
        ${esc(formatDate(i.published_date))}${i.competitor ? " · " + esc(i.competitor) : ""}
        ${i.simulated ? ' · <span class="tag sim">demo</span>' : ""}
      </p>
      <p class="excerpt">${esc(i.summary || i.what_happened)}</p>
      <div class="why">
        <span>Why it matters</span>
        <p>${esc(i.why_it_matters)}</p>
      </div>
      <p class="action">→ ${esc(i.recommended_action)}</p>
      <div class="card-actions">
        <button type="button" class="mini-btn" data-detail="${esc(i.id)}">View details</button>
        ${i.source_url ? `<a class="mini-btn" href="${esc(i.source_url)}" target="_blank" rel="noopener">Source ↗</a>` : ""}
      </div>
    </article>`;
  }).join("");
}

function formatDate(iso) {
  if (!iso) return "date unknown";
  const d = new Date(iso + "T00:00:00");
  return isNaN(d) ? iso
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

/* ═══ AGENT STORY (judge-facing proof) ═════════════════ */
function renderStory(result) {
  const st = result.state || {};
  const m = result.metrics || {};
  const steps = [];

  steps.push(`<li><b>Goal understood</b><em>${esc(st.user_goal || "")}</em></li>`);

  const plan = st.plan || {};
  const req = (plan.needs || []).filter((n) => n.required).map((n) => n.key);
  steps.push(`<li><b>Plan created</b><em>Must satisfy: ${esc(req.join(", ") || "none")}${
    plan.author ? ` · by ${esc(plan.author)}` : ""}</em></li>`);

  const held = (plan.needs || []).filter((n) => !n.required).map((n) => n.key);
  if (held.length) {
    steps.push(`<li><b>Held back ${esc(held.join(", "))}</b><em>Not called unless the evidence justified it — this is the agent choosing, not a fixed script.</em></li>`);
  }

  (st.tool_calls || []).forEach((call, idx) => {
    const dec = (st.decisions || [])[idx];
    steps.push(`<li><b>Tool selected: ${esc(TOOL_LABEL[call.tool] || call.tool)}</b><em>${
      esc(dec ? dec.reasoning : call.reasoning || "")}</em></li>`);
    const obs = (st.observations || []).find((o) => o.iteration === call.iteration);
    steps.push(`<li><b>Observed ${call.items_returned} result${call.items_returned === 1 ? "" : "s"}</b><em>${
      esc(obs ? obs.summary : "")}</em></li>`);
    if (obs) {
      steps.push(`<li><b>Analyzed relevance</b><em>${esc(obs.relevant_items)} relevant, yield judged “${
        esc(obs.yield_quality)}”${obs.signals && obs.signals.length ? ` · signals: ${esc(obs.signals.join(", "))}` : ""}</em></li>`);
    }
  });

  steps.push(`<li><b>Decided collection was complete</b><em>${esc(st.stop_reason || st.final_decision || "")}</em></li>`);
  steps.push(`<li><b>Generated ${m.insights || 0} prioritized insight${m.insights === 1 ? "" : "s"}</b><em>${
    esc(`${(m.priority_counts || {}).HIGH || 0} high, ${(m.priority_counts || {}).MEDIUM || 0} medium, ${(m.priority_counts || {}).LOW || 0} low`)}</em></li>`);

  ui.flow.innerHTML = steps.join("");

  const llm = m.llm || {};
  ui.techStats.innerHTML = [
    ["Reasoning steps", `${m.iterations ?? "–"}/${m.max_iterations ?? "–"}`],
    ["Tool calls", m.tool_calls ?? "–"],
    ["Findings", m.findings_total ?? "–"],
    ["Relevant", m.findings_relevant ?? "–"],
    ["Duplicates cut", m.duplicates_suppressed ?? 0],
    ["Duration", m.duration_ms != null ? (m.duration_ms / 1000).toFixed(1) + "s" : "–"],
    ["Reasoner", m.reasoner || "–"],
    ["Model calls", llm.calls ?? 0],
    ["Est. cost", "$" + (llm.cost_usd ?? 0)],
    ["Errors handled", m.errors ?? 0],
  ].map(([label, v]) => `<div class="tech-stat"><b>${esc(v)}</b><span>${esc(label)}</span></div>`).join("");

  ui.techLog.innerHTML = (result.activity_log || []).map((e) => `
    <div class="tech-line ${e.phase === "warning" ? "warn" : e.phase === "error" ? "err" : ""}">
      <span class="t">${esc(e.icon)}</span>
      <span><b>${esc(e.label)}${e.iteration ? ` · step ${e.iteration}` : ""}${
        e.title && e.title !== e.label ? " — " + esc(e.title) : ""}</b>
        ${e.detail ? `<p>${esc(e.detail)}</p>` : ""}</span>
      <time>${(e.elapsed_ms / 1000).toFixed(1)}s</time>
    </div>`).join("");

  ui.techObs.innerHTML = (st.observations || []).map((o) => `
    <div class="tech-line">
      <span class="t">👁</span>
      <span><b>step ${o.iteration} · ${esc(o.tool)} · ${esc(o.yield_quality)}</b>
        <p>${esc(o.summary)}</p>
        ${(o.top_titles || []).map((t) => `<p>• ${esc(t)}</p>`).join("")}</span>
      <time></time>
    </div>`).join("") || `<p class="muted">No observations.</p>`;

  ui.techFind.innerHTML = (result.findings || []).map((f) => `
    <div class="tech-line">
      <span class="t"><span class="tag ${esc(f.source)}">${esc(f.source)}</span></span>
      <span><b>${f.url ? `<a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.title)}</a>` : esc(f.title)}</b>
        <p>${esc(f.provider)} · ${esc(f.published_date || "date unknown")}${
          f.competitor ? " · " + esc(f.competitor) : ""}${f.simulated ? " · demo data" : ""}${
          (f.signals || []).length ? " · " + esc(f.signals.join(", ")) : ""}</p></span>
      <time>${esc(f.relevance)}</time>
    </div>`).join("") || `<p class="muted">No findings.</p>`;
}

/* ═══ MODAL ════════════════════════════════════════════ */
function openDetail(id) {
  if (!lastResult) return;
  const i = (lastResult.insights || []).find((x) => x.id === id);
  if (!i) return;

  ui.modalTitle.textContent = i.priority + " priority insight";
  ui.modalBody.innerHTML = `
    <div>${badge(i.priority)}</div>
    <h3>${esc(i.title)}</h3>
    <div class="modal-field"><span>What happened</span><p>${esc(i.what_happened)}</p></div>
    <div class="modal-field"><span>Summary</span><p>${esc(i.summary)}</p></div>
    <div class="modal-field"><span>Why it matters</span><p>${esc(i.why_it_matters)}</p></div>
    <div class="modal-field"><span>Recommended action</span><p>${esc(i.recommended_action)}</p></div>
    <div class="modal-field"><span>Source</span><p>${esc(i.source)} · ${esc(formatDate(i.published_date))}${
      i.competitor ? " · " + esc(i.competitor) : ""}<br>
      <span class="muted" style="font-size:12px">confidence: ${esc(i.confidence)} · relevance: ${esc(i.score)} · written by ${esc(i.author)}</span></p></div>
    ${i.source_url ? `<a class="mini-btn" href="${esc(i.source_url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}`;

  ui.modal.hidden = false;
  ui.modalBackdrop.hidden = false;
  $("modal-close").focus();
}

const closeModal = () => { ui.modal.hidden = true; ui.modalBackdrop.hidden = true; };
const openDrawer = () => {
  renderSettings();
  ui.drawer.hidden = false; ui.drawerBackdrop.hidden = false;
  $("settings-btn").setAttribute("aria-expanded", "true");
  $("drawer-close").focus();
};
const closeDrawer = () => {
  ui.drawer.hidden = true; ui.drawerBackdrop.hidden = true;
  $("settings-btn").setAttribute("aria-expanded", "false");
};

/* ═══ WIRING ═══════════════════════════════════════════ */
ui.form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
ui.stopBtn.addEventListener("click", () => controller && controller.abort());

ui.goal.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
});

$("suggestions").addEventListener("click", (e) => {
  const btn = e.target.closest(".suggestion");
  if (!btn) return;
  ui.goal.value = btn.dataset.goal || "";
  keywords.set((btn.dataset.kw || "").split(",").map((s) => s.trim()).filter(Boolean));
  competitors.set((btn.dataset.comp || "").split(",").map((s) => s.trim()).filter(Boolean));
  ui.goal.focus();
});

ui.liveLogToggle.addEventListener("click", () => {
  const show = ui.liveLog.hidden;
  ui.liveLog.hidden = !show;
  ui.liveLogToggle.setAttribute("aria-expanded", String(show));
  ui.liveLogToggle.textContent = show ? "Hide detailed agent activity" : "View detailed agent activity";
  if (show) ui.liveLog.scrollTop = ui.liveLog.scrollHeight;
});

function syncPills() {
  document.querySelectorAll(".pill").forEach((p) => {
    p.classList.toggle("active", filter[p.dataset.filter] === p.dataset.value);
  });
}

ui.filters.addEventListener("click", (e) => {
  const pill = e.target.closest(".pill");
  if (!pill) return;
  filter[pill.dataset.filter] = pill.dataset.value;
  syncPills();
  renderCards((lastResult && lastResult.insights) || []);
});

/* One delegated listener covers every "View details" button. */
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-detail]");
  if (btn) { openDetail(btn.dataset.detail); return; }
  const tab = e.target.closest(".tech-tab");
  if (tab) {
    document.querySelectorAll(".tech-tab").forEach((t) => t.classList.toggle("active", t === tab));
    ["log", "observations", "findings"].forEach((n) => {
      const panel = $("tech-" + (n === "log" ? "log" : n === "observations" ? "observations" : "findings"));
      panel.hidden = n !== tab.dataset.tech;
    });
  }
});

$("settings-btn").addEventListener("click", openDrawer);
$("drawer-close").addEventListener("click", closeDrawer);
ui.drawerBackdrop.addEventListener("click", closeDrawer);
$("modal-close").addEventListener("click", closeModal);
ui.modalBackdrop.addEventListener("click", closeModal);

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!ui.modal.hidden) closeModal();
  else if (!ui.drawer.hidden) closeDrawer();
});

ui.newScan.addEventListener("click", () => {
  ui.report.hidden = true;
  ui.goal.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

loadCapabilities();
})();
