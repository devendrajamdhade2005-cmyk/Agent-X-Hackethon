/* InsightPulse AI — application shell.
 *
 * Architecture notes:
 *  - One run lives in the store; nav sections are client-side views over it, so
 *    switching sections never refetches and never re-runs the agent (§27).
 *  - During streaming only three things touch the DOM: the tracker, one status
 *    line, and an appended tick row. The dashboard renders once, on completion.
 *  - Heavy panels (charts, landscape, feeds) render lazily on first view.
 */

import * as api from "./core/api.js";
import { $, countUp, delegate, esc, onVisible, qsa, setHTML, setText, show } from "./core/dom.js";
import {
  CATEGORY, PRIORITY, categoryIcon, categoryLabel, categoryOf,
  formatDate, pct, providerLabel, toolHuman, truncate,
} from "./core/format.js";
import { store } from "./core/store.js";
import { areaChart, bubbles, donut, donutColor, meter } from "./analytics/charts.js";
import * as cards from "./intelligence/cards.js";
import * as agentUI from "./agent/activity.js";
import { renderMultiAgent } from "./agent/multiagent.js";
import { renderMemory } from "./agent/memory.js";
import { initFramework } from "./agent/framework.js";
import { initEvaluation } from "./agent/evaluation.js";
import { closeDrawer, openDrawer } from "./drawers/detailDrawer.js";

const VIEWS = ["overview", "research", "competitors", "patents", "news", "insights", "framework", "evaluation", "reports"];
const rendered = new Set();
let controller = null;
let trackerIndex = -1;

/* ═══════════════════════════════════════════════════════════
   BOOT
   ═══════════════════════════════════════════════════════════ */
async function boot() {
  agentUI.renderTracker($("tracker"));
  wireEvents();
  renderEmptyDashboard();

  try {
    store.tools = await api.getTools();
    renderSystemInfo();
    setAgentStatus("ready", "Agent Ready");
  } catch (err) {
    setAgentStatus("error", "Backend offline");
    setHTML($("sysinfo"), `<p class="muted small">Could not reach the backend: ${esc(err.message)}</p>`);
  }
}

/* ═══════════════════════════════════════════════════════════
   AGENT STATUS
   ═══════════════════════════════════════════════════════════ */
function setAgentStatus(state, text) {
  const el = $("agent-status");
  el.dataset.state = state;
  setText($("agent-status-text"), text);
}

/* ═══════════════════════════════════════════════════════════
   RUN
   ═══════════════════════════════════════════════════════════ */
function payload() {
  return {
    goal: $("goal").value.trim(),
    keywords: chips.keywords.get(),
    competitors: chips.competitors.get(),
    max_iterations: Math.min(25, Math.max(1, Number($("max-iter").value) || 10)),
    simulation_mode: $("mode").value === "sim",
  };
}

async function run() {
  const body = payload();
  if (body.goal.length < 3) {
    setText($("search-error"), "Tell the agent what to track first.");
    show($("search-error"), true);
    $("goal").focus();
    return;
  }
  show($("search-error"), false);

  // live UI
  show($("live"), true);
  show($("dash"), false);
  setHTML($("ticks"), "");
  trackerIndex = -1;
  agentUI.advanceTracker($("tracker"), "goal");
  setText($("live-msg"), "Starting the agent…");
  $("run-btn").disabled = true;
  show($("stop-btn"), true);
  setAgentStatus("running", "Agent Working");
  $("live").scrollIntoView({ behavior: "smooth", block: "center" });

  controller = new AbortController();
  try {
    // Collecting and rendering are separate failure domains. A render bug must
    // never be reported as "scan failed" — that would discard a run that
    // actually succeeded and send the agent out to do the work a second time.
    const result = await api.runAgentStream(body, onAgentEvent, controller.signal);
    finish(result);
  } catch (err) {
    if (err.name === "AbortError") {
      show($("live"), false);
      setAgentStatus("ready", "Agent Ready");
      if (store.hasRun()) show($("dash"), true);
    } else if (err.__render) {
      reportRenderFailure(err);
    } else {
      await fallbackRun(body, err);
    }
  } finally {
    controller = null;
    $("run-btn").disabled = false;
    show($("stop-btn"), false);
  }
}

/** SSE blocked by a proxy? The plain endpoint still works. */
async function fallbackRun(body, streamErr) {
  setText($("live-msg"), "Working… (live updates unavailable)");
  try {
    const result = await api.runAgent(body);
    setHTML($("ticks"), (result.activity_log || []).map(agentUI.liveRow).join(""));
    finish(result);
  } catch (err) {
    setAgentStatus("error", "Run failed");
    show($("live"), false);
    setText($("search-error"), `Scan failed: ${err.message}${streamErr ? ` (stream: ${streamErr.message})` : ""}`);
    show($("search-error"), true);
  }
}

function onAgentEvent(event) {
  if (event.type !== "activity" || !event.entry) return;
  const entry = event.entry;

  const step = agentUI.phaseToStep(entry);
  if (step) {
    const idx = agentUI.STEPS.findIndex((s) => s.key === step);
    if (idx >= trackerIndex) {
      trackerIndex = idx;
      const label = entry.phase === "action" && entry.data?.tool
        ? `Searching ${toolHuman(entry.data.tool)}`
        : null;
      agentUI.advanceTracker($("tracker"), step, label);
    }
  }

  const msg = agentUI.humanMessage(entry);
  if (msg) setText($("live-msg"), msg);

  const ticks = $("ticks");
  ticks.insertAdjacentHTML("beforeend", agentUI.liveRow(entry));
  if (!$("ticks-wrap").hidden) ticks.scrollTop = ticks.scrollHeight;
}

function finish(result) {
  // Store the run first: even if a panel fails to draw, the data is safe and the
  // user keeps their scan.
  store.setRun(result);
  agentUI.completeTracker($("tracker"));
  setAgentStatus("ready", "Agent Monitoring");
  rendered.clear();

  try {
    renderDashboard();
  } catch (err) {
    err.__render = true;
    show($("live"), false);
    show($("dash"), true);
    throw err;
  }

  show($("live"), false);
  show($("dash"), true);
  switchView("overview");
  $("dash").scrollIntoView({ behavior: "smooth", block: "start" });
}

/** A panel failed to draw. Keep the run, show what worked, say what didn't. */
function reportRenderFailure(err) {
  show($("dash"), true);
  setAgentStatus("ready", "Agent Monitoring");
  setText($("search-error"),
    `The scan completed and your data is safe, but part of the dashboard failed to render: ${err.message}`);
  show($("search-error"), true);
  try { switchView("overview"); } catch { /* already reported */ }
}

/** Each panel renders independently, so one failure cannot blank the dashboard. */
function safeRender(label, fn) {
  try { fn(); } catch (err) {
    console.error(`[InsightPulse] ${label} failed to render`, err);
    return err;
  }
  return null;
}

/* ═══════════════════════════════════════════════════════════
   VIEW ROUTING (client-side, no refetch)
   ═══════════════════════════════════════════════════════════ */
function switchView(view) {
  if (!VIEWS.includes(view)) view = "overview";
  store.view = view;

  qsa("[data-nav]").forEach((b) =>
    b.classList.toggle("is-on", b.dataset.nav === view));
  qsa("[data-view]").forEach((sec) => { sec.hidden = sec.dataset.view !== view; });

  // The framework view is self-driving (it launches its own LangGraph runs), so it
  // must work before any classic scan has happened. Every [data-view] section lives
  // inside #dash, which stays hidden until a scan finishes — so reveal it here, or
  // un-hiding the section alone would still leave the whole subtree display:none.
  if (view === "framework") {
    show($("dash"), true);
    initFramework($("framework-body"));
  }
  // Evaluation is likewise self-driving: it runs its own benchmark suites and reads
  // stored results, so it must work before any classic scan has happened.
  if (view === "evaluation") {
    show($("dash"), true);
    initEvaluation($("evaluation-body"));
  }

  renderView(view);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderView(view) {
  if (!store.hasRun()) return;
  if (rendered.has(view)) return;
  rendered.add(view);

  const renderers = {
    overview:    renderOverview,
    research:    () => renderFeedView("research"),
    patents:     () => renderFeedView("patent"),
    news:        renderNewsView,
    competitors: renderCompetitors,
    insights:    renderInsights,
    reports:     renderReports,
  };
  const err = safeRender(`${view} view`, renderers[view] || (() => {}));
  if (err) {
    rendered.delete(view);
    const host = $(`${view}-body`);
    if (host) {
      setHTML(host, cards.emptyState({
        icon: "⚠️", title: "This section could not be displayed",
        body: `${err.message}. Your scan data is intact — other sections still work.`,
      }));
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   DASHBOARD
   ═══════════════════════════════════════════════════════════ */
function renderEmptyDashboard() {
  const empty = cards.emptyState({
    icon: "🔬",
    title: "No intelligence discovered yet",
    body: "Set a tracking goal above and run a scan. The agent will decide which sources to search, read what it finds, and report only what matters.",
    action: '<button type="button" class="btn-primary" data-scroll-search>Run Intelligence Scan</button>',
  });
  setHTML($("overview-body"), empty);
}

function renderDashboard() {
  const failures = [
    safeRender("KPIs", renderKPIs),
    safeRender("executive summary", renderSummary),
    safeRender("reasoning trail", renderTrail),
    safeRender("report bar", renderReportBar),
  ].filter(Boolean);

  if (failures.length === 4) {
    // Everything failed — that is a real defect worth surfacing.
    throw failures[0];
  }
}

function renderKPIs() {
  const list = store.kpis();
  const wrap = $("kpis");
  if (!list) { show(wrap, false); return; }
  show(wrap, true);

  setHTML(wrap, list.map((k) => `
    <article class="kpi accent-${k.accent}">
      <div class="kpi-top">
        <span class="kpi-icon" aria-hidden="true">${k.icon}</span>
        ${k.delta === null
          ? '<span class="kpi-delta none" title="No previous scan to compare against">—</span>'
          : `<span class="kpi-delta ${k.delta >= 0 ? "up" : "down"}">${esc(pct(k.delta))}</span>`}
      </div>
      <b class="kpi-value" data-count="${k.value}" data-suffix="${k.suffix || ""}">0</b>
      <span class="kpi-label">${esc(k.label)}</span>
      <span class="kpi-sub">${esc(k.sub)}</span>
    </article>`).join(""));

  qsa(".kpi-value", wrap).forEach((el) =>
    countUp(el, Number(el.dataset.count), { suffix: el.dataset.suffix || "" }));
}

function renderSummary() {
  const counts = store.counts();
  const insights = store.insights();
  setHTML($("summary"), cards.executiveSummary({
    counts,
    trend: store.trend(),
    nextStep: insights[0]?.recommended_action || "",
    findings: store.findings().length,
    fullSummary: (store.run.summary || "").trim(),
    simulated: store.run.metrics?.simulated_data_used,
  }));

  const top = insights[0];
  show($("hero"), Boolean(top));
  if (top) {
    setHTML($("hero"), cards.topInsight(top));
    $("hero").dataset.tone = (PRIORITY[top.priority] || PRIORITY.MEDIUM).tone;
  }
}

function renderOverview() {
  const body = $("overview-body");
  setHTML(body, `
    <section class="panel" id="chart-panel">
      <div class="panel-head">
        <div>
          <span class="eyebrow">Analytics</span>
          <h2>Intelligence Activity</h2>
          <p class="panel-sub">Findings by publication date, from this scan</p>
        </div>
        <div class="seg" role="group" aria-label="Time window">
          ${[7, 30, 90, 365].map((d) => `
            <button type="button" data-window="${d}"
              class="${d === store.chartWindow ? "is-on" : ""}">${d === 365 ? "1Y" : `${d}D`}</button>`).join("")}
        </div>
      </div>
      <div id="chart-slot"></div>
      <div class="legend">
        <span><i class="sw" style="background:#6366f1"></i>All findings</span>
        <span><i class="sw dash" style="background:#06b6d4"></i>High relevance</span>
      </div>
    </section>

    <div class="grid-2">
      <section class="panel" id="topics-panel">
        <div class="panel-head"><div>
          <span class="eyebrow">Momentum</span><h2>Emerging Topics</h2>
          <p class="panel-sub">Derived from this scan · growth compares recent vs. earlier findings</p>
        </div></div>
        <div id="topics-slot"></div>
      </section>

      <section class="panel" id="sources-panel">
        <div class="panel-head"><div>
          <span class="eyebrow">Evidence</span><h2>Sources &amp; Coverage</h2>
          <p class="panel-sub">Where this intelligence came from</p>
        </div></div>
        <div id="sources-slot"></div>
      </section>
    </div>

    <section class="panel" id="chains-panel">
      <div class="panel-head"><div>
        <span class="eyebrow">Cross-source analysis</span><h2>Connected Intelligence</h2>
        <p class="panel-sub">Signals that appear across more than one source type</p>
      </div></div>
      <div id="chains-slot"></div>
    </section>

    <div class="grid-2">
      <section class="panel" id="landscape-panel">
        <div class="panel-head"><div>
          <span class="eyebrow">Map</span><h2>Research Landscape</h2>
          <p class="panel-sub">Bubble size = volume · colour = growth · click to filter</p>
        </div></div>
        <div id="landscape-slot"></div>
      </section>

      <section class="panel" id="people-panel">
        <div class="panel-head"><div>
          <span class="eyebrow">Attribution</span><h2>Top Contributors</h2>
          <p class="panel-sub">Ranked by citations recorded on this scan's findings</p>
        </div></div>
        <div id="people-slot"></div>
      </section>
    </div>`);

  renderChart();
  onVisible($("topics-panel"), renderTopics);
  onVisible($("sources-panel"), renderSources);
  onVisible($("chains-panel"), renderChains);
  onVisible($("landscape-panel"), renderLandscape);
  onVisible($("people-panel"), renderPeople);
}

function renderChart() {
  const slot = $("chart-slot");
  if (!slot) return;
  const series = store.activity(store.chartWindow);
  if (!series) {
    setHTML(slot, cards.emptyState({
      icon: "📈",
      title: "Not enough dated findings to chart",
      body: "Only findings with a publication date can be plotted. Widen the time window, or run a scan that includes research or news sources.",
    }));
    return;
  }
  setHTML(slot, `<div class="chart-wrap">${areaChart(series)}</div>
    <p class="chart-note">${series.used} of ${series.total} finding(s) carry a publication date within ${series.windowDays} days.</p>`);
}

function renderTopics() {
  const list = store.topics();
  const slot = $("topics-slot");
  if (!list.length) {
    setHTML(slot, cards.emptyState({
      icon: "🔥", title: "No recurring topics yet",
      body: "Topics emerge when several findings share a theme. Try broader keywords or more sources.",
    }));
    return;
  }
  const max = Math.max(...list.map((t) => t.count));
  setHTML(slot, `<ul class="topic-list">
    ${list.map((t) => `
      <li>
        <button type="button" data-topic="${esc(t.key)}"
          class="${store.filters.topic === t.key ? "is-on" : ""}">
          <span class="topic-main">
            <b>${esc(t.label)}</b>
            <span class="topic-meta">${t.count} finding${t.count === 1 ? "" : "s"} · ${t.confidence}% avg relevance</span>
          </span>
          ${meter(t.count, max, { tone: t.isNew ? "pink" : "blue" })}
          <span class="topic-growth ${growthClass(t)}">${growthLabel(t)}</span>
        </button>
      </li>`).join("")}
  </ul>`);
}

const growthClass = (t) => (t.isNew ? "new" : (t.growth ?? 0) > 0 ? "up" : (t.growth ?? 0) < 0 ? "down" : "flat");
const growthLabel = (t) =>
  t.isNew ? "NEW" : t.growth === null ? "—" : `${t.growth > 0 ? "+" : ""}${Math.round(t.growth)}%`;

function renderSources() {
  const list = store.sources();
  const slot = $("sources-slot");
  if (!list.length) {
    setHTML(slot, cards.emptyState({ icon: "🗂", title: "No sources recorded", body: "Run a scan to populate source coverage." }));
    return;
  }
  const slices = list.map((s) => ({ label: providerLabel(s.key), count: s.count }));
  setHTML(slot, `
    <div class="source-wrap">
      <div class="donut-wrap">${donut(slices)}</div>
      <ul class="source-list">
        ${list.map((s, i) => `
          <li>
            <i class="sw" style="background:${donutColor(i)}"></i>
            <span class="source-name">${esc(providerLabel(s.key))}</span>
            <span class="source-tags">
              ${s.isLive ? '<span class="tag tag-live">LIVE</span>' : ""}
              ${s.isMixed ? '<span class="tag tag-mixed">MIXED</span>' : ""}
              ${s.isSimulated ? '<span class="tag tag-sim">SIMULATED</span>' : ""}
            </span>
            <span class="source-num">${s.count}</span>
            <span class="source-meta">${s.freshestDays === null ? "no dates" : `freshest ${s.freshestDays}d`} · ${s.quality}% avg</span>
          </li>`).join("")}
      </ul>
    </div>`);
}

function renderChains() {
  const list = store.connections();
  const slot = $("chains-slot");
  if (!list.length) {
    setHTML(slot, cards.emptyState({
      icon: "🔗", title: "No cross-source connections found",
      body: "Connections need the same company or signal to appear in two different source types. Add competitors, or include more source types in the scan.",
    }));
    return;
  }
  setHTML(slot, `<div class="chain-grid">${list.map(cards.connectionChain).join("")}</div>`);
}

function renderLandscape() {
  const items = store.landscape();
  const slot = $("landscape-slot");
  if (items.length < 2) {
    setHTML(slot, cards.emptyState({ icon: "🗺", title: "Landscape needs more topics", body: "At least two recurring topics are required to map the space." }));
    return;
  }
  setHTML(slot, `<div class="bubble-wrap">${bubbles(items)}</div>
    <p class="chart-note">Pink = new this window · red/amber = fastest growing · grey = declining</p>`);
}

function renderPeople() {
  const list = store.contributors();
  const slot = $("people-slot");
  if (!list.length) {
    setHTML(slot, cards.emptyState({
      icon: "👥", title: "No attribution available",
      body: "The providers used in this scan did not return author or assignee data. Research and patent sources supply it.",
    }));
    return;
  }
  const anySim = list.some((c) => c.simulated || c.partlySimulated);
  setHTML(slot, `<ol class="people-list">
    ${list.map((c, i) => `
      <li>
        <span class="rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="person">
          <b>${esc(truncate(c.name, 40))}
            ${c.simulated ? '<span class="tag tag-sim">SIMULATED</span>' : ""}
            ${c.partlySimulated ? '<span class="tag tag-mixed">PARTLY SIMULATED</span>' : ""}</b>
          <em>${esc(c.institution || c.venue || (c.kind === "organisation" ? "patent assignee" : "researcher"))}</em>
        </span>
        <span class="person-stats">
          ${c.citations ? `<b>${c.citations.toLocaleString()}</b><span>citations</span>`
                        : `<b>${c.works}</b><span>finding${c.works === 1 ? "" : "s"}</span>`}
        </span>
      </li>`).join("")}
  </ol>
  ${anySim ? `<p class="chart-note">Entries marked SIMULATED come from providers with no API key
    configured and are not real attribution. Configure a Semantic Scholar key for live author data.</p>` : ""}`);
}

/* ── feed views ─────────────────────────────────────────── */
function renderFeedView(category) {
  const host = $(`${category === "patent" ? "patents" : category}-body`);
  const items = store.visibleFindings(category);
  const renderer = category === "patent" ? cards.patentRow : cards.feedRow;

  if (!items.length) {
    setHTML(host, filterBar(category) + cards.emptyState({
      icon: categoryIcon(category),
      title: `No ${categoryLabel(category).toLowerCase()} found`,
      body: "Try broader keywords, a longer time window, or a goal that points at this source type.",
    }));
    return;
  }

  setHTML(host, filterBar(category) + `
    <div class="feed">${items.map((f) => renderer(f, store.insightForFinding(f.id))).join("")}</div>`);
}

function renderNewsView() {
  const host = $("news-body");
  const items = [...store.visibleFindings("news"), ...store.visibleFindings("web")]
    .sort((a, b) => (b.relevance || 0) - (a.relevance || 0));

  if (!items.length) {
    setHTML(host, filterBar("news") + cards.emptyState({
      icon: "📰", title: "No industry news found",
      body: "The curated feeds and live web search returned nothing matching this goal in the window.",
    }));
    return;
  }
  setHTML(host, filterBar("news") + `
    <div class="feed">${items.map((f) => cards.feedRow(f, store.insightForFinding(f.id))).join("")}</div>`);
}

function renderCompetitors() {
  const host = $("competitors-body");
  const list = store.competitors();

  if (!list.length) {
    setHTML(host, cards.emptyState({
      icon: "🏢", title: "No competitors being tracked",
      body: "Add company names to the Competitors field and run a scan. The agent will search news, repos, forums and the live web for each one.",
      action: '<button type="button" class="btn-primary" data-scroll-search>Add competitors</button>',
    }));
    return;
  }

  const max = Math.max(...list.map((c) => c.total)) || 1;
  setHTML(host, `
    <section class="panel">
      <div class="panel-head"><div>
        <span class="eyebrow">Comparison</span><h2>Competitive Position</h2>
        <p class="panel-sub">Signal volume by source type, from this scan</p>
      </div></div>
      <div class="compare">
        ${list.map((c) => `
          <div class="compare-row">
            <span class="compare-name">${esc(c.name)}</span>
            <div class="compare-bars">
              ${["research", "patent", "news", "web", "competitor"].map((key) => {
                const n = c.byCategory[key] || 0;
                if (!n) return "";
                return `<span class="compare-seg accent-${(CATEGORY[key] || {}).accent}"
                  style="flex:${n}" title="${esc(categoryLabel(key))}: ${n}">${n}</span>`;
              }).join("") || '<span class="compare-seg empty">no signals</span>'}
            </div>
            <span class="compare-total">${c.total}</span>
          </div>`).join("")}
      </div>
      <div class="legend">
        ${["research", "patent", "news", "web", "competitor"].map((k) =>
          `<span><i class="sw accent-${(CATEGORY[k] || {}).accent}"></i>${esc(categoryLabel(k))}</span>`).join("")}
      </div>
    </section>

    <div class="comp-grid">${list.map(cards.competitorCard).join("")}</div>`);
  void max;
}

function renderInsights() {
  const host = $("insights-body");
  const items = store.visibleInsights();

  const bar = `
    <div class="filters">
      <div class="filter-row" role="group" aria-label="Filter by priority">
        ${[["all", "All"], ["HIGH", "🔴 High"], ["MEDIUM", "🟡 Medium"], ["LOW", "🟢 Low"]]
          .map(([v, l]) => `<button type="button" class="pill ${store.filters.priority === v ? "is-on" : ""}"
            data-filter="priority" data-value="${v}">${esc(l)}</button>`).join("")}
      </div>
      <div class="filter-row" role="group" aria-label="Filter by source">
        ${[["all", "All sources"], ...Object.keys(CATEGORY).map((k) => [k, categoryLabel(k)])]
          .map(([v, l]) => `<button type="button" class="pill ${store.filters.source === v ? "is-on" : ""}"
            data-filter="source" data-value="${v}">${esc(l)}</button>`).join("")}
      </div>
      ${activeFilterNote()}
    </div>`;

  if (!items.length) {
    setHTML(host, bar + cards.emptyState({
      icon: "🎯", title: "Nothing matches these filters",
      body: "Clear a filter, or run a broader scan.",
    }));
    return;
  }

  setHTML(host, bar + `<div class="feed">${items.map((i) => {
    const f = store.findingById(i.finding_id);
    return f ? cards.feedRow(f, i) : "";
  }).join("")}</div>`);
}

function renderReports() {
  const host = $("reports-body");
  const m = store.run.metrics || {};
  setHTML(host, `
    <section class="panel report-panel">
      <div class="panel-head"><div>
        <span class="eyebrow">Deliverable</span><h2>Intelligence Report</h2>
        <p class="panel-sub">A complete professional document built from this scan —
          ${store.insights().length} prioritized insight(s), the agent's execution trail,
          ${m.findings_total || 0} detailed finding(s), sources and caveats.</p>
      </div></div>

      <dl class="report-facts">
        <div><dt>Tracking goal</dt><dd>${esc(store.run.goal)}</dd></div>
        <div><dt>Run ID</dt><dd class="mono">${esc(store.run.run_id)}</dd></div>
        <div><dt>Tools used</dt><dd>${(m.tools_used || []).map((t) => esc(toolHuman(t))).join(", ") || "—"}</dd></div>
        <div><dt>Data</dt><dd>${m.simulated_data_used ? "includes clearly-labelled simulated findings" : "all findings from live sources"}</dd></div>
      </dl>

      <div class="report-actions">
        <button type="button" class="btn-ghost" data-report="preview">👁 Preview Report</button>
        <button type="button" class="btn-primary" data-report="pdf">📥 Download PDF</button>
        <button type="button" class="btn-ghost compact" data-report="md">Markdown</button>
        <button type="button" class="btn-ghost compact" data-report="json">JSON</button>
      </div>
      <p class="report-status" id="report-status" aria-live="polite"></p>
    </section>`);
}

function renderReportBar() {
  setHTML($("report-bar"), `
    <span class="rb-text">📄 Intelligence report ready for this scan</span>
    <span class="rb-actions">
      <button type="button" class="btn-ghost compact" data-report="preview">Preview</button>
      <button type="button" class="btn-primary compact" data-report="pdf">Download PDF</button>
    </span>`);
  show($("report-bar"), true);
}

function renderTrail() {
  // Multi-agent execution first — it is the headline proof of orchestration.
  setHTML($("agents-body"), renderMultiAgent(store.run));
  // Context & memory sits with the orchestration proof: it is the same story seen
  // from the other side — what was retained and shared, rather than who ran.
  setHTML($("memory-body"), renderMemory(store.run));
  setHTML($("trail-body"), agentUI.reasoningTrail(store.run) + `
    <details class="reveal tech">
      <summary>Show technical details</summary>
      <div class="reveal-body">${agentUI.technicalDetails(store.run)}</div>
    </details>`);
  show($("trail"), true);
  $("trail").open = false;
}

/* ── filter helpers ─────────────────────────────────────── */
function filterBar(category) {
  const note = activeFilterNote();
  return note ? `<div class="filters">${note}</div>` : "";
}

function activeFilterNote() {
  const bits = [];
  if (store.filters.topic) {
    bits.push(`<span class="chip-filter">Topic: <b>${esc(store.topicLabel(store.filters.topic))}</b>
      <button type="button" data-clear="topic" aria-label="Clear topic filter">×</button></span>`);
  }
  if (store.filters.competitor) {
    bits.push(`<span class="chip-filter">Company: <b>${esc(store.filters.competitor)}</b>
      <button type="button" data-clear="competitor" aria-label="Clear company filter">×</button></span>`);
  }
  return bits.length ? `<div class="active-filters">${bits.join("")}</div>` : "";
}

function invalidateFeeds() {
  ["overview", "research", "patents", "news", "insights", "competitors"].forEach((v) => rendered.delete(v));
  renderView(store.view);
}

/* ═══════════════════════════════════════════════════════════
   REPORTS
   ═══════════════════════════════════════════════════════════ */
function reportStatus(text, kind = "") {
  const el = $("report-status");
  if (el) { setText(el, text); el.className = `report-status ${kind}`; }
  const rb = $("report-bar");
  if (rb) rb.dataset.status = kind;
}

async function ensureReport({ force = false } = {}) {
  if (!store.run) throw new Error("run a scan first");
  if (store.report && !force) return store.report;
  reportStatus(force ? "Regenerating…" : "Building report…", "busy");
  const data = await api.generateReport(store.run.run_id, { force });
  store.report = data.report;
  reportStatus(`Report ${data.report.report_id} ready${data.cached ? " (cached)" : ""}.`, "ok");
  return store.report;
}

async function handleReport(action) {
  try {
    if (action === "preview") return openPreview();
    const report = await ensureReport();
    reportStatus(`Preparing ${action.toUpperCase()}…`, "busy");
    await api.downloadReport(report.report_id, action);
    reportStatus(`${action.toUpperCase()} downloaded.`, "ok");
  } catch (err) {
    reportStatus(`Could not complete: ${err.message}`, "err");
  }
}

async function openPreview({ force = false } = {}) {
  show($("preview"), true);
  show($("preview-frame"), false);
  show($("preview-loading"), true);
  setText($("preview-loading"), force ? "Regenerating your report…" : "Building your report…");
  document.body.classList.add("no-scroll");
  try {
    const report = await ensureReport({ force });
    setText($("preview-id"), report.report_id);
    const frame = $("preview-frame");
    frame.src = api.reportPreviewUrl(report.report_id);
    frame.onload = () => { show($("preview-loading"), false); show(frame, true); };
  } catch (err) {
    setText($("preview-loading"), `Could not build the report: ${err.message}`);
  }
}

function closePreview() {
  show($("preview"), false);
  $("preview-frame").removeAttribute("src");
  document.body.classList.remove("no-scroll");
}

/* ═══════════════════════════════════════════════════════════
   SYSTEM INFO (technical detail lives here, not the navbar)
   ═══════════════════════════════════════════════════════════ */
function renderSystemInfo() {
  const t = store.tools;
  if (!t) return;
  const caps = t.capabilities || {};
  const llm = caps.llm || {};
  const keyed = caps.keyed_sources || {};

  setHTML($("sysinfo"), `
    <div class="sys-group">
      <h3>Reasoning engine</h3>
      <div class="sys-row"><span>Mode</span><span class="${llm.live ? "ok" : "warn"}">
        ${llm.live ? "AI model" : "built-in deterministic reasoner"}</span></div>
      <div class="sys-row"><span>Provider</span><span>${esc(llm.provider || "—")}</span></div>
      <div class="sys-row"><span>Model</span><span class="mono">${esc(llm.model || "—")}</span></div>
      ${!llm.live ? `<p class="muted small">No AI credential is active, so planning, tool
        selection and prioritization run on the deterministic reasoner. The agent still
        reasons — it just says so honestly.</p>` : ""}
    </div>

    <div class="sys-group">
      <h3>Agent tools (${(t.usable || []).length})</h3>
      ${(t.tools || []).map((tool) => `
        <div class="sys-tool">
          <b>${esc(tool.display_name)}</b>
          <p>${esc(tool.when_to_use)}</p>
          <div class="feed-tags">
            ${(tool.providers_live || []).map((p) => `<span class="tag tag-live">${esc(providerLabel(p))}</span>`).join("")}
            ${(tool.providers_simulated || []).map((p) => `<span class="tag tag-sim">${esc(providerLabel(p))}</span>`).join("")}
          </div>
        </div>`).join("")}
    </div>

    <div class="sys-group">
      <h3>Source credentials</h3>
      ${Object.entries(keyed).map(([k, v]) => `
        <div class="sys-row"><span>${esc(k.replace(/_/g, " "))}</span>
        <span class="${v ? "ok" : "warn"}">${v ? "configured" : "not set — simulated"}</span></div>`).join("")}
      <div class="sys-row"><span>simulation mode</span><span>${t.simulation_mode ? "on" : "off"}</span></div>
    </div>`);
}

/* ═══════════════════════════════════════════════════════════
   CHIPS
   ═══════════════════════════════════════════════════════════ */
function chipField(boxId, chipsId, inputId) {
  const values = [];
  const chipsEl = $(chipsId);
  const inputEl = $(inputId);

  const render = () => setHTML(chipsEl, values.map((v, i) =>
    `<span class="chip">${esc(v)}<button type="button" data-i="${i}" aria-label="Remove ${esc(v)}">×</button></span>`
  ).join(""));

  const add = (raw) => {
    String(raw).split(",").map((s) => s.trim()).filter(Boolean).forEach((v) => {
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

  return { get: () => [...values], set: (list) => { values.length = 0; list.forEach((v) => v && values.push(v)); render(); }, add };
}

const chips = {};

/* ═══════════════════════════════════════════════════════════
   EVENTS
   ═══════════════════════════════════════════════════════════ */
function wireEvents() {
  chips.keywords = chipField("kw-box", "kw-chips", "kw-input");
  chips.competitors = chipField("comp-box", "comp-chips", "comp-input");

  $("search-form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("stop-btn").addEventListener("click", () => controller && controller.abort());
  $("goal").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
  });

  // presets
  delegate($("presets"), "[data-preset]", "click", (e, btn) => {
    $("goal").value = btn.dataset.goal || "";
    chips.keywords.set((btn.dataset.kw || "").split(",").map((s) => s.trim()).filter(Boolean));
    chips.competitors.set((btn.dataset.comp || "").split(",").map((s) => s.trim()).filter(Boolean));
    $("goal").focus();
  });

  // nav
  delegate(document, "[data-nav]", "click", (e, btn) => switchView(btn.dataset.nav));

  // live log toggle
  $("ticks-toggle").addEventListener("click", () => {
    const wrap = $("ticks-wrap");
    const open = wrap.hidden;
    show(wrap, open);
    $("ticks-toggle").setAttribute("aria-expanded", String(open));
    setText($("ticks-toggle"), open ? "Hide live agent activity" : "View live agent activity");
    if (open) $("ticks").scrollTop = $("ticks").scrollHeight;
  });

  // one delegated handler for the whole document (§27)
  document.addEventListener("click", (e) => {
    const t = e.target;

    const detail = t.closest("[data-detail]");
    if (detail) { openDrawer(detail.dataset.detail, store); return; }

    const filter = t.closest("[data-filter]");
    if (filter) {
      store.filters[filter.dataset.filter] = filter.dataset.value;
      rendered.delete("insights");
      renderView("insights");
      return;
    }

    const topic = t.closest("[data-topic]");
    if (topic) {
      const key = topic.dataset.topic;
      const clearing = store.filters.topic === key;
      store.filters.topic = clearing ? null : key;
      invalidateFeeds();
      // Setting a filter must show its effect. Staying on Overview would apply an
      // invisible filter, so jump to the feed it actually filters.
      if (!clearing) switchView("insights");
      return;
    }

    const comp = t.closest("[data-competitor-filter]");
    if (comp) {
      store.filters.competitor = comp.dataset.competitorFilter;
      rendered.delete("insights");
      switchView("insights");
      return;
    }

    const clear = t.closest("[data-clear]");
    if (clear) { store.filters[clear.dataset.clear] = null; invalidateFeeds(); return; }

    const chain = t.closest("[data-chain-explore]");
    if (chain) {
      const found = store.connections().find((c) => c.key === chain.dataset.chainExplore);
      if (found) openDrawer(found.members[0].id, store);
      return;
    }

    const win = t.closest("[data-window]");
    if (win) {
      store.chartWindow = Number(win.dataset.window);
      qsa("[data-window]").forEach((b) => b.classList.toggle("is-on", b === win));
      renderChart();
      return;
    }

    const report = t.closest("[data-report]");
    if (report) { handleReport(report.dataset.report); return; }

    const tech = t.closest(".tech-tab");
    if (tech) {
      qsa(".tech-tab").forEach((b) => b.classList.toggle("is-on", b === tech));
      qsa("[data-tech-panel]").forEach((p) => { p.hidden = p.dataset.techPanel !== tech.dataset.tech; });
      return;
    }

    const save = t.closest("[data-save]");
    if (save) {
      const on = store.toggleSaved(save.dataset.save);
      setText(save, on ? "★ Saved" : "☆ Save");
      return;
    }

    const track = t.closest("[data-track]");
    if (track) {
      const f = store.findingById(track.dataset.track);
      const term = f?.competitor || (f?.title || "").split(" ").slice(0, 3).join(" ");
      if (!term) return;
      store.addTracked(term);
      if (f?.competitor) chips.competitors.add(f.competitor); else chips.keywords.add(term);
      setText(track, "✓ Added to tracking");
      return;
    }

    if (t.closest("[data-scroll-search]")) {
      $("search-panel").scrollIntoView({ behavior: "smooth", block: "center" });
      $("goal").focus();
    }
  });

  // bubbles are SVG <g> — keyboard support
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      const bubble = e.target.closest?.("[data-topic]");
      if (bubble && bubble.tagName.toLowerCase() === "g") { e.preventDefault(); bubble.dispatchEvent(new Event("click", { bubbles: true })); }
    }
    if (e.key !== "Escape") return;
    if (!$("preview").hidden) closePreview();
    else if (!$("drawer").hidden) closeDrawer();
    else if (!$("sysinfo-panel").hidden) toggleSysinfo(false);
  });

  // drawer / preview / settings
  $("drawer-close").addEventListener("click", closeDrawer);
  $("drawer-scrim").addEventListener("click", closeDrawer);
  $("preview-back").addEventListener("click", closePreview);
  $("preview-download").addEventListener("click", () => handleReport("pdf"));
  $("preview-regen").addEventListener("click", () => openPreview({ force: true }));
  $("settings-btn").addEventListener("click", () => toggleSysinfo());
  $("sysinfo-close").addEventListener("click", () => toggleSysinfo(false));
  $("sysinfo-scrim").addEventListener("click", () => toggleSysinfo(false));
}

function toggleSysinfo(force) {
  const panel = $("sysinfo-panel");
  const open = force === undefined ? panel.hidden : force;
  show(panel, open);
  show($("sysinfo-scrim"), open);
  $("settings-btn").setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("no-scroll", open);
}

boot();
