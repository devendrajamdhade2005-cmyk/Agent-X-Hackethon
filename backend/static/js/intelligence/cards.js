/* Reusable renderers for every intelligence surface.
   Pure functions: state in, HTML string out. Cheap to call, easy to test. */

import { esc } from "../core/dom.js";
import {
  CATEGORY, PRIORITY, SIGNAL_LABEL, ago, categoryIcon, categoryLabel,
  categoryOf, formatDate, providerLabel, relevanceBand, shortenUrl, truncate,
} from "../core/format.js";

/* ── shared atoms ───────────────────────────────────────── */
export function priorityBadge(priority) {
  const p = PRIORITY[priority] || PRIORITY.MEDIUM;
  return `<span class="badge tone-${p.tone}"><i aria-hidden="true">${p.dot}</i>${esc(p.label)}</span>`;
}

export function relevanceBadge(score) {
  const b = relevanceBand(score);
  return `<span class="badge tone-${b.tone}"><i aria-hidden="true">${b.dot}</i>${esc(b.label)}</span>`;
}

export function categoryTag(key) {
  const c = CATEGORY[key] || { label: key, accent: "slate" };
  return `<span class="tag accent-${c.accent}">${categoryIcon(key)} ${esc(c.label)}</span>`;
}

export const simTag = (on) =>
  on ? '<span class="tag tag-sim" title="Generated in simulation mode — not verified real-world data">SIMULATED</span>' : "";

export const signalTags = (signals = []) =>
  signals.slice(0, 3)
    .map((s) => `<span class="tag tag-signal">${esc(SIGNAL_LABEL[s] || s)}</span>`)
    .join("");

export function metricPill(label, value) {
  return `<span class="metric-pill"><b>${esc(value)}</b><span>${esc(label)}</span></span>`;
}

/* ── executive summary ──────────────────────────────────── */
export function executiveSummary({ counts, trend, nextStep, findings, fullSummary, simulated }) {
  const rows = [
    ["HIGH", counts.HIGH, "require immediate attention", "requires immediate attention"],
    ["MEDIUM", counts.MEDIUM, "worth monitoring", "worth monitoring"],
    ["LOW", counts.LOW, "low priority", "low priority"],
  ].filter(([, n]) => n > 0);

  return `
  <div class="panel-head">
    <div>
      <span class="eyebrow">Today's Intelligence</span>
      <h2>${counts.total} relevant development${counts.total === 1 ? "" : "s"} found</h2>
    </div>
    ${simulated ? simTag(true) : ""}
  </div>
  <ul class="prio-list">
    ${rows.map(([k, n, plural, singular]) => {
      const p = PRIORITY[k];
      return `<li class="tone-${p.tone}"><i aria-hidden="true">${p.dot}</i>
        <b>${n}</b> ${esc(n === 1 ? singular : plural)}</li>`;
    }).join("") || '<li class="muted">No insights cleared the relevance bar.</li>'}
  </ul>
  <div class="summary-grid">
    ${trend ? `<div class="mini-block accent-yellow">
      <span class="mini-label">Main trend</span><p>${esc(trend)}</p></div>` : ""}
    ${nextStep ? `<div class="mini-block accent-purple">
      <span class="mini-label">Recommended next step</span><p>${esc(nextStep)}</p></div>` : ""}
  </div>
  <p class="summary-meta">${findings} item(s) collected across all sources.</p>
  ${fullSummary ? `<details class="reveal">
    <summary>Read the full analyst summary</summary>
    <div class="reveal-body">${fullSummary.split("\n\n").map((p) => `<p>${esc(p)}</p>`).join("")}</div>
  </details>` : ""}`;
}

/* ── top intelligence (hero insight) ────────────────────── */
export function topInsight(insight) {
  if (!insight) return "";
  const cat = categoryOf(insight.source);
  const p = PRIORITY[insight.priority] || PRIORITY.MEDIUM;

  return `
  <div class="hero-top">
    <span class="eyebrow">Top Intelligence</span>
    <div class="hero-badges">
      ${priorityBadge(insight.priority)}
      ${categoryTag(cat)}
      ${insight.competitor ? `<span class="tag accent-orange">${esc(insight.competitor)}</span>` : ""}
      ${simTag(insight.simulated)}
    </div>
  </div>
  <h2 class="hero-title">${esc(insight.title)}</h2>
  <div class="hero-fields">
    <div><span class="mini-label">What happened</span><p>${esc(insight.what_happened)}</p></div>
    <div class="accent-pink"><span class="mini-label">Why it matters</span><p>${esc(insight.why_it_matters)}</p></div>
    <div class="accent-purple"><span class="mini-label">Recommended action</span><p>${esc(insight.recommended_action)}</p></div>
  </div>
  <div class="hero-foot">
    <span class="hero-src">${esc(insight.source)} · ${esc(formatDate(insight.published_date))}</span>
    <span class="hero-actions">
      <button type="button" class="btn-mini" data-detail="${esc(insight.finding_id || insight.id)}">View evidence</button>
      ${insight.source_url ? `<a class="btn-mini" href="${esc(insight.source_url)}" target="_blank" rel="noopener">View source ↗</a>` : ""}
    </span>
  </div>`;
}

/* ── intelligence feed row (research / news / web) ───────── */
export function feedRow(finding, insight) {
  const band = relevanceBand(finding.relevance);
  const meta = finding.meta || {};
  const cat = finding.source;

  const stats = [];
  if (finding.relevance != null) stats.push(metricPill("relevance", `${Math.round(finding.relevance * 100)}%`));
  if (meta.citation_count) stats.push(metricPill("citations", meta.citation_count));
  if (meta.stars) stats.push(metricPill("stars", meta.stars));
  if (meta.points) stats.push(metricPill("points", meta.points));
  if (meta.tavily_score) stats.push(metricPill("rank", meta.tavily_score.toFixed(2)));

  const byline = [
    finding.author ? truncate(finding.author, 68) : "",
    meta.venue || meta.outlet || providerLabel(finding.provider),
    finding.published_date ? ago(finding.published_date) : "",
  ].filter(Boolean).join(" · ");

  return `
  <article class="feed-row accent-${(CATEGORY[cat] || {}).accent || "slate"}" data-id="${esc(finding.id)}">
    <div class="feed-rail">
      <span class="feed-icon">${categoryIcon(cat)}</span>
      <span class="feed-dot tone-${band.tone}" title="${esc(band.label)}"></span>
    </div>

    <div class="feed-body">
      <div class="feed-tags">
        ${relevanceBadge(finding.relevance)}
        ${categoryTag(cat)}
        ${finding.competitor ? `<span class="tag accent-orange">${esc(finding.competitor)}</span>` : ""}
        ${signalTags(finding.signals)}
        ${simTag(finding.simulated)}
      </div>

      <h3 class="feed-title">${esc(finding.title)}</h3>
      ${byline ? `<p class="feed-byline">${esc(byline)}</p>` : ""}
      ${finding.summary ? `<p class="feed-excerpt">${esc(truncate(finding.summary, 260))}</p>` : ""}

      ${insight ? `
      <div class="feed-insight">
        <div><span class="mini-label">Why it matters</span><p>${esc(truncate(insight.why_it_matters, 200))}</p></div>
        <div><span class="mini-label">Recommended action</span><p>${esc(truncate(insight.recommended_action, 160))}</p></div>
      </div>` : ""}

      <div class="feed-actions">
        <button type="button" class="btn-mini" data-detail="${esc(finding.id)}">Evidence</button>
        ${finding.url ? `<a class="btn-mini" href="${esc(finding.url)}" target="_blank" rel="noopener">
          ${cat === "research" ? "Read paper" : cat === "patent" ? "Open filing" : "Open source"} ↗</a>` : ""}
        <span class="feed-provider">${esc(providerLabel(finding.provider))}</span>
      </div>
    </div>

    <div class="feed-stats">${stats.join("")}</div>
  </article>`;
}

/* ── patent row (assignee-forward) ──────────────────────── */
export function patentRow(finding, insight) {
  const meta = finding.meta || {};
  const assignee = meta.assignee || finding.author || finding.competitor || "";
  const isCompetitor = Boolean(finding.competitor);

  return `
  <article class="feed-row accent-cyan ${isCompetitor ? "is-flagged" : ""}" data-id="${esc(finding.id)}">
    <div class="feed-rail">
      <span class="feed-icon">📜</span>
      <span class="feed-dot tone-${isCompetitor ? "red" : "blue"}"></span>
    </div>
    <div class="feed-body">
      <div class="feed-tags">
        ${isCompetitor
          ? `<span class="badge tone-red"><i aria-hidden="true">🔴</i>Competitor-owned IP</span>`
          : relevanceBadge(finding.relevance)}
        ${categoryTag("patent")}
        ${meta.patent_number ? `<span class="tag">${esc(meta.patent_number)}</span>` : ""}
        ${simTag(finding.simulated)}
      </div>
      <h3 class="feed-title">${esc(finding.title)}</h3>
      <p class="feed-byline">
        ${assignee ? `Assignee: <b>${esc(truncate(assignee, 48))}</b>` : "Assignee unknown"}
        ${meta.filing_date ? ` · filed ${esc(formatDate(meta.filing_date))}` : ""}
        ${finding.published_date ? ` · published ${esc(formatDate(finding.published_date))}` : ""}
        ${meta.cpc ? ` · CPC ${esc(meta.cpc)}` : ""}
      </p>
      ${finding.summary ? `<p class="feed-excerpt">${esc(truncate(finding.summary, 240))}</p>` : ""}
      ${insight ? `
      <div class="feed-insight">
        <div><span class="mini-label">Strategic significance</span><p>${esc(truncate(insight.why_it_matters, 200))}</p></div>
        <div><span class="mini-label">Recommended action</span><p>${esc(truncate(insight.recommended_action, 160))}</p></div>
      </div>` : ""}
      <div class="feed-actions">
        <button type="button" class="btn-mini" data-detail="${esc(finding.id)}">Evidence</button>
        ${finding.url ? `<a class="btn-mini" href="${esc(finding.url)}" target="_blank" rel="noopener">Open filing ↗</a>` : ""}
        <span class="feed-provider">${esc(providerLabel(finding.provider))}</span>
      </div>
    </div>
    <div class="feed-stats">
      ${finding.relevance != null ? metricPill("relevance", `${Math.round(finding.relevance * 100)}%`) : ""}
    </div>
  </article>`;
}

/* ── competitor card ────────────────────────────────────── */
export function competitorCard(c) {
  const trendArrow = (n) => (n >= 3 ? "↑" : n > 0 ? "→" : "·");
  const p = PRIORITY[c.priority] || PRIORITY.MEDIUM;
  const rows = ["research", "patent", "news", "web"].map((key) => {
    const n = c.byCategory[key] || 0;
    return `<div class="comp-metric">
      <span>${categoryIcon(key)} ${esc(categoryLabel(key))}</span>
      <b>${n} <i class="arrow">${trendArrow(n)}</i></b>
    </div>`;
  }).join("");

  return `
  <article class="comp-card" data-competitor="${esc(c.name)}">
    <div class="comp-head">
      <div>
        <h3>${esc(c.name)}</h3>
        <p class="comp-sub">${c.total} signal${c.total === 1 ? "" : "s"} detected</p>
      </div>
      <span class="badge tone-${p.tone}"><i aria-hidden="true">${p.dot}</i>${esc(p.label)}</span>
    </div>

    <div class="comp-metrics">${rows}</div>

    ${c.signals.length ? `<div class="feed-tags">${signalTags(c.signals)}</div>` : ""}

    ${c.latest ? `<div class="comp-latest">
      <span class="mini-label">Latest development</span>
      <p>${esc(truncate(c.latest.title, 130))}</p>
      <span class="comp-date">${esc(ago(c.latest.published_date))} · ${esc(providerLabel(c.latest.provider))}</span>
    </div>` : `<p class="muted small">No dated development found in this window.</p>`}

    ${c.topInsight ? `
    <div class="comp-insight">
      <div><span class="mini-label">Why it matters</span><p>${esc(truncate(c.topInsight.why_it_matters, 170))}</p></div>
      <div><span class="mini-label">Recommended action</span><p>${esc(truncate(c.topInsight.recommended_action, 150))}</p></div>
    </div>` : ""}

    <div class="comp-foot">
      <button type="button" class="btn-mini" data-competitor-filter="${esc(c.name)}">View all signals</button>
      ${c.simulated ? simTag(true) : ""}
    </div>
  </article>`;
}

/* ── connected intelligence chain ───────────────────────── */
export function connectionChain(chain) {
  const steps = chain.members.map((m, i) => `
    ${i ? '<span class="chain-arrow" aria-hidden="true">↓</span>' : ""}
    <button type="button" class="chain-step accent-${(CATEGORY[m.source] || {}).accent || "slate"}"
            data-detail="${esc(m.id)}">
      <span class="chain-icon">${categoryIcon(m.source)}</span>
      <span class="chain-text">
        <b>${esc(categoryLabel(m.source))}</b>
        <span>${esc(truncate(m.title, 74))}</span>
      </span>
    </button>`).join("");

  return `
  <article class="chain-card">
    <div class="chain-head">
      <div>
        <span class="eyebrow">${chain.members.length} related signals detected</span>
        <h3>${esc(chain.anchor)}</h3>
        <p class="muted small">Linked by shared ${chain.anchorKind === "company" ? "company" : "strategic signal"}</p>
      </div>
      <span class="confidence" title="Derived link strength, not a source-provided figure">
        <b>${chain.confidence}%</b><span>confidence</span>
      </span>
    </div>
    <div class="chain-flow">${steps}</div>
    <div class="chain-foot">
      <button type="button" class="btn-mini" data-chain-explore="${esc(chain.key)}">Explore connection</button>
      ${chain.simulated ? simTag(true) : ""}
    </div>
  </article>`;
}

/* ── empty state ────────────────────────────────────────── */
export function emptyState({ icon = "🔬", title, body, action = "" }) {
  return `<div class="empty">
    <span class="empty-icon" aria-hidden="true">${icon}</span>
    <h3>${esc(title)}</h3>
    <p>${esc(body)}</p>
    ${action}
  </div>`;
}

/* ── skeleton loaders ──────────────────────────────────── */
export function skeletonRows(n = 3) {
  return Array.from({ length: n }, () => `
    <div class="skel-row">
      <div class="skel skel-rail"></div>
      <div class="skel-lines">
        <div class="skel skel-line w70"></div>
        <div class="skel skel-line w40"></div>
        <div class="skel skel-line w90"></div>
      </div>
    </div>`).join("");
}
