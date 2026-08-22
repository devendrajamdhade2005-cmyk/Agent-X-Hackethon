/* Right-side evidence drawer.
   Opens over the current view so the user never loses their place (§23). */

import { $, esc, setHTML } from "../core/dom.js";
import {
  categoryIcon, categoryLabel, categoryOf, formatDate, providerLabel,
  relevanceBand, truncate,
} from "../core/format.js";
import { categoryTag, priorityBadge, relevanceBadge, signalTags, simTag } from "../intelligence/cards.js";

let lastFocused = null;

export function openDrawer(findingId, store) {
  const finding = store.findingById(findingId);
  if (!finding) return;

  const insight = store.insightForFinding(findingId);
  const related = store.relatedTo(finding);
  const meta = finding.meta || {};
  const band = relevanceBand(finding.relevance);

  const facts = [
    ["Relevance", finding.relevance != null ? `${Math.round(finding.relevance * 100)}% (${band.label})` : null],
    ["Category", categoryLabel(finding.source)],
    ["Published", finding.published_date ? formatDate(finding.published_date) : null],
    ["Provider", providerLabel(finding.provider)],
    ["Source quality", finding.credibility],
    ["Citations", meta.citation_count || null],
    ["Venue", meta.venue || null],
    ["Institution", (meta.institutions || [])[0] || meta.institution || null],
    ["Assignee", meta.assignee || null],
    ["Patent number", meta.patent_number || null],
    ["Filed", meta.filing_date ? formatDate(meta.filing_date) : null],
    ["Technology class", meta.cpc || null],
    ["Language", meta.language || null],
    ["Stars", meta.stars || null],
    ["Outlet", meta.outlet || null],
    ["Search rank", meta.tavily_score || null],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");

  const techTags = [...(meta.concepts || []), ...(meta.categories || [])].slice(0, 8);

  setHTML($("drawer-body"), `
    <div class="drawer-tags">
      ${insight ? priorityBadge(insight.priority) : relevanceBadge(finding.relevance)}
      ${categoryTag(finding.source)}
      ${finding.competitor ? `<span class="tag accent-orange">${esc(finding.competitor)}</span>` : ""}
      ${signalTags(finding.signals)}
      ${simTag(finding.simulated)}
    </div>

    <h2 class="drawer-title">${esc(finding.title)}</h2>
    ${finding.author ? `<p class="drawer-authors">${esc(truncate(finding.author, 220))}</p>` : ""}

    ${finding.summary ? `<section class="drawer-sec">
      <span class="mini-label">Abstract / excerpt</span>
      <p>${esc(finding.summary)}</p></section>` : ""}

    ${insight ? `
    <section class="drawer-sec accent-purple">
      <span class="mini-label">AI summary</span>
      <p>${esc(insight.summary || insight.what_happened)}</p>
    </section>
    <section class="drawer-sec accent-pink">
      <span class="mini-label">Why it matters</span>
      <p>${esc(insight.why_it_matters)}</p>
    </section>
    <section class="drawer-sec accent-cyan">
      <span class="mini-label">Recommended action</span>
      <p>${esc(insight.recommended_action)}</p>
    </section>
    <p class="drawer-attr">Analysis written by ${esc(insight.author || "the agent")}.</p>` : `
    <section class="drawer-sec">
      <p class="muted small">This finding was collected and scored but did not make the
      prioritized briefing, so it has no written analysis.</p>
    </section>`}

    ${techTags.length ? `<section class="drawer-sec">
      <span class="mini-label">Related technologies</span>
      <div class="feed-tags">${techTags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
    </section>` : ""}

    <section class="drawer-sec">
      <span class="mini-label">Evidence &amp; metadata</span>
      <dl class="drawer-facts">
        ${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}
      </dl>
    </section>

    ${related.length ? `<section class="drawer-sec">
      <span class="mini-label">Related signals (${related.length})</span>
      <ul class="drawer-related">
        ${related.map((r) => `<li>
          <button type="button" data-detail="${esc(r.id)}">
            <span aria-hidden="true">${categoryIcon(r.source)}</span>
            <span><b>${esc(truncate(r.title, 78))}</b>
            <em>${esc(categoryLabel(r.source))} · ${esc(providerLabel(r.provider))}</em></span>
          </button></li>`).join("")}
      </ul>
    </section>` : ""}

    <div class="drawer-actions">
      ${finding.url ? `<a class="btn-primary compact" href="${esc(finding.url)}"
        target="_blank" rel="noopener">Open original ↗</a>` : ""}
      <button type="button" class="btn-ghost compact" data-save="${esc(finding.id)}">
        ${store.isSaved(finding.id) ? "★ Saved" : "☆ Save"}</button>
      <button type="button" class="btn-ghost compact" data-track="${esc(finding.id)}">
        + Add to tracking</button>
    </div>`);

  lastFocused = document.activeElement;
  $("drawer").hidden = false;
  $("drawer-scrim").hidden = false;
  document.body.classList.add("no-scroll");
  $("drawer-close").focus();
}

export function closeDrawer() {
  $("drawer").hidden = true;
  $("drawer-scrim").hidden = true;
  document.body.classList.remove("no-scroll");
  if (lastFocused && lastFocused.isConnected) lastFocused.focus();
}
