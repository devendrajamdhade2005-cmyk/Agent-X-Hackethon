/* Client state + memoized selectors.
 *
 * One run in memory, derived views computed once per run and cached. Switching
 * nav sections or applying a filter never refetches and never re-derives (§27).
 */

import * as derive from "../analytics/derive.js";
import { categoryOf } from "./format.js";

const CACHE = new Map();

export const store = {
  run: null,
  previousRun: null,
  tools: null,
  report: null,
  view: "overview",
  filters: { priority: "all", source: "all", topic: null, competitor: null },
  saved: new Set(JSON.parse(localStorage.getItem("ip.saved") || "[]")),
  tracked: new Set(JSON.parse(localStorage.getItem("ip.tracked") || "[]")),
  chartWindow: 30,

  setRun(result) {
    if (this.run && this.run.run_id !== result.run_id) this.previousRun = this.run;
    this.run = result;
    this.report = null;
    this.filters = { priority: "all", source: "all", topic: null, competitor: null };
    CACHE.clear();
    // Index findings once for O(1) lookups from the drawer and chains.
    this._byId = new Map((result.findings || []).map((f) => [f.id, f]));
    this._insightByFinding = new Map(
      (result.insights || []).map((i) => [i.finding_id, i])
    );
  },

  hasRun() {
    return Boolean(this.run && (this.run.findings || []).length >= 0);
  },

  /* ── memoized derivations ─────────────────────────────── */
  _memo(key, fn) {
    if (!CACHE.has(key)) CACHE.set(key, fn());
    return CACHE.get(key);
  },

  kpis()         { return this._memo("kpis", () => derive.kpis(this.run, this.previousRun)); },
  topics()       { return this._memo("topics", () => derive.topics(this.run)); },
  landscape()    { return this._memo("landscape", () => derive.landscape(this.run)); },
  sources()      { return this._memo("sources", () => derive.sources(this.run)); },
  competitors()  { return this._memo("competitors", () => derive.competitors(this.run)); },
  contributors() { return this._memo("contributors", () => derive.contributors(this.run)); },
  connections()  { return this._memo("connections", () => derive.connections(this.run)); },
  counts()       { return this._memo("counts", () => derive.summaryCounts(this.run)); },
  trend()        { return this._memo("trend", () => derive.mainTrend(this.run)); },

  activity(windowDays = this.chartWindow) {
    return this._memo(`activity:${windowDays}`, () => derive.activitySeries(this.run, windowDays));
  },

  /* ── lookups ──────────────────────────────────────────── */
  findingById(id) { return this._byId ? this._byId.get(id) : null; },
  insightForFinding(id) { return this._insightByFinding ? this._insightByFinding.get(id) : null; },

  insights() { return this.run?.insights || []; },
  findings() { return this.run?.findings || []; },

  findingsByCategory(category) {
    return this._memo(`cat:${category}`, () =>
      (this.run?.findings || []).filter((f) => f.source === category)
        .sort((a, b) => (b.relevance || 0) - (a.relevance || 0)));
  },

  /** Findings that share a company or a signal with the given one. */
  relatedTo(finding, limit = 5) {
    const all = this.run?.findings || [];
    const signals = new Set(finding.signals || []);
    const company = (finding.competitor || "").toLowerCase();
    return all
      .filter((f) => {
        if (f.id === finding.id) return false;
        if (company && (f.competitor || "").toLowerCase() === company) return true;
        return (f.signals || []).some((s) => signals.has(s));
      })
      .sort((a, b) => (b.relevance || 0) - (a.relevance || 0))
      .slice(0, limit);
  },

  /** Client-side filtering — instant, no network. */
  visibleInsights() {
    const { priority, source, topic, competitor } = this.filters;
    const topicItems = topic
      ? new Set((this.topics().find((t) => t.key === topic) || {}).items || [])
      : null;

    return this.insights().filter((i) => {
      if (priority !== "all" && i.priority !== priority) return false;
      if (source !== "all" && categoryOf(i.source) !== source) return false;
      if (competitor && (i.competitor || "").toLowerCase() !== competitor.toLowerCase()) return false;
      if (topicItems && !topicItems.has(i.finding_id)) return false;
      return true;
    });
  },

  visibleFindings(category = null) {
    const { topic, competitor } = this.filters;
    const topicItems = topic
      ? new Set((this.topics().find((t) => t.key === topic) || {}).items || [])
      : null;

    return this.findings()
      .filter((f) => {
        if (category && f.source !== category) return false;
        if (competitor && !matchesCompetitor(f, competitor)) return false;
        if (topicItems && !topicItems.has(f.id)) return false;
        return true;
      })
      .sort((a, b) => (b.relevance || 0) - (a.relevance || 0));
  },

  topicLabel(key) {
    return (this.topics().find((t) => t.key === key) || {}).label || "";
  },

  /* ── saved / tracked (local only, clearly user-scoped) ── */
  isSaved(id) { return this.saved.has(id); },
  toggleSaved(id) {
    this.saved.has(id) ? this.saved.delete(id) : this.saved.add(id);
    localStorage.setItem("ip.saved", JSON.stringify([...this.saved]));
    return this.saved.has(id);
  },
  isTracked(term) { return this.tracked.has(term); },
  addTracked(term) {
    this.tracked.add(term);
    localStorage.setItem("ip.tracked", JSON.stringify([...this.tracked]));
  },
};

function matchesCompetitor(f, name) {
  const needle = name.toLowerCase();
  if ((f.competitor || "").toLowerCase() === needle) return true;
  return `${f.title} ${f.summary}`.toLowerCase().includes(needle);
}
