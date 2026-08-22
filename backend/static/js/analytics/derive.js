/* Derived analytics.
 *
 * Hard rule (§29): every number here is computed from the actual agent run.
 * Nothing is hardcoded and nothing is invented. Where a metric cannot be derived
 * from the available data, these functions return null and the UI renders a
 * limited-data state instead of a fabricated figure.
 */

import { CATEGORY, SIGNAL_LABEL, relativeDays } from "../core/format.js";

const HIGH_RELEVANCE = 0.7;
const EMERGING_MIN = 0.55;

/* ── KPIs ────────────────────────────────────────────────── */
/** Deltas compare against the previous run in this session, or null if none. */
export function kpis(run, previous) {
  if (!run) return null;
  const m = run.metrics || {};
  const findings = run.findings || [];
  const insights = run.insights || [];

  const relevant = findings.filter((f) => (f.relevance || 0) >= EMERGING_MIN).length;
  const competitorSignals = findings.filter(
    (f) => f.competitor || (f.signals || []).length
  ).length;
  const highPriority = (m.priority_counts || {}).HIGH || 0;
  const avgRelevance = findings.length
    ? findings.reduce((s, f) => s + (f.relevance || 0), 0) / findings.length
    : 0;

  const prev = previous
    ? {
        findings: (previous.findings || []).length,
        topics: topics(previous).length,
        relevance: (previous.findings || []).length
          ? (previous.findings || []).reduce((s, f) => s + (f.relevance || 0), 0) /
            (previous.findings || []).length
          : 0,
        competitor: (previous.findings || []).filter(
          (f) => f.competitor || (f.signals || []).length
        ).length,
      }
    : null;

  const delta = (now, before) => {
    if (!prev || before === undefined || before === null || before === 0) return null;
    return ((now - before) / before) * 100;
  };

  const topicList = topics(run);

  return [
    {
      key: "findings",
      icon: "📚",
      accent: "blue",
      label: "Intelligence Gathered",
      value: findings.length,
      sub: `${relevant} above relevance bar`,
      delta: delta(findings.length, prev?.findings),
    },
    {
      key: "topics",
      icon: "🔥",
      accent: "yellow",
      label: "Emerging Topics",
      value: topicList.length,
      sub: topicList.length ? `led by ${topicList[0].label}` : "none detected",
      delta: delta(topicList.length, prev?.topics),
    },
    {
      key: "relevance",
      icon: "🧠",
      accent: "purple",
      label: "Avg AI Relevance",
      value: Math.round(avgRelevance * 100),
      suffix: "%",
      sub: `${highPriority} high priority`,
      delta: delta(avgRelevance, prev?.relevance),
    },
    {
      key: "competitive",
      icon: "🏢",
      accent: "orange",
      label: "Competitive Signals",
      value: competitorSignals,
      sub: `${insights.length} insights written`,
      delta: delta(competitorSignals, prev?.competitor),
    },
  ];
}

/* ── activity timeline (real publication dates) ──────────── */
/**
 * Buckets findings by publication date. Returns null when too few findings carry
 * a date to draw an honest series.
 */
export function activitySeries(run, windowDays = 30) {
  const findings = (run?.findings || []).filter((f) => f.published_date);
  if (findings.length < 3) return null;

  const buckets = new Map();
  const step = windowDays <= 7 ? 1 : windowDays <= 30 ? 1 : windowDays <= 90 ? 7 : 30;
  const slots = Math.max(4, Math.ceil(windowDays / step));

  for (let i = slots - 1; i >= 0; i--) {
    const end = i * step;
    buckets.set(end, { bucket: end, total: 0, high: 0, emerging: 0 });
  }

  let used = 0;
  for (const f of findings) {
    const age = relativeDays(f.published_date);
    if (age === null || age > windowDays) continue;
    const slot = Math.min(slots - 1, Math.floor(age / step)) * step;
    const b = buckets.get(slot);
    if (!b) continue;
    b.total += 1;
    if ((f.relevance || 0) >= HIGH_RELEVANCE) b.high += 1;
    else if ((f.relevance || 0) >= EMERGING_MIN) b.emerging += 1;
    used += 1;
  }
  if (used < 3) return null;

  const points = [...buckets.values()]
    .sort((a, b) => b.bucket - a.bucket)
    .map((b) => ({
      ...b,
      label: b.bucket === 0 ? "now" : step === 1 ? `${b.bucket}d` : `-${b.bucket}d`,
    }));

  return { points, used, windowDays, step, total: findings.length };
}

/* ── emerging topics ─────────────────────────────────────── */
const STOP = new Set([
  "the","a","an","and","or","for","with","from","that","this","into","over","its",
  "new","using","based","toward","towards","via","of","in","on","to","by","as","at",
  "is","are","be","how","why","what","when","which","their","our","your","we","it",
  "can","will","has","have","been","more","than","after","before","first","also",
  "study","paper","research","report","news","says","said","announces","announced",
  "system","method","systems","methods","approach","model","models","data","use",
]);

/**
 * Topics are derived two ways and merged:
 *   1. strategic signals the agent itself detected (authoritative)
 *   2. recurring multi-word phrases across finding titles (observed)
 * Growth compares the recent half of the window against the older half.
 */
export function topics(run, limit = 8) {
  const findings = run?.findings || [];
  if (findings.length < 3) return [];

  const dated = findings.filter((f) => relativeDays(f.published_date) !== null);
  const ages = dated.map((f) => relativeDays(f.published_date));
  const midpoint = ages.length ? median(ages) : 30;

  const counts = new Map();
  const bump = (key, label, f, kind) => {
    if (!counts.has(key)) {
      counts.set(key, {
        key, label, kind, count: 0, recent: 0, older: 0,
        relevanceSum: 0, categories: new Set(), items: [],
      });
    }
    const t = counts.get(key);
    t.count += 1;
    t.relevanceSum += f.relevance || 0;
    t.categories.add(f.source);
    if (t.items.length < 24) t.items.push(f.id);
    const age = relativeDays(f.published_date);
    if (age === null) return;
    if (age <= midpoint) t.recent += 1; else t.older += 1;
  };

  for (const f of findings) {
    for (const sig of f.signals || []) {
      bump(`signal:${sig}`, SIGNAL_LABEL[sig] || sig, f, "signal");
    }
    for (const phrase of phrases(f.title)) {
      bump(`phrase:${phrase}`, phrase, f, "phrase");
    }
  }

  const out = [...counts.values()]
    .filter((t) => t.count >= 2)
    .map((t) => {
      const growth = t.older > 0
        ? ((t.recent - t.older) / t.older) * 100
        : t.recent > 0 ? null : 0;   // null = new this window, no baseline
      return {
        key: t.key,
        label: t.label,
        kind: t.kind,
        count: t.count,
        growth,
        isNew: t.older === 0 && t.recent > 0,
        confidence: Math.round((t.relevanceSum / t.count) * 100),
        categories: [...t.categories],
        items: t.items,
      };
    })
    .sort((a, b) => {
      const ga = a.isNew ? 999 : (a.growth ?? 0);
      const gb = b.isNew ? 999 : (b.growth ?? 0);
      if (gb !== ga) return gb - ga;
      return b.count - a.count;
    });

  // Prefer variety: don't fill the panel with eight near-identical phrases.
  const picked = [];
  const seenWords = new Set();
  for (const t of out) {
    const words = new Set(t.label.toLowerCase().split(/\s+/));
    const overlap = [...words].filter((w) => seenWords.has(w)).length;
    if (overlap >= words.size && picked.length) continue;
    words.forEach((w) => seenWords.add(w));
    picked.push(t);
    if (picked.length >= limit) break;
  }
  return picked;
}

function phrases(title) {
  const words = String(title || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));
  const out = [];
  for (let i = 0; i < words.length - 1; i++) {
    out.push(`${words[i]} ${words[i + 1]}`);
  }
  return out.slice(0, 8);
}

function median(nums) {
  const s = [...nums].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

/* ── research landscape (bubbles) ────────────────────────── */
export function landscape(run, limit = 12) {
  const list = topics(run, limit);
  if (!list.length) return [];
  const max = Math.max(...list.map((t) => t.count));
  return list.map((t) => ({
    ...t,
    weight: max ? t.count / max : 0,
  }));
}

/* ── source distribution ─────────────────────────────────── */
export function sources(run) {
  const findings = run?.findings || [];
  if (!findings.length) return [];
  const map = new Map();
  for (const f of findings) {
    const key = f.provider || "unknown";
    if (!map.has(key)) {
      map.set(key, { key, count: 0, live: 0, simulated: 0, freshestDays: null, relevanceSum: 0 });
    }
    const s = map.get(key);
    s.count += 1;
    s.relevanceSum += f.relevance || 0;
    if (f.simulated) s.simulated += 1; else s.live += 1;
    const age = relativeDays(f.published_date);
    if (age !== null && (s.freshestDays === null || age < s.freshestDays)) s.freshestDays = age;
  }
  return [...map.values()]
    .map((s) => ({
      ...s,
      quality: Math.round((s.relevanceSum / s.count) * 100),
      isLive: s.live > 0 && s.simulated === 0,
      isMixed: s.live > 0 && s.simulated > 0,
      isSimulated: s.live === 0,
    }))
    .sort((a, b) => b.count - a.count);
}

/* ── competitor intelligence ─────────────────────────────── */
export function competitors(run) {
  const names = run?.state?.competitors || [];
  const findings = run?.findings || [];
  const insights = run?.insights || [];
  if (!names.length) return [];

  return names.map((name) => {
    const needle = name.toLowerCase();
    const mine = findings.filter(
      (f) =>
        (f.competitor || "").toLowerCase() === needle ||
        `${f.title} ${f.summary}`.toLowerCase().includes(needle)
    );
    const byCategory = {};
    for (const key of Object.keys(CATEGORY)) {
      byCategory[key] = mine.filter((f) => f.source === key).length;
    }
    const signals = new Set();
    mine.forEach((f) => (f.signals || []).forEach((s) => signals.add(s)));

    const related = insights.filter(
      (i) => (i.competitor || "").toLowerCase() === needle
    );
    const top = related.sort(
      (a, b) => rank(a.priority) - rank(b.priority) || (b.score || 0) - (a.score || 0)
    )[0] || null;

    const dated = mine
      .filter((f) => f.published_date)
      .sort((a, b) => String(b.published_date).localeCompare(String(a.published_date)));

    return {
      name,
      total: mine.length,
      byCategory,
      signals: [...signals],
      priority: top ? top.priority : mine.length ? "MEDIUM" : "LOW",
      latest: dated[0] || null,
      topInsight: top,
      items: mine.map((f) => f.id),
      simulated: mine.length > 0 && mine.every((f) => f.simulated),
    };
  }).sort((a, b) => b.total - a.total);
}

const rank = (p) => ({ HIGH: 0, MEDIUM: 1, LOW: 2 }[p] ?? 3);

/* ── top researchers / organisations ─────────────────────── */
/**
 * Only counts findings that actually carry an author or assignee. Returns [] when
 * the providers used did not supply attribution — never a placeholder name.
 */
export function contributors(run, limit = 6) {
  const findings = run?.findings || [];
  const map = new Map();

  for (const f of findings) {
    const meta = f.meta || {};
    const isPatent = f.source === "patent";
    const raw = isPatent ? meta.assignee || f.author : f.author;
    if (!raw) continue;

    const names = isPatent
      ? [String(raw)]
      : String(raw).split(",").map((n) => n.trim()).filter(Boolean).slice(0, 4);

    for (const name of names) {
      if (name.length < 3 || name.length > 60) continue;
      const key = name.toLowerCase();
      if (!map.has(key)) {
        map.set(key, {
          name, works: 0, citations: 0, kind: isPatent ? "organisation" : "researcher",
          venues: new Set(), institutions: new Set(), items: [],
          simulatedWorks: 0,
        });
      }
      const c = map.get(key);
      c.works += 1;
      if (f.simulated) c.simulatedWorks += 1;
      c.citations += Number(meta.citation_count) || 0;
      if (meta.venue) c.venues.add(meta.venue);
      for (const inst of meta.institutions || (meta.institution ? [meta.institution] : [])) {
        if (inst) c.institutions.add(inst);
      }
      if (c.items.length < 12) c.items.push(f.id);
    }
  }

  return [...map.values()]
    .filter((c) => c.works > 0)
    .map((c) => ({
      ...c,
      venue: [...c.venues][0] || "",
      institution: [...c.institutions][0] || "",
      // A name that only ever appeared in simulated findings is not a real person
      // or organisation, and the UI must say so rather than imply attribution.
      simulated: c.simulatedWorks === c.works,
      partlySimulated: c.simulatedWorks > 0 && c.simulatedWorks < c.works,
    }))
    .sort((a, b) => b.citations - a.citations || b.works - a.works)
    .slice(0, limit);
}

/* ── connected intelligence (cross-source links) ─────────── */
/**
 * Finds genuine relationships between findings from *different* categories.
 * A link requires either a shared tracked company or a shared strategic signal,
 * plus title-token overlap. Confidence is the normalised strength of that
 * evidence — it is a derived score, and the UI labels it as such.
 */
export function connections(run, limit = 4) {
  const findings = (run?.findings || []).filter((f) => (f.relevance || 0) >= 0.4);
  if (findings.length < 2) return [];

  const byCompany = new Map();
  const bySignal = new Map();
  for (const f of findings) {
    if (f.competitor) {
      const k = f.competitor.toLowerCase();
      if (!byCompany.has(k)) byCompany.set(k, []);
      byCompany.get(k).push(f);
    }
    for (const s of f.signals || []) {
      if (!bySignal.has(s)) bySignal.set(s, []);
      bySignal.get(s).push(f);
    }
  }

  const chains = [];

  const build = (group, anchorLabel, anchorKind) => {
    const seenCat = new Map();
    for (const f of group) {
      const best = seenCat.get(f.source);
      if (!best || (f.relevance || 0) > (best.relevance || 0)) seenCat.set(f.source, f);
    }
    if (seenCat.size < 2) return;

    const members = [...seenCat.values()].sort(
      (a, b) => order(a.source) - order(b.source)
    );
    const overlap = tokenOverlap(members);
    const avgRelevance =
      members.reduce((s, f) => s + (f.relevance || 0), 0) / members.length;
    const confidence = Math.round(
      Math.min(97, 40 + seenCat.size * 12 + overlap * 22 + avgRelevance * 18)
    );

    chains.push({
      key: `${anchorKind}:${anchorLabel}`,
      anchor: anchorLabel,
      anchorKind,
      members,
      categories: members.map((m) => m.source),
      confidence,
      simulated: members.some((m) => m.simulated),
    });
  };

  for (const [company, group] of byCompany) {
    const display = group[0].competitor || company;
    build(group, display, "company");
  }
  for (const [signal, group] of bySignal) {
    if (group.length < 2) continue;
    build(group, SIGNAL_LABEL[signal] || signal, "signal");
  }

  const seen = new Set();
  return chains
    .sort((a, b) => b.members.length - a.members.length || b.confidence - a.confidence)
    .filter((c) => {
      const sig = c.members.map((m) => m.id).sort().join("|");
      if (seen.has(sig)) return false;
      seen.add(sig);
      return true;
    })
    .slice(0, limit);
}

const CHAIN_ORDER = { research: 0, patent: 1, competitor: 2, web: 3, news: 4 };
const order = (k) => CHAIN_ORDER[k] ?? 9;

function tokenOverlap(items) {
  if (items.length < 2) return 0;
  const sets = items.map(
    (f) => new Set(
      String(f.title || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/).filter((w) => w.length > 3 && !STOP.has(w))
    )
  );
  let shared = 0;
  let comparisons = 0;
  for (let i = 0; i < sets.length; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      comparisons += 1;
      const inter = [...sets[i]].filter((w) => sets[j].has(w)).length;
      const union = new Set([...sets[i], ...sets[j]]).size || 1;
      shared += inter / union;
    }
  }
  return comparisons ? shared / comparisons : 0;
}

/* ── executive summary counts ────────────────────────────── */
export function summaryCounts(run) {
  const c = (run?.metrics?.priority_counts) || {};
  return {
    HIGH: c.HIGH || 0,
    MEDIUM: c.MEDIUM || 0,
    LOW: c.LOW || 0,
    total: (run?.insights || []).length,
  };
}

/** One readable sentence about the dominant pattern, derived from signals. */
export function mainTrend(run) {
  const detected = run?.metrics?.signals_detected || [];
  const cov = run?.metrics?.coverage || {};
  const topicList = topics(run, 3);
  const parts = [];

  if (detected.length) {
    const phrase = detected.slice(0, 2).map((s) => SIGNAL_LABEL[s] || s).join(" and ");
    parts.push(`Rising ${phrase}`);
  } else if (topicList.length) {
    parts.push(`Activity concentrating around ${topicList[0].label}`);
  } else {
    const top = Object.entries(cov).sort((a, b) => b[1] - a[1])[0];
    if (top) parts.push(`Most signal is coming from ${(CATEGORY[top[0]] || {}).label || top[0]}`);
  }

  const companies = [...new Set((run?.insights || []).map((i) => i.competitor).filter(Boolean))];
  if (companies.length) parts.push(`concentrated around ${companies.slice(0, 2).join(" and ")}`);

  return parts.length ? `${parts.join(", ")}.` : "";
}
