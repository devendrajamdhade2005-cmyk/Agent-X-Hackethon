/* Shared vocabulary and formatting. One definition of every label and colour so
   the semantic colour system stays consistent across every panel. */

export const CATEGORY = {
  research:   { label: "Research",    icon: "🔬", accent: "blue"   },
  patent:     { label: "Patents",     icon: "📜", accent: "cyan"   },
  competitor: { label: "Competitors", icon: "🏢", accent: "orange" },
  news:       { label: "News",        icon: "📰", accent: "green"  },
  web:        { label: "Live Web",    icon: "🌐", accent: "purple" },
};

export const TOOL = {
  research_search:   { label: "Research Search",        human: "research papers"     },
  patent_search:     { label: "Patent Search",          human: "patents"             },
  news_search:       { label: "Industry News Search",   human: "industry news"       },
  competitor_search: { label: "Competitor Intelligence", human: "competitor activity" },
  web_search:        { label: "Live Web Intelligence",  human: "the live web"        },
};

export const PROVIDER = {
  arxiv: "arXiv",
  openalex: "OpenAlex",
  semantic_scholar: "Semantic Scholar",
  patentsview: "PatentsView",
  serpapi: "Google Patents",
  newsapi: "NewsAPI",
  gnews: "GNews",
  rss: "Curated RSS",
  hackernews: "Hacker News",
  reddit: "Reddit",
  github: "GitHub",
  tavily: "Tavily Web",
};

/** Priority bands — always paired with text, never colour alone (§10). */
export const PRIORITY = {
  HIGH:   { label: "High Priority",   dot: "🔴", tone: "red"   },
  MEDIUM: { label: "Worth Watching",  dot: "🟡", tone: "amber" },
  LOW:    { label: "Low Priority",    dot: "🟢", tone: "green" },
};

/** Relevance bands used by the research feed (§10). */
export function relevanceBand(score) {
  const s = Number(score) || 0;
  if (s >= 0.7) return { key: "high",     label: "High Relevance", dot: "🔵", tone: "blue"   };
  if (s >= 0.55) return { key: "emerging", label: "Emerging",       dot: "🟣", tone: "purple" };
  if (s >= 0.4) return { key: "watch",    label: "Watch",          dot: "🟠", tone: "orange" };
  return { key: "low", label: "Low Priority", dot: "⚪", tone: "slate" };
}

export const SIGNAL_LABEL = {
  patent: "patent filing",
  launch: "product launch",
  funding: "funding",
  partnership: "partnership",
  acquisition: "acquisition",
  regulatory: "regulatory action",
  benchmark: "performance claim",
  hiring: "hiring move",
  "shipped-code": "shipped open-source code",
  "competitor-assignee": "competitor-owned IP",
};

export const categoryLabel = (key) => (CATEGORY[key] || { label: key }).label;
export const categoryIcon = (key) => (CATEGORY[key] || { icon: "•" }).icon;
export const toolLabel = (key) => (TOOL[key] || { label: key }).label;
export const toolHuman = (key) => (TOOL[key] || { human: key }).human;
export const providerLabel = (key) => PROVIDER[key] || key || "unknown";

/** "Competitor activity · rss" -> "competitor" */
export function categoryOf(sourceText) {
  const head = String(sourceText || "").split("·")[0].trim().toLowerCase();
  for (const key of ["research", "patent", "competitor", "news", "web"]) {
    if (head.startsWith(key)) return key;
  }
  return "news";
}

export function formatDate(iso) {
  if (!iso) return "date unknown";
  const d = new Date(`${String(iso).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function relativeDays(iso) {
  if (!iso) return null;
  const d = new Date(`${String(iso).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(d.getTime())) return null;
  return Math.max(0, Math.round((Date.now() - d.getTime()) / 86400000));
}

export function ago(iso) {
  const days = relativeDays(iso);
  if (days === null) return "date unknown";
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.round(days / 30)}mo ago`;
  return `${Math.round(days / 365)}y ago`;
}

export const pct = (n) => `${n > 0 ? "+" : ""}${(Number(n) || 0).toFixed(1)}%`;

export function shortenUrl(url, limit = 46) {
  const text = String(url || "").replace(/^https?:\/\//, "").replace(/\/$/, "");
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
}

export function truncate(text, limit) {
  const t = String(text || "");
  return t.length <= limit ? t : `${t.slice(0, limit - 1)}…`;
}

/** Sentence-case a derived phrase without mangling acronyms. */
export function sentence(text) {
  const t = String(text || "").trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : "";
}
