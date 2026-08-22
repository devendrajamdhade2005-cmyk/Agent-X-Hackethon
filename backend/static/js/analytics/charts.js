/* Hand-rolled inline SVG charts.
 *
 * Deliberately not Recharts: this frontend has no React and no build step, so
 * pulling in React + Recharts for three charts would mean a bundler, a ~150KB
 * runtime and a slower first paint — directly against the performance rules in
 * §27. These render as static SVG strings, cost nothing to re-render, animate
 * with CSS only, and stay responsive via viewBox.
 */

import { esc } from "../core/dom.js";

/* ── stacked area / line chart ───────────────────────────── */
export function areaChart(series, { width = 760, height = 220 } = {}) {
  const points = series?.points || [];
  if (points.length < 2) return "";

  const pad = { top: 16, right: 14, bottom: 26, left: 34 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const max = Math.max(1, ...points.map((p) => p.total));
  const stepX = points.length > 1 ? w / (points.length - 1) : w;

  const x = (i) => pad.left + i * stepX;
  const y = (v) => pad.top + h - (v / max) * h;

  const line = (key) =>
    points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  const area = (key) =>
    `${line(key)} L${x(points.length - 1).toFixed(1)},${(pad.top + h).toFixed(1)} ` +
    `L${pad.left.toFixed(1)},${(pad.top + h).toFixed(1)} Z`;

  const ticks = 4;
  const gridLines = Array.from({ length: ticks + 1 }, (_, i) => {
    const value = Math.round((max / ticks) * i);
    const gy = y(value);
    return `<line x1="${pad.left}" x2="${width - pad.right}" y1="${gy.toFixed(1)}" y2="${gy.toFixed(1)}" class="grid"/>
            <text x="${pad.left - 8}" y="${(gy + 3.5).toFixed(1)}" class="axis" text-anchor="end">${value}</text>`;
  }).join("");

  const everyNth = Math.max(1, Math.ceil(points.length / 7));
  const xLabels = points
    .map((p, i) =>
      i % everyNth === 0 || i === points.length - 1
        ? `<text x="${x(i).toFixed(1)}" y="${height - 8}" class="axis" text-anchor="middle">${esc(p.label)}</text>`
        : ""
    ).join("");

  const dots = points
    .map((p, i) =>
      `<circle cx="${x(i).toFixed(1)}" cy="${y(p.total).toFixed(1)}" r="3.2" class="dot-total"
        tabindex="0" role="img"
        aria-label="${esc(p.label)}: ${p.total} findings, ${p.high} high relevance">
        <title>${esc(p.label)} — ${p.total} finding(s), ${p.high} high relevance</title>
      </circle>`
    ).join("");

  return `
<svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img"
     aria-label="Intelligence activity over time">
  <defs>
    <linearGradient id="gTotal" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#6366f1" stop-opacity=".28"/>
      <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="gHigh" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#06b6d4" stop-opacity=".26"/>
      <stop offset="100%" stop-color="#06b6d4" stop-opacity="0"/>
    </linearGradient>
  </defs>
  ${gridLines}
  <path d="${area("total")}" fill="url(#gTotal)" class="area"/>
  <path d="${line("total")}" fill="none" stroke="#6366f1" stroke-width="2.2"
        stroke-linejoin="round" stroke-linecap="round" class="series"/>
  <path d="${area("high")}" fill="url(#gHigh)" class="area"/>
  <path d="${line("high")}" fill="none" stroke="#06b6d4" stroke-width="1.8"
        stroke-dasharray="4 3" stroke-linejoin="round" class="series"/>
  ${dots}
  ${xLabels}
</svg>`;
}

/* ── donut ───────────────────────────────────────────────── */
const DONUT_COLORS = [
  "#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b",
  "#ec4899", "#10b981", "#ef4444", "#64748b",
];

export function donut(slices, { size = 168, thickness = 22 } = {}) {
  const total = slices.reduce((s, x) => s + x.count, 0);
  if (!total) return "";

  const r = (size - thickness) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  const arcs = slices.map((slice, i) => {
    const frac = slice.count / total;
    const len = frac * circumference;
    const dash = `${len.toFixed(2)} ${(circumference - len).toFixed(2)}`;
    const rotation = (offset / circumference) * 360 - 90;
    offset += len;
    return `<circle cx="${c}" cy="${c}" r="${r.toFixed(2)}" fill="none"
      stroke="${DONUT_COLORS[i % DONUT_COLORS.length]}" stroke-width="${thickness}"
      stroke-dasharray="${dash}" stroke-linecap="butt"
      transform="rotate(${rotation.toFixed(2)} ${c} ${c})" class="arc">
      <title>${esc(slice.label)} — ${slice.count} (${Math.round(frac * 100)}%)</title>
    </circle>`;
  }).join("");

  return `
<svg class="donut" viewBox="0 0 ${size} ${size}" role="img" aria-label="Source distribution">
  <circle cx="${c}" cy="${c}" r="${r.toFixed(2)}" fill="none" stroke="#eef1f7" stroke-width="${thickness}"/>
  ${arcs}
  <text x="${c}" y="${c - 2}" class="donut-value" text-anchor="middle">${total}</text>
  <text x="${c}" y="${c + 14}" class="donut-label" text-anchor="middle">findings</text>
</svg>`;
}

export const donutColor = (i) => DONUT_COLORS[i % DONUT_COLORS.length];

/* ── landscape bubbles ──────────────────────────────────── */
/** Size = volume, colour = growth. Deterministic packing, no physics sim. */
export function bubbles(items, { width = 560, height = 260 } = {}) {
  if (!items.length) return "";

  const placed = [];
  const maxR = Math.min(54, height / 3.4);
  const minR = 20;

  const sorted = [...items].sort((a, b) => b.weight - a.weight);
  for (const item of sorted) {
    const r = minR + (maxR - minR) * Math.sqrt(item.weight);
    let best = null;
    // Spiral search for the first non-overlapping slot — stable and cheap.
    for (let attempt = 0; attempt < 220; attempt++) {
      const angle = attempt * 2.399;
      const dist = 6 + attempt * 2.05;
      const cx = width / 2 + Math.cos(angle) * dist * (width / height) * 0.62;
      const cy = height / 2 + Math.sin(angle) * dist * 0.62;
      if (cx - r < 4 || cx + r > width - 4 || cy - r < 4 || cy + r > height - 4) continue;
      const clash = placed.some(
        (p) => Math.hypot(p.cx - cx, p.cy - cy) < p.r + r + 4
      );
      if (!clash) { best = { cx, cy, r }; break; }
    }
    if (best) placed.push({ ...best, item });
  }

  return `
<svg class="bubbles" viewBox="0 0 ${width} ${height}" role="img" aria-label="Research landscape">
  ${placed.map(({ cx, cy, r, item }) => {
    const fill = growthColor(item);
    const label = item.label.length > 16 ? `${item.label.slice(0, 15)}…` : item.label;
    const fontSize = Math.max(9, Math.min(12.5, r / 3.2));
    return `<g class="bubble" tabindex="0" role="button"
              data-topic="${esc(item.key)}"
              aria-label="${esc(item.label)}: ${item.count} findings">
      <title>${esc(item.label)} — ${item.count} finding(s)${
        item.isNew ? ", new this window" : item.growth !== null ? `, ${item.growth > 0 ? "+" : ""}${Math.round(item.growth)}%` : ""
      }</title>
      <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}"
              fill="${fill}" fill-opacity=".16" stroke="${fill}" stroke-width="1.4"/>
      <text x="${cx.toFixed(1)}" y="${(cy + 1).toFixed(1)}" text-anchor="middle"
            class="bubble-label" style="font-size:${fontSize.toFixed(1)}px" fill="${fill}">${esc(label)}</text>
      <text x="${cx.toFixed(1)}" y="${(cy + fontSize + 3).toFixed(1)}" text-anchor="middle"
            class="bubble-count">${item.count}</text>
    </g>`;
  }).join("")}
</svg>`;
}

function growthColor(item) {
  if (item.isNew) return "#ec4899";              // new this window
  const g = item.growth ?? 0;
  if (g >= 40) return "#ef4444";
  if (g >= 15) return "#f59e0b";
  if (g > 0) return "#8b5cf6";
  if (g === 0) return "#6366f1";
  return "#64748b";                              // declining
}

/* ── sparkline (competitor rows) ─────────────────────────── */
export function sparkline(values, { width = 68, height = 22, color = "#6366f1" } = {}) {
  if (!values || values.length < 2) return "";
  const max = Math.max(1, ...values);
  const stepX = width / (values.length - 1);
  const d = values
    .map((v, i) => `${i ? "L" : "M"}${(i * stepX).toFixed(1)},${(height - (v / max) * height).toFixed(1)}`)
    .join(" ");
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" aria-hidden="true">
    <path d="${d}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round"/>
  </svg>`;
}

/* ── horizontal meter ───────────────────────────────────── */
export function meter(value, max, { tone = "blue" } = {}) {
  const pctv = max ? Math.round((value / max) * 100) : 0;
  return `<span class="meter tone-${tone}" role="img" aria-label="${value} of ${max}">
    <span style="width:${Math.max(3, pctv)}%"></span></span>`;
}
