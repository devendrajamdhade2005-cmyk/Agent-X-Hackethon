/* Tiny DOM helpers. No framework: the whole point is a fast first paint and
   incremental updates during SSE streaming. */

export const $ = (id) => document.getElementById(id);
export const qs = (sel, root = document) => root.querySelector(sel);
export const qsa = (sel, root = document) => [...root.querySelectorAll(sel)];

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Set innerHTML only when it actually changed — avoids needless reflow. */
export function setHTML(node, html) {
  if (!node) return;
  if (node.__last === html) return;
  node.__last = html;
  node.innerHTML = html;
}

export function setText(node, text) {
  if (!node) return;
  const v = String(text ?? "");
  if (node.textContent !== v) node.textContent = v;
}

export function show(node, visible = true) {
  if (!node) return;
  node.hidden = !visible;
}

/** Delegated listener — one handler for a whole list instead of N handlers. */
export function delegate(root, selector, event, handler) {
  if (!root) return;
  root.addEventListener(event, (e) => {
    const match = e.target.closest(selector);
    if (match && root.contains(match)) handler(e, match);
  });
}

/** Render only when the section scrolls into view (§27 lazy-render). */
export function onVisible(node, fn, { once = true } = {}) {
  if (!node) return;
  if (!("IntersectionObserver" in window)) { fn(); return; }
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      fn();
      if (once) io.disconnect();
    }
  }, { rootMargin: "160px" });
  io.observe(node);
}

/** Count-up for KPI values. Skipped when the user prefers reduced motion. */
export function countUp(node, to, { duration = 620, suffix = "" } = {}) {
  if (!node) return;
  const target = Number(to);
  if (!Number.isFinite(target)) { setText(node, to); return; }
  // Respect reduced-motion, and tolerate environments without matchMedia.
  const reduced = typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || typeof requestAnimationFrame !== "function") {
    setText(node, target.toLocaleString() + suffix);
    return;
  }
  const start = Date.now();
  const from = 0;
  const step = () => {
    const p = Math.min(1, (Date.now() - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const value = Math.round(from + (target - from) * eased);
    node.textContent = value.toLocaleString() + suffix;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
