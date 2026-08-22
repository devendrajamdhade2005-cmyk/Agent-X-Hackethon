/* Single source of truth for where the backend lives.
 *
 * This is the ONE place to configure the backend base URL. Everything in
 * core/api.js routes through apiUrl(), so nothing else hardcodes a host.
 *
 * How the base URL is resolved (first match wins):
 *   1. window.__API_BASE__            explicit runtime override (console/inline script)
 *   2. <meta name="api-base" content> optional static override in index.html
 *   3. localhost / 127.0.0.1          same-origin — the local FastAPI dev server
 *   4. anything else (e.g. Vercel)    PRODUCTION_API_BASE below
 *
 * Local development: served by FastAPI at http://localhost:8000, so the hostname
 * is localhost and API_BASE is "" (same-origin). Nothing to configure.
 *
 * Production: the static frontend is hosted on Vercel (a different origin), so it
 * targets the Render backend below. The backend URL is public, not a secret — no
 * API keys are ever placed in frontend code.
 */

const PRODUCTION_API_BASE = "https://insightpulse-ai-agent.onrender.com";

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"]);

function resolveApiBase() {
  if (typeof window !== "undefined") {
    if (window.__API_BASE__) {
      return String(window.__API_BASE__).replace(/\/+$/, "");
    }
    const meta =
      typeof document !== "undefined"
        ? document.querySelector('meta[name="api-base"]')
        : null;
    if (meta && meta.content) {
      return meta.content.trim().replace(/\/+$/, "");
    }
  }

  const host =
    typeof location !== "undefined" && location.hostname ? location.hostname : "";
  if (LOCAL_HOSTS.has(host)) {
    return ""; // same-origin: hit whatever served this page (local FastAPI)
  }
  return PRODUCTION_API_BASE;
}

/** Configured backend origin. "" means same-origin (local dev). */
export const API_BASE = resolveApiBase();

/** Join the configured base with an absolute API path like "/api/…" or "/health". */
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}
