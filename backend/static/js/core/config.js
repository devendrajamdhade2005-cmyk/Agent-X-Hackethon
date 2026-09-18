/* Single source of truth for where the backend lives.
 *
 * This is the ONE place to configure the backend base URL. Everything in
 * core/api.js routes through apiUrl(), so nothing else hardcodes a host.
 *
 * How the base URL is resolved (first match wins):
 *   1. window.__API_BASE__            explicit runtime override (console/inline script)
 *   2. <meta name="api-base" content> static override in index.html
 *   3. same-origin ("")               whatever host served this page
 *
 * No host is hardcoded. Same-origin is the default because the usual deployment is
 * one service: FastAPI serves this page *and* the API, which is true both locally
 * and on a platform host like Railway. In that arrangement there is nothing to
 * configure — the UI simply calls the host it was loaded from, so the same build
 * works on localhost, a preview URL and production without an edit.
 *
 * The override exists for split hosting, where the static frontend lives on a
 * different origin from the API (e.g. a CDN or Vercel front end pointed at a
 * Railway backend). Set the meta tag in index.html to the API origin:
 *
 *     <meta name="api-base" content="https://your-app.up.railway.app" />
 *
 * A backend origin is public information, not a secret. No API key is ever placed
 * in frontend code — every credential stays server-side in the backend settings.
 */

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

  // Same-origin. Correct for the single-service deployment and for local dev, and
  // it keeps the deployed host out of the source entirely.
  return "";
}

/** True when the page is being served from a local dev host. */
export const IS_LOCAL_HOST = (() => {
  const host =
    typeof location !== "undefined" && location.hostname ? location.hostname : "";
  return LOCAL_HOSTS.has(host);
})();

/** Configured backend origin. "" means same-origin (local dev). */
export const API_BASE = resolveApiBase();

/** Join the configured base with an absolute API path like "/api/…" or "/health". */
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}
