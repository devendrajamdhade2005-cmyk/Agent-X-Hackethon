# Railway deployment

InsightPulse deploys as **one service**: FastAPI serves the API *and* the dashboard,
so there is no separate frontend to host and no CORS or API-URL wiring to do.

---

## Deploy steps

1. **Push the repository to GitHub** (Railway deploys from a connected repo).

2. In Railway: **New Project → Deploy from GitHub repo**, and select this repository.

3. Railway reads [`railway.json`](./railway.json) from the repo root and uses the
   build and start commands there. Nixpacks detects Python and installs
   `backend/requirements.txt`. No Dockerfile is needed.

4. **Add the environment variables** (next section) under
   *Service → Variables*. The app boots with none of them set — it falls back to its
   heuristic reasoner and simulated sources — so add at minimum `GEMINI_API_KEY` to
   get real reasoning.

5. **Generate a domain**: *Service → Settings → Networking → Generate Domain*.
   Open it and you get the dashboard. `/docs` serves the OpenAPI UI.

6. Confirm the deployment is healthy:

   ```bash
   curl https://<your-app>.up.railway.app/health
   # {"status":"ok", ...}
   ```

> **Do not set `PORT` yourself.** Railway injects it and the app binds to it. Setting
> it manually to a different value than Railway routes to will fail the healthcheck.

### Optional: persist run history across redeploys

Container filesystems are ephemeral, so run history, reports and traces reset on each
deploy. To keep them, attach a volume (*Service → Settings → Volumes*, mount at
`/data`) and set:

```
DATA_DIR=/data
EXPORT_DIR=/data/exports
```

Everything works without this — the data simply does not survive a restart.

---

## Required environment variables

Nothing is mandatory. Each key you add switches a capability from *simulated* to
*live*, and `/health` reports exactly which ones are active.

| Variable | Needed for | If omitted |
|---|---|---|
| `GEMINI_API_KEY` | LLM reasoning (planning, insights) | Falls back to the deterministic heuristic reasoner |
| `TAVILY_API_KEY` | Web search tool | Web search degrades to other providers |
| `NEWSAPI_KEY` | News provider | Other news providers still used |
| `NEWSDATA_API_KEY` | News provider | as above |
| `GNEWS_API_KEY` | News provider | as above |
| `SEMANTIC_SCHOLAR_API_KEY` | Higher research rate limits, citation counts | Works unauthenticated, lower limits |
| `SERPAPI_KEY` | Patent search via Google Patents | Patent search falls back to PatentsView |
| `PATENTSVIEW_API_KEY` | Patent provider | Serves clearly-labelled simulated fixtures |
| `GITHUB_TOKEN` | Raises GitHub rate limit (60/hr → 5000/hr) | Anonymous access, more throttling |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit source | Reddit 403s most server IPs without it |

### Configuration (non-secret)

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Set to `production` on Railway |
| `SIMULATION_MODE` | `false` | `true` forces offline fixtures — useful for a zero-cost demo |
| `AGENT_API_TOKEN` | *(empty)* | Set it to require `X-API-Token` on run endpoints and lift rate limits for that caller |
| `CORS_ORIGINS` | localhost list | Only needed for split hosting; single-service needs nothing |
| `DATA_DIR` / `EXPORT_DIR` | `backend/data`, `backend/exports` | Point at a mounted volume to persist data |
| `PORT` | injected by Railway | Do not set manually |

Full list with comments: [`backend/.env.example`](./backend/.env.example).

**Never commit real keys.** `.env` and any `.env.*` variant are git-ignored; only
`.env.example` (blank placeholders) is tracked. Set real values in the Railway
dashboard, not in a file.

---

## Start command

From [`railway.json`](./railway.json):

```bash
cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

- `0.0.0.0` is required — binding `localhost` makes the container unreachable to
  Railway's router and the healthcheck fails.
- `$PORT` is assigned by Railway at runtime.
- `python main.py` also works and reads `$PORT` the same way, if you prefer it.

`numReplicas` is pinned to **1** deliberately: run history, traces, the rate limiter
and the evaluation store are all in-process, so a second replica would answer with
inconsistent state.

---

## Healthcheck

| Setting | Value |
|---|---|
| Path | `/health` |
| Expected | `200` with `{"status":"ok", ...}` |
| Timeout | `300` s (`healthcheckTimeout` in `railway.json`) |
| On failure | Restart, max 3 retries (`ON_FAILURE`) |

The endpoint also reports the active reasoner, usable tools, which provider keys are
live, and the current rate limits — useful for confirming a deploy picked up its
variables. It performs no outbound calls, so it stays fast and cheap.

---

## Frontend API URL setup

**Single service (the default): nothing to configure.** FastAPI serves the dashboard,
so the frontend resolves the API base to same-origin (`""`) and calls whatever host
served the page. The same build works on localhost, a Railway domain and a custom
domain with no edit. No deployment host is hardcoded anywhere in the frontend.

**Split hosting** — only if you serve the static frontend from a different origin
(a CDN, or an existing Vercel deployment) and point it at the Railway API. Set the
API origin in [`backend/static/index.html`](./backend/static/index.html):

```html
<meta name="api-base" content="https://<your-app>.up.railway.app" />
```

Or override at runtime before the app module loads:

```html
<script>window.__API_BASE__ = "https://<your-app>.up.railway.app";</script>
```

Resolution order is `window.__API_BASE__` → `<meta name="api-base">` → same-origin.

For split hosting you must also allow the frontend's origin on the backend:

```
CORS_ORIGINS=https://your-frontend.example
```

A backend origin is public information, not a secret. No API key is ever placed in
frontend code — all credentials stay server-side.
