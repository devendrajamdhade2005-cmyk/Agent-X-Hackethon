# 🔍 InsightPulse AI — Autonomous Research & Competitor Intelligence Agent

> An autonomous AI agent that gathers intelligence from research papers, patents, competitor
> activity, industry news and the live web — reasons about what it finds, and delivers a
> prioritized, actionable briefing with a downloadable report.

---

## 👥 Team Members

- Akash Pingale
- Devendra Jamdhade
- Shubham Paithankar
- Gaurav Bodkhe
- Shubham Sonwane

---

## 📌 Problem Statement

Organizations, startups and research institutions operate in highly competitive and rapidly
evolving environments where staying updated on research trends, patent developments, competitor
strategies and industry news is critical. Manually monitoring scientific publications, patent
databases, news platforms and social sources is time-consuming, inefficient and prone to missing
important updates.

InsightPulse is an **autonomous AI agent** that continuously tracks research and competitor
activity, analyses many information sources, and delivers concise, actionable insights.

---

## ⚡ Quick start

No API keys are required. With zero configuration the agent runs end to end, clearly labelling
which data is live and which is simulated.

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**, then press **Run Intelligence Scan**.

| URL | What it is |
|---|---|
| `http://localhost:8000` | Intelligence dashboard |
| `http://localhost:8000/docs` | OpenAPI / Swagger |
| `http://localhost:8000/health` | Capability report — which reasoner and sources are live |

Optional keys go in `backend/.env` (see `backend/.env.example`, which documents every one and
where to get it free).

---

## 🧠 How the agent works — a real ReAct loop

```
GOAL → REASON → DECIDE NEXT ACTION → SELECT TOOL → CALL TOOL
                        ↑                                │
                        └──── ANALYZE ←── OBSERVE ───────┘
                                   │
                        enough evidence? → PRIORITIZED INSIGHTS
```

This is **not** a fixed pipeline. The agent chooses each next action from its current state, and
the same goal can produce different tool sequences depending on what earlier calls returned.

Proven by the test suite:

| Goal | Tool sequence chosen |
|---|---|
| "Track research developments in AI agents" | `research_search` |
| "Monitor patents related to generative AI" | `patent_search` |
| "Monitor industry news about solid-state batteries" | `news_search` |
| "Track competitor announcements from OpenAI and Anthropic" | `web_search → competitor_search → news_search` |
| "Track AI agent research, patents and competitor moves" | `web_search → competitor_search → patent_search → research_search → news_search` |

A tool can also be pulled in **mid-run by an observation** rather than by the goal text. For
example, live web search is held back until one of these happens:

1. another tool returned a `thin` / `empty` result
2. a market signal (launch, funding, acquisition, partnership, regulatory) needs corroborating
3. a tracked company still has zero coverage
4. total relevant evidence is below the reporting threshold

### The 6 agentic components

| Component | Implementation | File |
|---|---|---|
| **Goal** | Natural-language tracking goal → structured information needs | `app/agents/planner.py` |
| **Planning** | Declares *what must be learned*, marks needs required vs. conditional | `app/agents/planner.py` |
| **Reasoning** | Scores relevance, novelty and strategic significance; writes the justification | `app/agents/decision_engine.py` |
| **Tools** | 5 tools over 12 providers, behind one interface with retry + circuit breaker | `app/tools/`, `app/sources/` |
| **Memory** | Per-run dedup (URL identity → normalized-title fingerprint), coverage and signal tracking | `app/agents/state.py` |
| **Action** | Prioritized insights, executive summary, downloadable intelligence report | `app/agents/insight_generator.py`, `app/reports/` |

---

## 👥 Multi-agent architecture

The loop above runs **inside specialist agents** coordinated by an orchestrator. Specialisation is
structural, not cosmetic: each specialist is scoped to a **disjoint** set of tools, so it physically
cannot do the other's job.

| Agent | Owns | Tools | Answers |
|---|---|---|---|
| 🧠 **Intelligence Orchestrator** | Decomposition, delegation, consolidation | — (no tools) | *Who should work on this, and what does it all mean together?* |
| 🔬 **Research Intelligence Agent** | Academic + IP evidence | `research_search`, `patent_search` | *What is technically real and who filed it?* |
| 🏢 **Competitive Intelligence Agent** | Market + company activity | `competitor_search`, `news_search`, `web_search` | *What are competitors actually shipping?* |

```
GOAL → ORCHESTRATOR decomposes into information needs
         │
         ├─ needs research/patent?   → DELEGATE 🔬 Research Agent      → its own ReAct loop
         ├─ needs competitor/news/web? → DELEGATE 🏢 Competitive Agent → its own ReAct loop
         │                                        │
         │        ← recommends a follow-up need ──┘
         ↓
       CONSOLIDATE → cross-agent corroboration + handoffs → PRIORITIZED INSIGHTS
```

**The orchestrator only recruits agents it needs.** A patents-only goal never wakes the
Competitive Agent, so no Tavily call and no competitor sweep happens. Selection is gated on
`research_led` / `market_led` / `company_scoped` intent, and every SELECTED/SKIPPED decision is
written to the activity log with its reason.

**Agents genuinely collaborate**, in two modes:

| Mode | Trigger | Effect |
|---|---|---|
| `corroboration` | Both agents surface the *same event* — shared company or market signal **and** ≥34% title-token overlap | Confidence boost, both source URLs retained, finding marked `corroborated_by` |
| `handoff` | A competitive signal is topically backed by the other agent's technical evidence | Findings linked, no confidence boost |

Every finding carries `discovered_by`, so attribution survives into the API, the dashboard and the
PDF. The activity log is tagged per agent and per event type — `ORCHESTRATION`, `DELEGATION`,
`TOOL_CALL`, `OBSERVATION`, `COLLABORATION`, `RESULT`, `ERROR` — so you can watch the handoffs live.

Built with **no agent framework**. No LangGraph, no CrewAI: the orchestrator is ~500 readable lines,
adds zero dependencies, and streams every state transition instead of hiding it.

---

## 🛠 Intelligence tools

| Tool | Providers | When the agent picks it |
|---|---|---|
| **Research Search** | arXiv, OpenAlex, Semantic Scholar | Scientific/technical progress, new methods, benchmarks |
| **Patent Search** | PatentsView (USPTO), Google Patents | IP posture; competitor-owned filings are flagged |
| **Industry News** | Curated RSS (tier-1), Hacker News, NewsAPI, NewsData.io, GNews | Market context, launches, funding |
| **Competitor Intelligence** | RSS, Hacker News, GitHub, Reddit, NewsAPI, NewsData.io, GNews | Named-company activity, per company |
| **Live Web Intelligence** | Tavily | Current announcements the curated feeds miss |

### Live with no key

`arXiv` · `OpenAlex` · `curated RSS` · `Hacker News` · `GitHub`

### Needs a free key (otherwise clearly labelled `SIMULATED`)

| Key | Get it free | Unlocks |
|---|---|---|
| `PATENTSVIEW_API_KEY` | [patentsview.org/apis/keyrequest](https://patentsview.org/apis/keyrequest) | Real USPTO patent data |
| `SEMANTIC_SCHOLAR_API_KEY` | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#api-key-form) | Citation counts, venues, institutions |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Live open-web search |
| `NEWSAPI_KEY` | [newsapi.org/register](https://newsapi.org/register) | Broad news coverage (~1 month of history on the free plan) |
| `NEWSDATA_API_KEY` | [newsdata.io/register](https://newsdata.io/register) | Global news across ~80k sources, 200+ countries |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | LLM reasoning (see below) |

---

## 🤖 Reasoning engine

Primary is **Google Gemini** (`gemini-3.5-flash`), called over plain HTTPS with native JSON
output. The client verifies the credential on boot, fails over between models, and respects
per-call and per-run latency budgets.

**If no model is available the agent still reasons.** A deterministic rule-based reasoner takes
over planning, tool selection and prioritization, and the UI and report say so explicitly:

> ⚠️ LLM unavailable → continuing with the heuristic reasoner

The system never claims a model was involved when it wasn't.

---

## 📊 Dashboard

Every figure is computed from the actual run — nothing is hardcoded. Where a metric cannot be
derived, the UI shows a limited-data state rather than a fabricated number.

| Panel | Derived from |
|---|---|
| KPI cards | finding counts, mean relevance, competitive signals (deltas vs. previous scan, else `—`) |
| Intelligence Activity chart | findings bucketed by real publication date |
| Emerging Topics | detected signals + recurring title phrases; growth = recent vs. earlier half |
| Sources & Coverage | grouped by real provider, with live/simulated split |
| Connected Intelligence | genuine cross-source chains sharing a company or signal |
| Competitor Intelligence | per-company activity by source type |
| Top Contributors | author/assignee + citation counts, flagged when the source was simulated |

Sections: **Overview · Research · Competitors · Patents · News · Insights · Reports** —
client-side views over one run, so navigation never refetches or re-runs the agent.

**How the AI Agent Worked** is one click away on the dashboard: the full decision trail
(goal → plan → decision → tool → observation → analysis → …), with iteration counts, tool
inputs and timings behind *Show technical details*.

---

## 📄 Intelligence Report

Built from the completed run — no searches are repeated to produce a document.

- **PDF** — 13–16 pages, server-generated (ReportLab), A4, page numbers, light print theme
- **HTML preview** — in-app, print-ready
- **Markdown** and **JSON** exports

Seven sections: Executive Summary · Prioritized Insights · **Agent Contributions** ·
Agent Execution Summary · Detailed Findings · Sources & Coverage · Limitations & Caveats.

*Agent Contributions* documents the orchestrator's agent-selection reasoning (including which
agents were **skipped** and why), each specialist's tools, providers, coverage and confidence, and
every cross-agent collaboration event.

**Source provenance is auditable.** Each insight leads with the actual publisher
(`wsj.com — retrieved via Tavily Web Search`), and Sources & Coverage lists every provider's
operator, access model, exact API endpoint and the real domains harvested.

---

## 🔌 API

```
POST /api/agent/run              run the loop, return the full result
POST /api/agent/run/stream       same, streaming the activity log (SSE)
GET  /api/agent/tools            tool catalog + provider health
GET  /api/agent/runs             recent runs
POST /api/report/generate        build a report from a finished run
GET  /api/report/{id}/preview    print-ready HTML
GET  /api/report/{id}/download/{pdf|md|json}
GET  /health                     capability report
```

```bash
curl -X POST http://localhost:8000/api/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Track AI agents and monitor OpenAI and Anthropic",
       "keywords":["AI agents"],"competitors":["OpenAI","Anthropic"]}'
```

`AGENT_API_TOKEN` is unset by default so the local demo needs no setup. **Set it before exposing
this service on a network** — the run endpoint spends API quota and makes outbound requests.

---

## 🧱 Architecture

```
backend/
├── main.py                    entry point
├── app/
│   ├── main.py                FastAPI app, static hosting
│   ├── config.py              settings; every key optional
│   ├── security.py            input sanitisation, SSRF guard
│   ├── api/
│   │   ├── agent.py           run / stream / tools / runs
│   │   └── report.py          generate / preview / download
│   ├── agents/
│   │   ├── agent.py           host: ReAct primitives + run lifecycle
│   │   ├── orchestrator.py    agent selection, delegation, consolidation
│   │   ├── specialists.py     Research + Competitive agents (scoped tools)
│   │   ├── messages.py        inter-agent message + collaboration types
│   │   ├── planner.py         goal → information needs
│   │   ├── decision_engine.py next-action policy + observation analysis
│   │   ├── insight_generator.py  prioritized insights + summary
│   │   ├── llm.py             Gemini/Anthropic + deterministic fallback
│   │   ├── sanitize.py        prompt-injection defence
│   │   └── state.py           shared agent state
│   ├── tools/                 5 tools + shared signal detection
│   ├── sources/               12 providers + retry/breaker/simulation
│   ├── reports/               builder, HTML, PDF, Markdown
│   └── services/              activity logger (per-agent, typed events)
├── static/                    dashboard (vanilla ES modules, no build step)
└── tests/                     72 tests
```

**Stack:** Python 3.11+ · FastAPI · httpx · ReportLab · vanilla ES modules + hand-rolled SVG
charts. No build step, no frontend framework — the page loads instantly and streaming updates
touch only three DOM nodes.

---

## 🛡 Reliability & safety

- **One failing provider never fails a run.** Per-source retry with jittered backoff, circuit
  breaker, token-bucket rate limiting; degradation is reported in the activity log.
- **Iteration cap** (default 10) with a safe partial summary instead of a crash.
- **Prompt-injection defence** — ingested third-party text is sanitised, delimited and labelled
  as data; a finding containing *"ignore previous instructions"* cannot alter scoring.
- **Credibility ceiling** — unverified forum content can never be rated HIGH on its own.
- **HIGH is scarce** — capped so the priority column carries information.
- **Simulated data is always labelled**, in the UI and in every export.
- **Aggregator noise is filtered at ingestion.** Broad news APIs answer technical queries with
  package-registry pages (`pypi.org/project/agent2win`), job listings and course pages; those are
  dropped rather than down-ranked. Redirect wrappers such as `news.google.com/rss/articles/<blob>`
  are dropped too, since an opaque redirect cannot be cited as a source.
- **Syndication-aware dedup.** Beyond URL identity and a normalized-title fingerprint, a third
  stage compares the opening tokens of a headline, which collapses the same wire story
  republished as `"<headline> - The New York Times"`.

---

## ✅ Test suite

```bash
cd backend && .venv/bin/python -m pytest tests/ -q
# 72 passed
```

Covers goal→plan, dynamic tool selection per goal, observation-driven adaptation, self-termination,
iteration cap, tool-failure containment, dedup, priority bands, the credibility ceiling,
prompt-injection resistance, metric consistency, and the Tavily gating rules.

The multi-agent suite (`tests/test_multi_agent.py`, 28 tests) additionally proves tool scoping is
disjoint and enforced, that a research-only goal never recruits the Competitive Agent, that a
patents-only goal triggers no Tavily call, that both agents are recruited for a mixed goal, that
corroboration requires a genuinely shared event, that findings carry `discovered_by`, that one
specialist failing does not abort the run, and that the pre-existing single-agent result shape is
unchanged.

---

## 📋 Build status

| Area | Status |
|---|---|
| ReAct agent loop (plan → decide → act → observe → analyze → repeat) | ✅ |
| Multi-agent orchestration — 3 agents, scoped tools, real delegation | ✅ |
| Cross-agent collaboration (corroboration + handoff) with per-finding attribution | ✅ |
| Dynamic tool selection, verified per goal | ✅ |
| 5 tools over 12 providers | ✅ |
| Gemini reasoning + deterministic fallback | ✅ |
| Graceful provider degradation + circuit breakers | ✅ |
| Prioritized insights (what happened / why it matters / action) | ✅ |
| Live activity streaming (SSE) | ✅ |
| Intelligence dashboard with derived analytics | ✅ |
| Intelligence Report — PDF / HTML / Markdown / JSON | ✅ |
| Source provenance auditing | ✅ |
| 72 automated tests | ✅ |
| Public deployment | ❌ not yet — runs locally |
| Scheduled autonomous re-runs | ❌ out of scope for the current tasks |
| Multi-user accounts / persistence | ❌ runs are held in memory |
