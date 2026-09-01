# 🔍 InsightPulse AI — Autonomous Research & Competitor Intelligence Agent

> An autonomous AI agent that gathers intelligence from research papers, patents, competitor
> activity, industry news and the live web — reasons about what it finds, and delivers a
> prioritized, actionable briefing with a downloadable report.

---

## 👥 Team Members

-Team Leader - Devendra Jamdhade
-Member 1 - Akash Pingale
-Member 2 - Shubham Paithankar
-Member 3 - Gaurav Bodkhe
-Member 4 - Shubham Sonwane

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
| **Tools** | 5 tools over 13 providers, behind one interface with retry + circuit breaker | `app/tools/`, `app/sources/` |
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

The specialised agents and their scoped tools are coordinated by a **LangGraph** `StateGraph`
runtime — see the next section.

---

## ⚙️ Agent framework — LangGraph (Task 5)

### Why LangGraph

> InsightPulse requires a stateful, adaptive, multi-agent orchestration runtime rather than a
> fixed pipeline. LangGraph was selected because its `StateGraph` model provides explicit shared
> state, conditional routing, parallel fan-out, looping workflows, checkpointing/persistence,
> streaming, and fine-grained control over deterministic and LLM-driven steps. This matches the
> project's requirements for dynamic planning, autonomous replanning, memory-based reasoning,
> recovery, and adversarial execution.

LangGraph is the primary orchestration runtime (`app/graph/`). The Task 1–4 components — planner,
specialist agents, tool registry, source resilience and the memory manager — are **reused
unchanged**; LangGraph coordinates them, it does not replace them.

### The graph (dynamic, cyclic — not a fixed pipeline)

```
UNDERSTAND → PLAN → DECOMPOSE → RESOURCE/POLICY → DYNAMIC ROUTER
                                                     │  (conditional fan-out)
                        ┌────────────────────────────┼────────────────────┐
                        ▼                             ▼                     ▼
                 RESEARCH AGENT              COMPETITIVE AGENT          (observer)
                   (tools+patent)              (tools+web)
                        └──────────────┬──────────────┘
                                       ▼
                                   OBSERVER ─→ CONFLICT RESOLUTION ─→ SELF-EVALUATOR
                                                                          │ (conditional)
                              ┌───────────────────────────────────────────┼─────────┐
                              ▼                                            ▼         ▼
                          VERIFY ─→ (back to CONFLICT RESOLUTION)       REPLAN    FINALIZE
                                                                          │          ▼
                                                                   (back to ROUTER)  MEMORY UPDATE → END
```

The next node is chosen from the observed state, evidence, failures, uncertainty and remaining
budget — never a hard-coded order.

| Capability | How |
|---|---|
| **Shared state** | Typed `GraphState` (`state.py`) with order-independent reducers for parallel writes |
| **Dynamic planning** | Planner + goal-driven agent selection; the plan changes with the goal |
| **Adaptive decomposition** | Tasks derived from the plan; small goals stay small |
| **Conditional routing** | `add_conditional_edges` decides verify / replan / finalize from state |
| **Parallel execution** | Router fans out to independent agents in one superstep (reducers merge results) |
| **Checkpointing** | Checkpoint after understand / plan / agents / verification / synthesis / memory; resumable by `thread_id` (in-process `MemorySaver`, durable `SqliteSaver`) |
| **Failure recovery** | Real retry → fallback through the existing resilience layer; failure is data, not a crash |
| **Conflict resolution** | Contradiction detection → credibility/independence evaluation → resolve or verify |
| **Uncertainty** | Every claim carries confidence / evidence strength / verification status |
| **Self-evaluation** | Scores completion, coverage, evidence, confidence; can demand more work |
| **Autonomous replanning** | Replanner adds/removes tasks from observations; the plan version increments |
| **Hypothesis verification** | SUPPORTED / PARTIALLY / UNSUPPORTED / INCONCLUSIVE, never asserted as fact |
| **Resource governance** | Tool/LLM/step/time/cost budgets; low-value work is dropped under pressure |
| **Loop/deadlock detection** | Progress monitor + hard recursion limit (60) independent of LangGraph's own |
| **Memory** | Task 4 memory retrieved at start, updated during the run, consolidated at the end |

### Adversarial live test

A deterministic, repeatable adversarial mode (`adversarial.py`) injects controlled faults; the
graph recovers through the **production** path (nothing about the success is faked):

```
🎯 Plan created  →  🔬 research tool fails  →  🔄 retry  →  ↩ fallback (arXiv→OpenAlex)
🏢 competitive source times out  →  ↩ fallback (Tavily→news)  →  ⚠ evidence conflict detected
🔎 verification task created  →  ✓ independent source consulted  →  💰 budget constraint respected
↻ plan revised (v2)  →  🧭 self-evaluation  →  ✅ objective completed
```

Run it from the dashboard **Framework** tab (**🧪 Run Adversarial Test**) or the API:

```bash
curl -N -X POST http://localhost:8000/api/agent/graph/adversarial \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Analyze AI-agent developments and strategic competitive movement",
       "competitors":["OpenAI","Anthropic"],"simulation_mode":true,"scenario":"full"}'
```

Scenarios: `full`, `tool_failure`, `conflict`, `budget`.

---

## 🛠 Intelligence tools

| Tool | Providers | When the agent picks it |
|---|---|---|
| **Research Search** | arXiv, OpenAlex, Semantic Scholar | Scientific/technical progress, new methods, benchmarks |
| **Patent Search** | Google Patents (SerpApi), PatentsView (USPTO) | IP posture; competitor-owned filings are flagged |
| **Industry News** | Curated RSS (tier-1), Hacker News, NewsAPI, NewsData.io, GNews | Market context, launches, funding |
| **Competitor Intelligence** | RSS, Hacker News, GitHub, Reddit, NewsAPI, NewsData.io, GNews | Named-company activity, per company |
| **Live Web Intelligence** | Tavily | Current announcements the curated feeds miss |

### Live with no key

`arXiv` · `OpenAlex` · `curated RSS` · `Hacker News` · `GitHub`

A `GITHUB_TOKEN` is optional but lifts GitHub from 60 to 5,000 requests/hour.

### Needs a free key (otherwise clearly labelled `SIMULATED`)

| Key | Get it free | Unlocks |
|---|---|---|
| `SERPAPI_KEY` | [serpapi.com](https://serpapi.com) | Real Google Patents data (free plan: 250 searches/month) |
| `PATENTSVIEW_API_KEY` | [patentsview.org/apis/keyrequest](https://patentsview.org/apis/keyrequest) | Adds USPTO grants alongside Google Patents |
| `SEMANTIC_SCHOLAR_API_KEY` | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#api-key-form) | Citation counts, venues, institutions (keyless access is 429-throttled) |
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
POST /api/agent/graph/run        run the LangGraph runtime (Task 5)
POST /api/agent/graph/run/stream stream the framework events (SSE)
POST /api/agent/graph/adversarial run the adversarial demo (SSE)
GET  /api/agent/graph/info       graph topology, for visualisation
GET  /api/evaluation/cases       benchmark dataset + scenario coverage
GET  /api/evaluation/metrics     metric methodology + latest measured values
POST /api/evaluation/run         run an evaluation suite (Task 6)
POST /api/evaluation/run/stream  same, streaming live progress (SSE)
POST /api/evaluation/repeat      repeated-run reliability/consistency
GET  /api/evaluation/baseline    baseline vs InsightPulse comparison
GET  /api/evaluation/history     suite history + regression comparison
POST /api/evaluation/human-review submit reviewer scores
GET  /api/evaluation/report      evaluation report (json | md | html | pdf)
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
│   │   ├── graph.py           LangGraph run / stream / adversarial / info
│   │   ├── evaluation.py      cases / run / metrics / baseline / human review
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
│   ├── memory/
│   │   ├── task_context.py    UNDERSTAND — structured reading of the goal
│   │   ├── working.py         short-term memory, versioning, compression
│   │   ├── context_builder.py selective per-agent context packets
│   │   ├── long_term.py       durable memory + relevance retrieval
│   │   └── manager.py         lifecycle owner the agents talk to
│   ├── evaluation/            evaluation & benchmarking (Task 6)
│   │   ├── schemas.py         cases, thresholds, claims, metric results
│   │   ├── dataset.py         golden benchmark dataset (7 scenario classes)
│   │   ├── metrics.py         metric catalogue: definitions + formulas
│   │   ├── automated.py       12 automated evaluators + claim extraction
│   │   ├── baseline.py        single-pass LLM + fixed-pipeline baselines
│   │   ├── engine.py          execute a case for real, then score it
│   │   ├── runner.py          suites, repeats, baselines, aggregation
│   │   ├── human.py           human review + automated/human comparison
│   │   ├── regression.py      suite-over-suite regression detection
│   │   ├── reports.py         evaluation report export (pdf/md/html/json)
│   │   └── store.py           suite history (+ JSON mirror)
│   ├── graph/                 LangGraph runtime (Task 5)
│   │   ├── state.py           typed shared StateGraph state + reducers
│   │   ├── nodes.py           understand/plan/agents/observe/evaluate/verify/replan
│   │   ├── builder.py         StateGraph assembly + checkpointer
│   │   ├── engine.py          per-run live objects + specialist host shim
│   │   ├── governor.py        resource governor + loop/deadlock detection
│   │   ├── adversarial.py     deterministic fault injection
│   │   └── runner.py          entry point + SSE bridge + result projection
│   ├── observability/         tracing & self-improvement (Task 7)
│   │   ├── schemas.py         span/error/token/trace/diagnosis/plan types
│   │   ├── tracer.py          span hierarchy via ContextVar, token accounting
│   │   ├── instrument.py      node/tool/LLM wrappers (additive, never fails a run)
│   │   ├── providers.py       local store (source of truth) + optional OTLP export
│   │   ├── redaction.py       secret + prompt-content scrubbing at write time
│   │   ├── controlled_failure.py  deterministic injection keyed to one run_id
│   │   ├── analyzer.py        observations + evidence-weighted root-cause diagnosis
│   │   ├── policy.py          versioned, bounded runtime OptimizationPolicy
│   │   ├── improvement.py     propose/apply/revert + before-after acceptance
│   │   └── loop.py            the 8-stage trace→diagnose→improve→verify cycle
│   ├── tools/                 5 tools + shared signal detection
│   ├── sources/               13 providers + retry/breaker/simulation
│   ├── reports/               builder, HTML, PDF, Markdown
│   └── services/              activity logger (per-agent, typed events)
├── static/                    dashboard (vanilla ES modules, no build step)
└── tests/                     201 tests
```

**Stack:** Python 3.11+ (tested on 3.14) · FastAPI · LangGraph · httpx · ReportLab · vanilla ES modules + hand-rolled SVG
charts. No build step, no frontend framework — the page loads instantly and streaming updates
touch only three DOM nodes.

---

## 🧠 Context & memory

Three layers, deliberately separated (`app/memory/`):

| Layer | Lifetime | What it holds |
|---|---|---|
| **TaskContext** | one run | the structured reading of the goal: topics, research focus, competitors, entities, requested domains, time scope, constraints, continuation |
| **WorkingMemory** | one run | goal + plan state + the findings that mattered + decisions + coverage gaps + open questions, with a version counter |
| **LongTermStore** | across runs | what earned persistence, retrieved by relevance when a later run looks related |

```
USER GOAL
  → UNDERSTAND        task context extracted
  → RETRIEVE          relevant long-term memory (or an honest "none found")
  → PLAN              execution plan stored as tracked steps
  → DELEGATE          agent receives context built for *it*
  → USE TOOLS         real intelligence gathered
  → MEMORY UPDATE     important findings retained, version bumped
  → ORCHESTRATOR      reads updated memory, decides what happens next
  → FOLLOW-UP         relevant context shared with the next agent
  → CROSS-ANALYSIS    combined evidence, compared with baseline if one exists
  → CONSOLIDATE       select what deserves to outlive the run
```

**Context sharing is selective.** `ContextBuilder` assembles a packet per (agent,
objective) rather than handing everyone the run history. The Research Agent gets research
framing; the Competitive Agent gets tracked companies *plus the specific research findings
with competitive bearing*. Every packet records what was withheld and why, and the UI
renders that record — so the per-agent "context received" list is derived from the real
construction, not hardcoded.

**Shared context changes behaviour.** Terms drawn from the shared findings are passed into
the receiving agent's `DecisionEngine` as search focus, so a follow-up query is shaped by
what an earlier agent found rather than repeating the original goal. A run where the
Research Agent surfaced *Helios Dynamics* sends the Competitive Agent looking for
`['Helios Dynamics', 'multi-agent AI']`.

**An observation can create work.** When the Research Agent finds evidence with commercial
bearing — a named company, or a launch/funding/acquisition/partnership signal — it flags
`potential_competitive_relevance`. That flag is *stored in working memory*; the orchestrator
reads it back and decides a competitive check is now justified. So a goal that named no
companies can still trigger company monitoring, driven by evidence rather than phrasing.
A benchmark result deliberately does not count: normal academic output should not launch a
competitor sweep.

**Long-term memory is selective and honest.** Only items at or above HIGH importance are
persisted; transient errors, duplicates, low-relevance hits and *all simulated data* are
rejected at the door, so a keyless demo can never manufacture a fake history. Retrieval
scores topic + entity + competitor overlap, importance, recency and recurrence, with a floor
that keeps unrelated memory out — a quantum-computing goal retrieves nothing from AI-agent
history. `"Continue monitoring this"` carries no subject, so its topics and tracked
companies are restored from memory.

Historical comparison classifies findings as **NEW / PREVIOUSLY KNOWN / TREND ACCELERATING**,
and only ever against a baseline that was actually retrieved.

Storage note: this project has no database, and memory that died with the process would not
be long-term, so the store keeps the existing module-global bounded-`OrderedDict` shape and
adds an atomically-written JSON file under `DATA_DIR`. No new dependency, no new service.

**Memory never breaks a run.** Retrieval failure leaves the run on current context;
persistence failure leaves the completed intelligence intact; malformed persisted records are
quarantined individually. All three are reported honestly rather than hidden.

---

## ⚡ Performance

Providers are queried **concurrently**, so a tool costs its slowest provider instead of the sum of
all of them. Measured on live APIs:

| | Before | After |
|---|---|---|
| Research fan-out (3 providers) | 3.0s | 2.3s |
| News fan-out (5 providers) | 4.1s | 2.0s |
| Competitor sweep (2 companies × 3 groups) | 7.2s | 1.9s |
| Research goal, end to end | 6.5s | 3.2s |
| Mixed goal, end to end | 10.6s | 5.9s |

The competitor sweep gained the most because it fans out twice — once per company, then once per
provider group — and both levels were sequential.

Results are folded back in declared provider order, so `providers_used` and dedup order stay
deterministic rather than following whichever provider answered first (verified identical across
repeated runs).

A connector can also lower its own retry ceiling via `max_attempts`. Semantic Scholar answers 429
often even with a key, and under a concurrent fan-out its retries set the latency for the whole
research tool, so it retries once instead of twice.

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
- **Real evidence outranks simulated evidence.** Placeholder records are synthesised to look
  ideal — exactly on-topic, dated today — so they beat genuine findings on relevance and recency.
  Simulated findings therefore carry an explicit scoring penalty, and the patent tool ranks live
  filings above placeholders. Without this, a run with one live and one keyless provider builds its
  insights from the placeholders while real data sits below the cut.

---

## 📊 Evaluation & benchmarking (Task 6)

An evaluation layer (`app/evaluation/`) measures the quality of the real agent rather than
asserting it. It is distinct from the Task 5 self-evaluator:

| | Question | When |
|---|---|---|
| Task 5 evaluator | "what should the agent do next?" | online, steers routing |
| Task 6 evaluation | "how good was that performance?" | offline, scores quality |

Every score comes from an actual InsightPulse execution. Where a metric cannot be measured
from the available data it reports **unavailable with a reason** — never a fabricated number.

### Pipeline

```
Evaluation case → real agent execution (Tasks 1–5) → capture findings, evidence,
execution record, failures/recovery, timing/resources → automated evaluators →
metric calculation → optional human review → baseline comparison → PASS/PARTIAL/FAIL →
suite aggregation → regression history → report export
```

### Benchmark dataset

11 cases (`EVAL-001`…`EVAL-011`) covering all seven scenario classes: **NORMAL** (5),
**AMBIGUOUS**, **ADVERSARIAL**, **CONTRADICTORY**, **INCOMPLETE**, **TOOL_FAILURE**,
**UNSUPPORTED_CONCLUSION**. The demo suite includes one case per class, so a single run
populates every row of the scenario matrix. The suite runs in deterministic simulation mode
so repeated evaluation is stable and offline.

Repeated runs share a stable `case_id` and differ only by `repeat_index`, each keeping its own
`evaluation_run_id`. Reliability and consistency are grouped by `case_id` — grouping by run id
would place every repetition in its own bucket and make both metrics unmeasurable.

Ground truth is **structural and checkable** (entities that must be covered, source
categories that must be reached, subtasks that must be performed), not invented real-world
facts. The method is reported alongside the score so it is never mistaken for live fact-checking.

### Metric methodology

| Metric | Definition | Formula |
|---|---|---|
| **Accuracy** | Agreement with the case's checkable ground truth | F1 of ground-truth recall and evidence-linked insight precision |
| **Task completion** | Required subtasks actually performed, verified from execution evidence (never an HTTP status) | `completed_subtasks / required_subtasks` |
| **Reliability** | Repeated executions that completed successfully | `successful_runs / total_runs` |
| **Robustness** | Performance across scenario classes | unweighted mean of per-category scores |
| **Evidence quality** | Credibility, relevance, recency, independence, corroboration | `0.30·cred + 0.25·rel + 0.15·recency + 0.15·independence + 0.15·corrob` |
| **Efficiency** | Useful output per unit of work | normalised relevant-findings-per-tool-call |
| **Groundedness** | Factual claims supported by collected evidence | `(supported + 0.5·partial) / factual_claims` |
| **Hallucination rate** | Factual claims with no sufficient evidence | `unsupported_factual / factual_claims` |
| **Recovery rate** | Injected failures genuinely recovered | `recovered / injected` |
| **Consistency** | Substantive run-to-run agreement (not wording) | `0.35·findings + 0.25·conclusions + 0.20·priorities + 0.10·confidence + 0.10·completion` |
| **Latency** | Wall-clock, with stage breakdown | mean / median / p95 (≥5 samples) / min / max |
| **Resource efficiency** | Cost of a completed task | `1 − (tool_calls / ceiling)`, zero if incomplete |
| **Uncertainty handling** | Was expressed certainty calibrated to evidence strength | calibration match vs. case expectation |
| **Unsupported-conclusion rate** | Conclusions asserted without support | `asserted / opportunities` (correct refusal = 0) |

Claim classification is the backbone of groundedness: each insight carries a `finding_id`, so a
factual claim is grounded only when that link resolves to a finding the run actually collected
**and** the claim's content appears in it. `recommended_action` (an action, not an assertion) and
labelled hypotheses are deliberately **excluded** from factual scoring, and hedged statements
count as explicit uncertainty rather than hallucination.

### Outcomes and quality gates

`PASS` — all configured gates met · `PARTIAL` — soft gate missed, no critical breach ·
`FAIL` — a **critical** gate breached (hallucination above ceiling, groundedness below floor,
an unsupported conclusion asserted, a required recovery not achieved, or an injected
contradiction missed) · `ERROR` — execution raised. A single excellent metric can never mask a
critical failure. Thresholds are configurable and stored with each suite.

### Baselines

* **Baseline A — single-pass LLM**: one model call, no tools, no evidence links.
* **Baseline B — fixed pipeline**: the project's pre-LangGraph classic agent, same tools and
  fixtures, but no dynamic replanning, verification or conflict resolution.

Baselines run the same cases wherever that is fair; fault-injection cases are excluded (a system
with no tool layer cannot be fairly subjected to tool failure) and the exclusion is reported.

### Run it

```bash
# dashboard: Evaluation tab → 🧪 Run Evaluation Suite
curl -X POST http://localhost:8000/api/evaluation/run \
  -H 'Content-Type: application/json' -d '{"mode":"demo","include_baseline":true}'

curl http://localhost:8000/api/evaluation/metrics          # methodology + latest values
curl "http://localhost:8000/api/evaluation/report?format=pdf" -o evaluation.pdf
curl "http://localhost:8000/api/evaluation/report?format=md"  -o evaluation.md
```

The evaluation report exports as **PDF, Markdown, HTML or JSON** and contains the executive
summary, methodology, scenario coverage, per-case results, baseline comparison, human review,
failures, uncertainty and recovery cases, regression history and recommendations. The PDF reuses
the existing report engine's document template and styles rather than adding a second one.

**No model is used as a judge.** Every automated score is a checkable computation over the run's
own evidence, so it is reproducible and traceable to the data that produced it — and it does not
depend on provider availability. Semantic judgement comes from the human review layer, where it is
attributed to a reviewer and its disagreement with the automated score is shown, not hidden.

Modes: `demo`, `full`, `adversarial`, `single`, `repeated`, `scenario`.

---

## 🔍 Advanced tracing & observability (Task 7)

Tracing is only worth the effort if it changes something. This layer records every agent run in
full detail, then closes the loop:

```
TRACE → UNDERSTAND WHAT FAILED → DIAGNOSE ROOT CAUSE → CHOOSE A SAFE IMPROVEMENT
      → APPLY IT → RE-RUN THE SAME SCENARIO → MEASURE → VERIFY IT ACTUALLY IMPROVED
```

The system does not stop at "trace collected". It diagnoses, fixes, re-runs and then **proves or
disproves** its own improvement — and rolls the change back when the evidence does not support it.

### Why an internal tracer rather than LangSmith / Langfuse / OpenTelemetry

| Option | Why not |
|---|---|
| **LangSmith / Langfuse** | The dashboard would live in a third-party product. The brief requires the *application* to explain its own failures, so the diagnosis and improvement loop must read traces the app owns. A vendor SDK also adds an account and network dependency to a demo that must run offline. |
| **OpenTelemetry SDK** | The right answer at scale, but without a collector deployed it adds a dependency and no capability. Its *data model* is the good part, so that is what we borrowed. |

The chosen design is a small internal tracer (`app/observability/`) with a **provider abstraction**,
so the storage decision is not baked in:

* `LocalTraceProvider` — always on, in-memory (40 traces) with a JSON mirror on disk. Source of
  truth. Recorded **before** any export, so a failed export can never lose a trace.
* `ExternalTraceProvider` — optional, OTLP-shaped, fire-and-forget. Enable it to mirror spans to a
  collector, Langfuse or LangSmith gateway. If it fails the run is unaffected and the trace reports
  `"external telemetry unavailable — local trace retained"`.

Zero new dependencies were added.

### Span hierarchy

14 span kinds (`run`, `orchestrator`, `node`, `agent`, `decision`, `llm`, `tool`, `provider`,
`retry`, `fallback`, `memory`, `evaluation`, `verification`, `synthesis`). Nesting uses a
`ContextVar` stack, so parallel agents each keep their own branch instead of racing for a parent.
A real recorded trace (36 spans, zero orphans):

```
[run] agent_run 1312ms
  ├─ [node] understand 747ms
  │    └─ [llm] llm:task_context 739ms  degraded → heuristic reasoner
  ├─ [node] plan · decompose        [decision] resource_check · dispatch
  ├─ [agent] research_agent 493ms
  │    └─ [tool] research_search 482ms
  │         ├─ [provider] arxiv               attempts=0    0ms   ok
  │         ├─ [provider] openalex            attempts=0    0ms   ok
  │         └─ [provider] semantic_scholar    attempts=2  481ms   error  backoff=480ms
  ├─ [agent] competitive_agent 5ms → [tool] web_search → [provider] tavily
  ├─ [node] observer · conflict_resolution   [decision] self_evaluator · replan
  ├─ [synthesis] finalize
  └─ [memory] memory_update 4ms
```

Each span carries timing, status, attributes and events. Errors are separate records with a
category, component, HTTP status, retry count, recovery status and an `injected` flag.

### What is measured, and what is honestly not

Token usage is reported as **`measured`** or **`unavailable` with a reason**. A provider that
reported nothing is never recorded as "0 tokens" — that would be a fabricated measurement. The same
rule applies to cost. In the demo below the Gemini quota was exhausted, so the trace says so
verbatim rather than inventing a number.

Retry cost is measured, not estimated: the retry loop accumulates its real `asyncio.sleep` time into
`retry_wait_ms`, so "480ms of backoff" is a reading, not a model.

### Controlled failure

A deterministic fault can be armed against any of the 13 registered providers
(`rate_limit`, `timeout`, `server_error`, `bad_response`). It is **keyed to a single `run_id`**, so
arming one cannot leak into a concurrent normal run. The injected error is a real
`SourceError(status=429, retryable=True)` raised *inside* the production retry loop, so the retries,
backoff, circuit breaker and fallback that follow are all genuine.

### The improvement engine — configuration, never code

Improvements change a **versioned runtime `OptimizationPolicy`**, clamped to declared bounds
(`retry_attempts` 1–5, `timeout_seconds` 3–40). No source file is ever written — a test byte-compares
`resilience.py`, `runner.py` and `policy.py` across a full propose → apply → revert cycle to prove it.
Every version is reversible, and a rejected improvement is rolled back automatically.

### Acceptance is a gate, not a formality

A change is kept only when the targeted metric improves **and** no quality metric regresses. Quality
is judged by the **Task 6 evaluators**, not by this module's own opinion. Verdicts:

`IMPROVEMENT_VERIFIED` · `IMPROVEMENT_REJECTED` (quality regressed, errors rose, or task success
fell → auto-revert) · `NO_MATERIAL_CHANGE` (gain did not clear the noise floor) ·
`NOT_MEASURABLE` · `NO_SAFE_IMPROVEMENT` · `NO_DIAGNOSIS`

Metric direction is **declared** per metric, never inferred from the sign of a difference.

**The noise floor is measured, not assumed.** A single before/after pair cannot tell a real gain from
run-to-run variance. Measuring the same scenario six times with no change applied gave a 204ms
spread and 81ms standard deviation — so a fixed 50ms threshold would happily sell variance as an
improvement. Each side is therefore run three times (`repeats`, 1–5), compared on **medians**, and
the gain must exceed the larger of 50ms and the spread the workload actually showed:

```
sampling  : 3x/side  before=[8998, 8999, 9001]  after=[8999, 9002, 9004]
            medians 8999 -> 9002ms   observed noise 5ms
floor used: 50ms
VERDICT   : NO_MATERIAL_CHANGE   -> policy reverted to v1
```

This is deliberately conservative: it will occasionally reject a real but small gain rather than
risk claiming one that isn't there. `sampling.observed_noise_ms` and `comparison.noise_floor_used`
are reported on every cycle so the verdict can be audited.

### Verified demo run

Real output from `POST /api/observability/improve` (`cyc-1e9341489a`, simulation mode):

| Stage | Result |
|---|---|
| Trace | 36 spans, 3 errors |
| Understand | 3 errors, 1 provider with spent retries |
| Diagnose | `EXCESSIVE_RETRY` on `semantic_scholar` — confidence **97%** |
| Choose | `retry_attempts[semantic_scholar]` 2 → 1 |
| Apply | runtime policy **v0 → v1** |
| Re-run | same scenario, `tr-47dcf2aeef8147e7` |
| Measure | scored by the Task 6 evaluators |
| Verify | **accepted** — latency improved 984ms with no quality regression |

Evidence behind the diagnosis (generated, not written by hand):

> 2 rate-limit responses (HTTP 429) from semantic_scholar · the provider failed on 2 consecutive
> attempts, so the retries did not recover it · 1 retry attempt beyond the first was spent across 1
> failed call and still did not recover the provider · 480ms of that was backoff waiting between
> attempts, measured from the retry loop · semantic_scholar accounted for 37% of total run time ·
> every failure was classified retryable, so the retry policy governed the behaviour

Measured before / after:

| Metric | Before | After | |
|---|---|---|---|
| **duration_ms** (target) | 1312 | **328** | improved −984ms |
| retries | 1 | 0 | improved |
| findings | 29 | 30 | improved |
| groundedness | 1.00 | 1.00 | held |
| task_completion | 1.00 | 1.00 | held |
| evidence_quality | 0.5897 | 0.5899 | held |
| errors | 3 | 3 | unchanged |
| Task 6 outcome | **PASS** | **PASS** | held |

`errors` staying at 3 is the honest result: the provider is still rate-limited. The improvement
removes the *wasted retry latency*, not the upstream 429s.

**The diagnosis depends on what else went wrong in the run.** In the capture above the retry
evidence dominated, so the verdict was `EXCESSIVE_RETRY` at 97% confidence. On a run where the
Gemini quota was also exhausted, the same scenario reported:

```
root cause: MULTIPLE_POSSIBLE_CAUSES on semantic_scholar @ 60%
latency   : 1098ms -> 334ms (-764ms)   Task 6: PASS -> PASS
```

That is the tie-break rule working, not a defect: two explanations (the rate limit and the model
failure) fitted the evidence within 0.15 confidence of each other, so the analyzer refuses to
present one as settled, caps confidence at 60% and flags `validation_required`. It still proposes the
strongest candidate's fix and then *measures* whether it helped — which is the point of validating
empirically rather than trusting the diagnosis.

**Absolute latencies inside a controlled-failure cycle are higher than a normal run.** Arming a
fault routes the target provider through the real request path, so its genuine rate limiter applies
for the rest of the run — `semantic_scholar` allows 20/min, which is one token every 3s. That
inflates before *and* after equally, so the comparison stays fair; it is why a cycle can report
`11999ms → 9000ms` where an unarmed run of the same goal takes about a second.

Running the cycle a second time correctly refuses to invent another change:

```
verdict: NO_SAFE_IMPROVEMENT
choose  rejected  No change available: semantic_scholar already retries only 1 time(s),
                  so there is no retry latency left to remove.
```

### Run it

```bash
# dashboard: Observability tab → 🔬 Run Improvement Cycle   (streams stage by stage)

curl -X POST http://localhost:8000/api/observability/improve \
  -H 'Content-Type: application/json' \
  -d '{"target_source":"semantic_scholar","failure_type":"rate_limit","failure_count":2,"repeats":3}'

curl http://localhost:8000/api/observability/traces              # recent traces
curl http://localhost:8000/api/observability/traces/{id}/tree    # hierarchical timeline
curl http://localhost:8000/api/observability/root-cause/{id}     # diagnosis + evidence
curl http://localhost:8000/api/observability/policy              # active policy + versions
curl -X POST http://localhost:8000/api/observability/policy/reset # back to shipped defaults
```

17 endpoints under `/api/observability`, including an SSE stream at `/improve/stream`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OBSERVABILITY_ENABLED` | `true` | Master switch. Off → runs execute untraced, unaffected. |
| `TRACE_EXPORT_ENABLED` | `false` | Mirror spans to an external collector. |
| `TRACE_EXPORT_ENDPOINT` | `""` | OTLP-shaped HTTP endpoint. |
| `TRACE_EXPORT_API_KEY` | `""` | Sent as a bearer token. Never appears in a trace or in `/status`. |
| `TRACE_PROJECT` | `insightpulse` | Project label attached to exported traces. |

### Safety

Every attribute passes a redaction filter at **write** time, so a secret never reaches memory in the
first place. API keys, tokens and bearer credentials are replaced; prompt text, system prompts and
chain-of-thought are reduced to `<omitted: N chars>` — the *shape* of a prompt is observable, its
content is not. Instrumentation is wrapped so it can never fail a run: a broken span is counted in
`instrumentation_failures` and the run continues.

Normal mode is untouched. With no fault armed, a run records `optimization_version: 0`, zero
injected errors, and produces exactly what it did before this task.

---

## ✅ Test suite

```bash
cd backend && .venv/bin/python -m pytest tests/ -q
# 201 passed  (111 core + 19 LangGraph framework + 28 evaluation + 43 observability)
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
| Working (short-term) memory, updated after every step | ✅ |
| Selective per-agent context construction with recorded omissions | ✅ |
| Observation-driven follow-up read back from memory | ✅ |
| Long-term memory: selective persistence + relevance retrieval | ✅ |
| Historical change detection (NEW / PREVIOUSLY KNOWN / TREND ACCELERATING) | ✅ |
| Cross-agent collaboration (corroboration + handoff) with per-finding attribution | ✅ |
| Dynamic tool selection, verified per goal | ✅ |
| 5 tools over 13 providers | ✅ |
| Gemini reasoning + deterministic fallback | ✅ |
| Graceful provider degradation + circuit breakers | ✅ |
| Prioritized insights (what happened / why it matters / action) | ✅ |
| Live activity streaming (SSE) | ✅ |
| Intelligence dashboard with derived analytics | ✅ |
| Intelligence Report — PDF / HTML / Markdown / JSON | ✅ |
| Source provenance auditing | ✅ |
| End-to-end tracing — 14 span kinds, agent/decision/prompt/tool/token/error spans | ✅ |
| Deterministic controlled failure through the real retry loop | ✅ |
| Automatic root-cause diagnosis with evidence-weighted confidence | ✅ |
| Versioned, reversible improvement engine (runtime config, never code rewrite) | ✅ |
| Same-scenario re-run with Task 6 validated before/after and accept/reject | ✅ |
| 201 automated tests | ✅ |
| Public deployment | ❌ not yet — runs locally |
| Scheduled autonomous re-runs | ❌ out of scope for the current tasks |
| Multi-user accounts / persistence | ❌ runs are held in memory |
