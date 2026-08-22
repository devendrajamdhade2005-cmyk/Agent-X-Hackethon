# 🔍 InsightPulse — Autonomous Research & Competitor Intelligence Agent

> An autonomous AI agent that continuously monitors research publications, patents, competitor activities, and industry news — then delivers concise, actionable insights in real time.

---

## 👥 Team Members

- Akash Pingale
- Devendra Jamdhade
- shubham paithankar
- gaurav bodkhe
- shubham sonwane


---

## 📌 Problem Statement

Organizations, startups, and research institutions operate in highly competitive and rapidly evolving environments where staying updated on research trends, patent developments, competitor strategies, and industry news is critical. However, manually monitoring scientific publications, patent databases, news platforms, and social media sources is time-consuming, inefficient, and prone to missing important updates. The lack of timely insights can result in lost opportunities, delayed innovation, and weakened competitive positioning.

Therefore, there is a need for an **autonomous AI agent** capable of continuously tracking research and competitor activities, analyzing vast information sources, and delivering concise, actionable insights in real time.

---

## 📖 Project Description

**InsightPulse** is a fully autonomous AI agent built on a LangGraph-powered agentic loop that continuously gathers intelligence from multiple sources — research papers (arXiv, Semantic Scholar), patent databases, industry news (NewsAPI), and social signals (Reddit) — and converts raw data into prioritized, plain-language insights.

Unlike a basic search tool or information aggregator, InsightPulse exhibits genuine agentic behavior across all 6 required components:

| Agentic Component | Implementation |
|---|---|
| **Goal** | User defines a Tracking Profile: keywords, competitor names, source types, priority threshold |
| **Reasoning** | Claude LLM scores each finding for novelty, relevance, and strategic significance |
| **Planning** | Planner step breaks one Tracking Profile into a concrete per-source query plan |
| **Tools** | arXiv API, Semantic Scholar, Patents Search, NewsAPI, RSS, Reddit API |
| **Memory** | Vector store (Chroma) + PostgreSQL — deduplicates findings, tracks what's been seen |
| **Action** | Writes insights to DB, updates live dashboard via WebSocket, sends alerts when threshold is met |

---

## ✅ Build Status

> **Legend:** ✅ Complete · 🚧 In Progress · ❌ Not Started · ⚠️ Partial / Broken

### 🏗️ Foundation

| Component | Status | Notes |
|---|---|---|
| Project setup / repo structure | 🚧 | — |
| Backend — FastAPI skeleton | 🚧 | — |
| Frontend — Next.js dashboard UI | 🚧 | — |
| PostgreSQL DB schema + models | 🚧 | — |
| Environment / API key config | 🚧 | — |

### 🔌 Data Source Integrations

| Integration | Status | Notes |
|---|---|---|
| arXiv API | 🚧 | — |
| Semantic Scholar API | 🚧 | — |
| NewsAPI / RSS feeds | 🚧 | — |
| Reddit API (via PRAW) | 🚧 | — |
| Patents search (SerpAPI / Lens.org) | 🚧 | — |

### 🤖 Agent Core

| Component | Status | Notes |
|---|---|---|
| LangGraph agent loop (Planner → Tools → Reasoning → Action) | 🚧 | — |
| Claude LLM relevance scoring | 🚧 | — |
| Chroma vector memory + deduplication | 🚧 | — |
| APScheduler autonomous runs | 🚧 | — |

### 📊 Dashboard & Delivery

| Feature | Status | Notes |
|---|---|---|
| Live insight feed (WebSocket updates) | 🚧 | — |
| Insight cards with score + reasoning | 🚧 | — |
| Competitor activity timeline | 🚧 | — |
| Agent Activity Log (live, human-readable) | 🚧 | — |
| Digest generator (daily/weekly PDF/MD) | 🚧 | — |
| In-app alerts | 🚧 | — |
| Email alerts | 🚧 | — |

### 🚀 Deployment

| Component | Status | Notes |
|---|---|---|
| Frontend — Vercel | 🚧 | — |
| Backend — Render / Railway | 🚧 | — |
| Live URL accessible | 🚧 | — |

> **How to update this table:** As each item is completed, change 🚧 to ✅. If something is broken or skipped, mark ❌ or ⚠️ and add a short note.

---

## ⚙️ Technologies Used

| Layer | Technology |
|---|---|
| **Frontend** | Next.js + TypeScript + Tailwind CSS |
| **Backend** | Python 3.11 + FastAPI |
| **Agent Orchestration** | LangGraph |
| **LLM / AI** | Claude (Anthropic API) |
| **Scheduler** | APScheduler |
| **Database** | PostgreSQL (Supabase / Neon) |
| **Vector Memory** | Chroma |
| **Real-time Updates** | WebSockets (FastAPI native) |
| **Research Sources** | arXiv API, Semantic Scholar API |
| **Patent Sources** | Google Patents via SerpAPI, Lens.org |
| **News Sources** | NewsAPI, GNews, Curated RSS |
| **Social Signals** | Reddit API (via PRAW) |
| **Hosting** | Vercel (frontend) + Render / Railway (backend) |

---

## ✨ Features

- 🔍 **Research Tracking** — Fetches and summarizes new papers from arXiv and Semantic Scholar based on tracked topics.
- 📜 **Patent Tracking** — Monitors patent filings and flags when a tracked competitor appears as an assignee.
- 🏢 **Competitor Tracker** — Maintains a per-competitor activity timeline across news, patents, and web sources.
- 📰 **News Monitoring** — Pulls relevant industry news and scores articles for relevance before surfacing them.
- 🤖 **AI Summarization** — Every finding gets a plain-language 2–3 sentence summary and a "why this matters" line.
- 🧠 **Relevance Scoring** — Claude scores each finding High / Medium / Low with a one-sentence reasoning shown to the user.
- 🔁 **Autonomous Loop** — Runs on a configurable schedule without manual triggers.
- 🚨 **Alerts** — In-app and email notifications when a High-priority finding is detected.
- 📊 **Digest Generator** — Compiles a daily/weekly summary of top insights, exportable as PDF or Markdown.
- 🪵 **Agent Activity Log** — Live, human-readable log of every step the agent takes per run.
- 🔄 **Deduplication** — Vector-embedding-based memory prevents re-alerting on already-seen findings.

---

## 🚀 Installation & Setup

### Prerequisites

- Node.js 18+
- Python 3.11+
- PostgreSQL database (Supabase or Neon recommended)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/insightpulse.git
cd insightpulse
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file inside `backend/` using the template below:

```env
# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Database
DATABASE_URL=postgresql://user:password@host:5432/insightpulse

# Semantic Scholar
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

# News
NEWS_API_KEY=your_newsapi_key

# Reddit (via PRAW)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=InsightPulse/1.0

# Patents (SerpAPI)
SERP_API_KEY=your_serpapi_key

# Optional: Email alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 4. Initialize the Database

```bash
cd backend
alembic upgrade head
```

### 5. Set Up the Frontend

```bash
cd ../frontend
npm install
```

Create a `.env.local` file inside `frontend/`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## ▶️ How to Run the Project

### Start the Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### Start the Frontend (new terminal)

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Using the App

1. **Create a Tracking Profile** — Go to *Tracking Profiles* → *New Profile*. Enter topic keywords and competitor names.
2. **Run the Agent** — Click *Run Now*, or wait for the scheduled autonomous run.
3. **View Insights** — Dashboard live-updates with new insight cards as the agent processes findings.
4. **Check the Agent Log** — Open *Agent Activity Log* to see every step the agent took in plain language.
5. **Set Up Alerts** — Go to *Settings* → *Alerts* to enable email notifications for High-priority findings.

---

## 🗂️ Project Structure

```
insightpulse/
├── backend/
│   ├── agents/            # LangGraph agent graph (planner, tools, reasoning, action nodes)
│   ├── tools/             # Individual tool wrappers: arxiv, news, patents, reddit
│   ├── models/            # SQLAlchemy DB models
│   ├── routers/           # FastAPI route handlers
│   ├── scheduler.py       # APScheduler autonomous run loop
│   ├── memory.py          # Chroma vector store for dedup
│   └── main.py            # FastAPI app entrypoint
├── frontend/
│   ├── app/               # Next.js app directory
│   ├── components/        # Dashboard, InsightCard, AgentLog, etc.
│   └── lib/               # API client, WebSocket hooks
├── .env.example
└── README.md
```

