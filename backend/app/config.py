"""Central configuration. Everything is optional — the app degrades, never breaks."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── core ────────────────────────────────────────────────
    app_env: str = "development"
    app_name: str = "InsightPulse"
    app_version: str = "2.0.0"
    secret_key: str = "dev-only-change-me"
    database_url: str = "sqlite:///./data/insightpulse.db"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    access_token_ttl_minutes: int = 60 * 24 * 7

    # ── llm ─────────────────────────────────────────────────
    # auto → gemini if GEMINI_API_KEY is set, else anthropic, else heuristic.
    llm_provider: str = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    llm_run_budget_usd: float = 0.50
    llm_daily_budget_usd: float = 5.00
    llm_max_tokens: int = 2048

    # ── agent ───────────────────────────────────────────────
    simulation_mode: bool = False
    default_interval_minutes: int = 180
    scheduler_enabled: bool = True
    max_results_per_source: int = 12
    collect_timeout_seconds: float = 25.0
    near_duplicate_threshold: float = 0.88
    embedding_dim: int = 512
    embedding_provider: str = "hashing"
    max_findings_reasoned_per_run: int = 40

    # ── source keys ─────────────────────────────────────────
    semantic_scholar_api_key: str = ""
    newsapi_key: str = ""
    gnews_api_key: str = ""
    newsdata_api_key: str = ""
    serpapi_key: str = ""
    patentsview_api_key: str = ""
    tavily_api_key: str = ""
    github_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    # Reddit rejects terse user-agents with 403. A descriptive one is accepted on
    # api.reddit.com without OAuth, which is what keeps this source keyless.
    reddit_user_agent: str = "python:insightpulse.agent:2.0 (research intelligence agent)"

    # ── embeddings ──────────────────────────────────────────
    voyage_api_key: str = ""
    openai_api_key: str = ""

    # ── alerts ──────────────────────────────────────────────
    alert_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "insightpulse@localhost"
    smtp_starttls: bool = True

    # ── api access ──────────────────────────────────────────
    # Empty = open (local demo). Set this before exposing the service publicly:
    # /api/agent/run spends LLM quota and makes outbound requests on demand.
    agent_api_token: str = ""

    # ── demo ────────────────────────────────────────────────
    seed_on_startup: bool = True
    demo_user_email: str = "analyst@insightpulse.dev"
    demo_user_password: str = "insightpulse"

    @field_validator("embedding_provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = (v or "hashing").strip().lower()
        return v if v in {"hashing", "voyage", "openai"} else "hashing"

    @field_validator("secret_key")
    @classmethod
    def _strong_key(cls, v: str) -> str:
        """HS256 wants >=32 bytes. Stretch short/dev keys deterministically."""
        v = (v or "dev-only-change-me").strip()
        if len(v.encode()) >= 32:
            return v
        return hashlib.sha256(f"insightpulse::{v}".encode()).hexdigest()

    # ── derived ─────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        raw = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return raw or ["*"]

    @property
    def active_llm_provider(self) -> str:
        chosen = (self.llm_provider or "auto").strip().lower()
        if chosen == "auto":
            if self.gemini_api_key.strip():
                return "gemini"
            if self.anthropic_api_key.strip():
                return "anthropic"
            return "none"
        if chosen == "gemini" and self.gemini_api_key.strip():
            return "gemini"
        if chosen == "anthropic" and self.anthropic_api_key.strip():
            return "anthropic"
        return "none"

    @property
    def active_llm_model(self) -> str:
        return {
            "gemini": self.gemini_model,
            "anthropic": self.anthropic_model,
        }.get(self.active_llm_provider, "insightpulse-heuristic-v2")

    @property
    def llm_enabled(self) -> bool:
        """True when a real model can be called."""
        return self.active_llm_provider != "none"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def resolved_database_url(self) -> str:
        """Make relative SQLite paths absolute so CWD never matters."""
        url = self.database_url
        if url.startswith("sqlite:///./"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{DATA_DIR / url.split('./', 1)[1].split('/')[-1]}"
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            tail = url.replace("sqlite:///", "", 1)
            if not tail.startswith("/"):
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                return f"sqlite:///{DATA_DIR / Path(tail).name}"
        return url

    def source_credentials(self) -> dict[str, str]:
        return {
            "semantic_scholar": self.semantic_scholar_api_key,
            "newsapi": self.newsapi_key,
            "gnews": self.gnews_api_key,
            "newsdata": self.newsdata_api_key,
            "serpapi": self.serpapi_key,
            "patentsview": self.patentsview_api_key,
            "tavily": self.tavily_api_key,
            "github": self.github_token,
            "reddit": self.reddit_client_id,
        }

    def capability_report(self) -> dict[str, object]:
        """Honest, user-facing summary of what is live vs. simulated."""
        return {
            "simulation_mode": self.simulation_mode,
            "llm": {
                "provider": self.active_llm_provider
                if self.llm_enabled
                else "heuristic-fallback",
                "model": self.active_llm_model,
                "live": self.llm_enabled,
            },
            "embeddings": self.embedding_provider,
            # Verified working with no credentials at all.
            "keyless_sources": [
                "arxiv",
                "openalex",
                "rss",
                "hackernews",
                "reddit",
                "github",
            ],
            "keyed_sources": {
                "semantic_scholar_key": bool(self.semantic_scholar_api_key),
                "newsapi": bool(self.newsapi_key),
                "gnews": bool(self.gnews_api_key),
                "newsdata": bool(self.newsdata_api_key),
                "patentsview": bool(self.patentsview_api_key),
                "tavily_web_search": bool(self.tavily_api_key),
                "serpapi_patents": bool(self.serpapi_key),
                "github_token": bool(self.github_token),
            },
            "alerts": {
                "in_app": True,
                "webhook": bool(self.alert_webhook_url),
                "email": bool(self.smtp_host),
            },
        }


@lru_cache
def get_settings() -> Settings:
    # Allow tests to point at a scratch DB without touching .env
    override = os.environ.get("INSIGHTPULSE_TEST_DB")
    s = Settings()
    if override:
        s = Settings(database_url=override, seed_on_startup=False, scheduler_enabled=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
