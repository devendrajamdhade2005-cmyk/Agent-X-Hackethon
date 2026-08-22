"""Pydantic request/response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .security import clean_terms, clean_text

Band = Literal["high", "medium", "low"]
SourceType = Literal["research", "patent", "news", "social", "repo"]

ALL_SOURCE_TYPES: list[str] = ["research", "patent", "news", "social", "repo"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── auth ────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(default="Analyst", max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(ORMModel):
    id: str
    email: str
    name: str
    role: str


# ── profiles ────────────────────────────────────────────────
class ProfileCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    goal_statement: str = Field(default="", max_length=2000)
    keywords: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=lambda: list(ALL_SOURCE_TYPES))
    priority_threshold: Band = "medium"
    interval_minutes: int = Field(default=180, ge=5, le=10080)
    is_active: bool = True
    alert_channels: list[str] = Field(default_factory=lambda: ["in_app"])

    @field_validator("name", "goal_statement")
    @classmethod
    def _clean(cls, v: str) -> str:
        return clean_text(v, max_len=2000)

    @field_validator("keywords", "competitors")
    @classmethod
    def _clean_list(cls, v: list[str]) -> list[str]:
        return clean_terms(v)

    @field_validator("source_types")
    @classmethod
    def _dedupe_sources(cls, v: list[str]) -> list[str]:
        out = [s for s in ALL_SOURCE_TYPES if s in set(v)]
        return out or list(ALL_SOURCE_TYPES)


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    goal_statement: str | None = Field(default=None, max_length=2000)
    keywords: list[str] | None = None
    competitors: list[str] | None = None
    source_types: list[SourceType] | None = None
    priority_threshold: Band | None = None
    interval_minutes: int | None = Field(default=None, ge=5, le=10080)
    is_active: bool | None = None
    alert_channels: list[str] | None = None


class ProfileOut(ORMModel):
    id: str
    name: str
    goal_statement: str
    keywords: list[str]
    competitors: list[str]
    source_types: list[str]
    priority_threshold: str
    interval_minutes: int
    is_active: bool
    alert_channels: list[str]
    weights: dict[str, Any]
    last_run_at: datetime | None
    next_run_at: datetime | None
    created_at: datetime


class ProfileStats(BaseModel):
    profile_id: str
    findings: int = 0
    insights: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    threads: int = 0
    alerts: int = 0
    duplicates_suppressed: int = 0
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    runs: int = 0


# ── findings / insights ─────────────────────────────────────
class FindingOut(ORMModel):
    id: str
    source_type: str
    source_name: str
    title: str
    url: str
    author: str
    published_at: datetime | None
    discovered_at: datetime
    credibility: str
    is_simulated: bool
    meta: dict[str, Any]


class InsightOut(ORMModel):
    id: str
    profile_id: str
    finding_id: str
    run_id: str | None
    summary: str
    why_it_matters: str
    relevance: int
    novelty: int
    significance: int
    score: int
    band: str
    justification: str
    modifiers: list[Any]
    entities: list[Any]
    sentiment: str
    action_hint: str
    status: str
    reasoner: str
    created_at: datetime
    finding: FindingOut


class ThreadOut(ORMModel):
    id: str
    profile_id: str
    title: str
    narrative: str
    implication: str
    score: int
    band: str
    member_finding_ids: list[str]
    source_types: list[str]
    entities: list[str]
    reasoner: str
    created_at: datetime


class ThreadDetail(ThreadOut):
    members: list[FindingOut] = Field(default_factory=list)


# ── runs ────────────────────────────────────────────────────
class RunEventOut(ORMModel):
    seq: int
    node: str
    level: str
    message: str
    data: dict[str, Any]
    elapsed_ms: int
    ts: datetime


class RunOut(ORMModel):
    id: str
    profile_id: str
    trigger: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int
    stats: dict[str, Any]
    degraded_sources: list[Any]
    cost_usd: float
    tokens_in: int
    tokens_out: int
    error: str


class RunDetail(RunOut):
    plan: dict[str, Any] = Field(default_factory=dict)
    events: list[RunEventOut] = Field(default_factory=list)
    profile_name: str = ""


class RunTriggerResponse(BaseModel):
    run_id: str
    profile_id: str
    status: str
    message: str


# ── delivery ────────────────────────────────────────────────
class AlertOut(ORMModel):
    id: str
    profile_id: str
    insight_id: str | None
    thread_id: str | None
    title: str
    body: str
    channel: str
    status: str
    error: str
    read_at: datetime | None
    created_at: datetime


class DigestOut(ORMModel):
    id: str
    profile_id: str | None
    period: str
    title: str
    content_md: str
    stats: dict[str, Any]
    generated_at: datetime


class DigestRequest(BaseModel):
    profile_id: str | None = None
    period: Literal["daily", "weekly"] = "daily"


class BattlecardOut(ORMModel):
    id: str
    profile_id: str
    competitor: str
    content_md: str
    stats: dict[str, Any]
    citations: list[Any]
    reasoner: str
    generated_at: datetime


class FeedbackRequest(BaseModel):
    vote: Literal[-1, 1]
    note: str = Field(default="", max_length=500)


# ── intelligence views ──────────────────────────────────────
class RadarPoint(BaseModel):
    insight_id: str
    title: str
    novelty: int
    significance: int
    score: int
    band: str
    source_type: str
    quadrant: str
    url: str = ""


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str
    weight: int
    is_competitor: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: int
    source_types: list[str] = Field(default_factory=list)


class SignalGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class TrendSeries(BaseModel):
    entity: str
    kind: str
    points: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    velocity: float = 0.0
    zscore: float = 0.0
    spiking: bool = False


class TrendResponse(BaseModel):
    profile_id: str
    window_days: int
    series: list[TrendSeries] = Field(default_factory=list)
    spikes: list[TrendSeries] = Field(default_factory=list)


class CompetitorSummary(BaseModel):
    name: str
    total: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    high: int = 0
    last_activity: datetime | None = None
    momentum: float = 0.0
    top_insight_id: str | None = None
    top_insight_title: str = ""


# ── ask (RAG) ───────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=800)
    profile_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=25)

    @field_validator("question")
    @classmethod
    def _clean_q(cls, v: str) -> str:
        return clean_text(v, max_len=800)


class AskCitation(BaseModel):
    insight_id: str | None = None
    finding_id: str
    title: str
    url: str
    source_type: str
    score: int = 0
    similarity: float = 0.0


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[AskCitation] = Field(default_factory=list)
    reasoner: str
    used_memory_items: int = 0


# ── ops ─────────────────────────────────────────────────────
class SourceHealthOut(ORMModel):
    source: str
    state: str
    consecutive_failures: int
    total_calls: int
    total_failures: int
    p50_ms: int
    last_latency_ms: int
    last_error: str
    forced_failure: bool
    updated_at: datetime


class InjectFailureRequest(BaseModel):
    source: str
    enabled: bool = True


class DashboardOverview(BaseModel):
    profiles: int = 0
    active_profiles: int = 0
    findings: int = 0
    insights: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    threads: int = 0
    unread_alerts: int = 0
    duplicates_suppressed: int = 0
    runs_today: int = 0
    cost_today_usd: float = 0.0
    avg_latency_ms: int = 0
    next_run_at: datetime | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    top_entities: list[dict[str, Any]] = Field(default_factory=list)
    source_mix: dict[str, int] = Field(default_factory=dict)
    score_timeline: list[dict[str, Any]] = Field(default_factory=list)


TokenResponse.model_rebuild()
