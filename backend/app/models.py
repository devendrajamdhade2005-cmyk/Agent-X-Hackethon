"""SQLAlchemy models — the full InsightPulse data model.

JSON columns are used for list/dict fields so the identical schema runs on both
SQLite (hackathon / local) and PostgreSQL (deployed) with no migration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


# ─────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────
class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), default="Analyst")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="analyst")
    settings: Mapped[dict] = mapped_column(JSON, default=dict)

    profiles: Mapped[list["TrackingProfile"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ─────────────────────────────────────────────────────────────
# Goal
# ─────────────────────────────────────────────────────────────
class TrackingProfile(Base, TimestampMixin):
    """The agent's GOAL object."""

    __tablename__ = "tracking_profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    goal_statement: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    competitors: Mapped[list] = mapped_column(JSON, default=list)
    source_types: Mapped[list] = mapped_column(JSON, default=list)
    priority_threshold: Mapped[str] = mapped_column(String(16), default="medium")
    interval_minutes: Mapped[int] = mapped_column(Integer, default=180)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Learned scoring weights from user feedback (see intel/learning.py)
    weights: Mapped[dict] = mapped_column(JSON, default=dict)
    alert_channels: Mapped[list] = mapped_column(JSON, default=list)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="profiles")
    findings: Mapped[list["Finding"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


# ─────────────────────────────────────────────────────────────
# Raw evidence
# ─────────────────────────────────────────────────────────────
class Finding(Base):
    __tablename__ = "findings"
    __table_args__ = (
        UniqueConstraint("profile_id", "canonical_key", name="uq_finding_profile_key"),
        Index("ix_finding_profile_discovered", "profile_id", "discovered_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("tracking_profiles.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    source_type: Mapped[str] = mapped_column(String(32), index=True)  # research|patent|news|social|repo
    source_name: Mapped[str] = mapped_column(String(64))              # arxiv|patentsview|reddit...
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, default="")
    canonical_key: Mapped[str] = mapped_column(String(128), index=True)
    simhash: Mapped[str] = mapped_column(String(20), index=True, default="0")
    author: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    credibility: Mapped[str] = mapped_column(String(16), default="standard")  # high|standard|low|unverified
    sanitization: Mapped[dict] = mapped_column(JSON, default=dict)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)

    profile: Mapped[TrackingProfile] = relationship(back_populates="findings")
    insight: Mapped["Insight | None"] = relationship(
        back_populates="finding", cascade="all, delete-orphan", uselist=False
    )


# ─────────────────────────────────────────────────────────────
# Reasoning output
# ─────────────────────────────────────────────────────────────
class Insight(Base):
    __tablename__ = "insights"
    __table_args__ = (Index("ix_insight_profile_score", "profile_id", "score"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), index=True, unique=True
    )
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    summary: Mapped[str] = mapped_column(Text, default="")
    why_it_matters: Mapped[str] = mapped_column(Text, default="")
    relevance: Mapped[int] = mapped_column(Integer, default=0)      # 0-100
    novelty: Mapped[int] = mapped_column(Integer, default=0)
    significance: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[int] = mapped_column(Integer, default=0, index=True)
    band: Mapped[str] = mapped_column(String(8), default="low", index=True)  # high|medium|low
    justification: Mapped[str] = mapped_column(Text, default="")
    modifiers: Mapped[list] = mapped_column(JSON, default=list)
    entities: Mapped[list] = mapped_column(JSON, default=list)
    sentiment: Mapped[str] = mapped_column(String(16), default="neutral")
    action_hint: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="feed")  # alerted|feed|logged
    reasoner: Mapped[str] = mapped_column(String(48), default="heuristic")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    finding: Mapped[Finding] = relationship(back_populates="insight")


class SignalThread(Base):
    """Cross-source synthesis: several findings that tell one story."""

    __tablename__ = "signal_threads"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    title: Mapped[str] = mapped_column(Text)
    narrative: Mapped[str] = mapped_column(Text, default="")
    implication: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[int] = mapped_column(Integer, default=0, index=True)
    band: Mapped[str] = mapped_column(String(8), default="medium")
    member_finding_ids: Mapped[list] = mapped_column(JSON, default=list)
    source_types: Mapped[list] = mapped_column(JSON, default=list)
    entities: Mapped[list] = mapped_column(JSON, default=list)
    reasoner: Mapped[str] = mapped_column(String(48), default="heuristic")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


# ─────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────
class Embedding(Base):
    __tablename__ = "embeddings"

    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), primary_key=True
    )
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    dim: Mapped[int] = mapped_column(Integer, default=512)
    provider: Mapped[str] = mapped_column(String(24), default="hashing")
    vector: Mapped[bytes] = mapped_column(LargeBinary)


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("profile_id", "slug", name="uq_entity_profile_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(160))
    slug: Mapped[str] = mapped_column(String(160), index=True)
    kind: Mapped[str] = mapped_column(String(24), default="tech")  # company|tech|person|institution
    aliases: Mapped[list] = mapped_column(JSON, default=list)
    is_competitor: Mapped[bool] = mapped_column(Boolean, default=False)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class FindingEntity(Base):
    __tablename__ = "finding_entities"
    __table_args__ = (
        UniqueConstraint("finding_id", "entity_id", name="uq_finding_entity"),
        Index("ix_fe_entity", "entity_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id", ondelete="CASCADE"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)


class TrendPoint(Base):
    __tablename__ = "trend_points"
    __table_args__ = (
        UniqueConstraint("profile_id", "entity_slug", "bucket_date", name="uq_trend_bucket"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    entity_slug: Mapped[str] = mapped_column(String(160), index=True)
    bucket_date: Mapped[date] = mapped_column(Date, index=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


# ─────────────────────────────────────────────────────────────
# Action / delivery
# ─────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    insight_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    title: Mapped[str] = mapped_column(Text, default="")
    body: Mapped[str] = mapped_column(Text, default="")
    channel: Mapped[str] = mapped_column(String(24), default="in_app")
    status: Mapped[str] = mapped_column(String(16), default="sent")  # sent|failed|skipped
    error: Mapped[str] = mapped_column(Text, default="")
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Digest(Base):
    __tablename__ = "digests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    period: Mapped[str] = mapped_column(String(16), default="daily")
    title: Mapped[str] = mapped_column(Text, default="")
    content_md: Mapped[str] = mapped_column(Text, default="")
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Battlecard(Base):
    __tablename__ = "battlecards"
    __table_args__ = (UniqueConstraint("profile_id", "competitor", name="uq_battlecard"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    competitor: Mapped[str] = mapped_column(String(160))
    content_md: Mapped[str] = mapped_column(Text, default="")
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    reasoner: Mapped[str] = mapped_column(String(48), default="heuristic")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ─────────────────────────────────────────────────────────────
# Transparency
# ─────────────────────────────────────────────────────────────
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("tracking_profiles.id", ondelete="CASCADE"), index=True
    )
    trigger: Mapped[str] = mapped_column(String(24), default="manual")  # manual|scheduled|seed
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|completed|failed|partial
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    degraded_sources: Mapped[list] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")

    profile: Mapped[TrackingProfile] = relationship(back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunEvent.seq"
    )


class RunEvent(Base):
    """One line of the Agent Activity Log. Powers live narration + Run Replay."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    node: Mapped[str] = mapped_column(String(32), default="agent")
    level: Mapped[str] = mapped_column(String(12), default="info")  # info|warn|error|success
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="events")


class SourceHealth(Base):
    __tablename__ = "source_health"

    source: Mapped[str] = mapped_column(String(48), primary_key=True)
    state: Mapped[str] = mapped_column(String(16), default="closed")  # closed|open|half_open
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_failures: Mapped[int] = mapped_column(Integer, default=0)
    p50_ms: Mapped[int] = mapped_column(Integer, default=0)
    last_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str] = mapped_column(Text, default="")
    opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    forced_failure: Mapped[bool] = mapped_column(Boolean, default=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    insight_id: Mapped[str] = mapped_column(String(32), index=True)
    profile_id: Mapped[str] = mapped_column(String(32), index=True)
    vote: Mapped[int] = mapped_column(Integer, default=0)  # +1 useful, -1 noise
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LlmUsage(Base):
    """Cost governor ledger."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    model: Mapped[str] = mapped_column(String(64), default="")
    purpose: Mapped[str] = mapped_column(String(48), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
