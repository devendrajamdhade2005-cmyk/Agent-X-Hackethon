"""Source connector contract.

Every one of the nine data sources implements this same interface, which is what
lets the COLLECT node fan out in parallel and treat a failure in one source as a
local, recoverable event instead of a run-ending exception.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

SOURCE_TYPES = ("research", "patent", "news", "social", "repo")


@dataclass(slots=True)
class SourceQuery:
    """One planned query against one source, produced by the PLAN node."""

    source: str
    source_type: str
    query: str
    keywords: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    limit: int = 12
    since_days: int = 30
    rationale: str = ""
    allow_broaden: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def since(self) -> datetime:
        return datetime.now(UTC) - timedelta(days=self.since_days)

    def broadened(self) -> "SourceQuery":
        """A deliberately looser second attempt.

        A precise query returning nothing is information, not a dead end: the
        agent widens the time window and drops to the single strongest keyword,
        then reports in the Activity Log that it had to broaden.
        """
        head = (self.keywords[:1] or [self.query])[0]
        return SourceQuery(
            source=self.source,
            source_type=self.source_type,
            query=head,
            keywords=[head] if head else [],
            competitors=self.competitors[:1],
            limit=self.limit,
            since_days=min(self.since_days * 3, 180),
            rationale=f"broadened after zero results for '{self.query}'",
            allow_broaden=False,
            extra=dict(self.extra),
        )


@dataclass(slots=True)
class RawItem:
    """Normalized shape every connector must return."""

    source_type: str
    source_name: str
    title: str
    url: str
    raw_text: str = ""
    author: str = ""
    published_at: datetime | None = None
    external_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    credibility: str = "standard"
    is_simulated: bool = False

    def clean(self) -> "RawItem":
        self.title = _squash(self.title)[:400] or "(untitled)"
        self.raw_text = _squash(self.raw_text)[:6000]
        self.author = _squash(self.author)[:300]
        self.url = (self.url or "").strip()[:1000]
        if self.published_at is not None and self.published_at.tzinfo is not None:
            self.published_at = self.published_at.astimezone(UTC).replace(tzinfo=None)
        return self


_WS = re.compile(r"\s+")
_TAGS = re.compile(r"<[^>]+>")


def _retry_after(resp: httpx.Response) -> float | None:
    """Respect the server's own backoff instruction when it gives one."""
    raw = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return min(value, 10.0) if value > 0 else None


def _squash(value: Any) -> str:
    if value is None:
        return ""
    text = _TAGS.sub(" ", str(value))
    return _WS.sub(" ", text).strip()


class SourceError(RuntimeError):
    """Raised by a connector when a fetch cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class SourceConnector(ABC):
    """Base class. `fetch` hits the network; `simulate` never does."""

    name: str = "source"
    source_type: str = "news"
    label: str = "Source"
    requires_key: bool = False
    rate_limit_per_min: int = 30
    timeout_seconds: float = 12.0
    docs_url: str = ""

    def __init__(self, credentials: dict[str, str] | None = None) -> None:
        self.credentials = credentials or {}

    # ── capability ──────────────────────────────────────────
    @property
    def api_key(self) -> str:
        return (self.credentials.get(self.name) or "").strip()

    def available(self) -> bool:
        """False → the runner serves simulated items and says so in the log."""
        return not self.requires_key or bool(self.api_key)

    # ── work ────────────────────────────────────────────────
    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        """Live network fetch. Raise SourceError on failure."""

    @abstractmethod
    def simulate(self, q: SourceQuery) -> list[RawItem]:
        """Deterministic offline items so the loop is always demoable."""

    # ── helpers for subclasses ──────────────────────────────
    async def _get(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            resp = await client.get(
                url, params=params, headers=headers, timeout=self.timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise SourceError(f"timeout after {self.timeout_seconds}s", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise SourceError(f"transport error: {exc}", retryable=True) from exc

        if resp.status_code == 429:
            raise SourceError(
                "rate limited (429)",
                retryable=True,
                status=429,
                retry_after=_retry_after(resp),
            )
        if resp.status_code in (401, 403):
            raise SourceError(
                f"auth rejected ({resp.status_code}) — check the API key",
                retryable=False,
                status=resp.status_code,
            )
        if resp.status_code >= 500:
            raise SourceError(f"upstream {resp.status_code}", retryable=True, status=resp.status_code)
        if resp.status_code >= 400:
            raise SourceError(
                f"bad request ({resp.status_code})", retryable=False, status=resp.status_code
            )
        return resp

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y%m%d", "%d %b %Y"):
            try:
                dt = datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
            except (ValueError, TypeError):
                continue
            return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt
        return None
