"""Connector registry — the agent's TOOL layer.

The PLAN node reads this to decide what it can call; the COLLECT node fans out
over it. Adding a tenth source means adding one class and one line here.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..config import Settings, settings as default_settings
from .base import SourceConnector
from .news import (
    GNewsConnector,
    HackerNewsConnector,
    NewsApiConnector,
    NewsDataConnector,
    RssConnector,
)
from .patents import GooglePatentsConnector, PatentsViewConnector
from .repos import GitHubConnector
from .research import ArxivConnector, OpenAlexConnector, SemanticScholarConnector
from .social import RedditConnector
from .web import TavilyConnector

CONNECTOR_CLASSES: list[type[SourceConnector]] = [
    ArxivConnector,
    SemanticScholarConnector,
    OpenAlexConnector,
    PatentsViewConnector,
    GooglePatentsConnector,
    NewsApiConnector,
    GNewsConnector,
    NewsDataConnector,
    RssConnector,
    HackerNewsConnector,
    RedditConnector,
    GitHubConnector,
    TavilyConnector,
]

# Order matters: within a source type the agent prefers the earlier entries and
# only reaches for the rest when it needs more coverage.
PREFERENCE: dict[str, list[str]] = {
    "research": ["arxiv", "openalex", "semantic_scholar"],
    "patent": ["patentsview", "serpapi"],
    "news": ["rss", "hackernews", "newsapi", "newsdata", "gnews"],
    "social": ["reddit"],
    "repo": ["github"],
    "web": ["tavily"],
}


@dataclass(frozen=True)
class ConnectorInfo:
    name: str
    label: str
    source_type: str
    requires_key: bool
    available: bool
    rate_limit_per_min: int
    docs_url: str


class SourceRegistry:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.settings = cfg or default_settings
        creds = self.settings.source_credentials()
        creds["reddit_user_agent"] = self.settings.reddit_user_agent
        self._connectors: dict[str, SourceConnector] = {
            cls.name: cls(credentials=creds) for cls in CONNECTOR_CLASSES
        }

    # ── lookup ──────────────────────────────────────────────
    def all(self) -> list[SourceConnector]:
        return list(self._connectors.values())

    def get(self, name: str) -> SourceConnector | None:
        return self._connectors.get(name)

    def for_types(self, source_types: list[str]) -> list[SourceConnector]:
        """Connectors the agent should try for the requested source types."""
        picked: list[SourceConnector] = []
        for st in source_types:
            for name in PREFERENCE.get(st, []):
                c = self._connectors.get(name)
                if c is None:
                    continue
                # Skip key-gated duplicates when a keyless sibling already covers
                # the type — no point burning a quota on a redundant call.
                if c.requires_key and not c.available() and self._has_available_sibling(st):
                    continue
                picked.append(c)
        return picked

    def _has_available_sibling(self, source_type: str) -> bool:
        return any(
            (c := self._connectors.get(n)) is not None and c.available()
            for n in PREFERENCE.get(source_type, [])
        )

    def describe(self) -> list[ConnectorInfo]:
        return [
            ConnectorInfo(
                name=c.name,
                label=c.label,
                source_type=c.source_type,
                requires_key=c.requires_key,
                available=c.available(),
                rate_limit_per_min=c.rate_limit_per_min,
                docs_url=c.docs_url,
            )
            for c in sorted(self._connectors.values(), key=lambda x: (x.source_type, x.name))
        ]


def build_http_client(timeout: float = 20.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "InsightPulse/2.0 (autonomous research intelligence agent)",
            "Accept": "application/json, application/xml, text/xml, */*",
        },
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )


registry = SourceRegistry()
