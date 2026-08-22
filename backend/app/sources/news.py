"""News connectors: NewsAPI, GNews, curated RSS, Hacker News."""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery
from .credibility import classify

_EPOCH = datetime(1970, 1, 1)

# Curated, keyless, high-signal feeds. Kept small on purpose: a wide net is the
# fastest way to make an intel feed feel noisy instead of actionable.
CURATED_FEEDS: list[tuple[str, str]] = [
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("IEEE Spectrum", "https://spectrum.ieee.org/feeds/feed.rss"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Science Daily — Matter & Energy", "https://www.sciencedaily.com/rss/matter_energy.xml"),
    ("Nature News", "https://www.nature.com/nature.rss"),
    ("Phys.org", "https://phys.org/rss-feed/technology-news/"),
]


class NewsApiConnector(SourceConnector):
    name = "newsapi"
    source_type = "news"
    label = "NewsAPI"
    requires_key = True
    rate_limit_per_min = 30
    docs_url = "https://newsapi.org/docs/endpoints/everything"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:3] or [q.query]
        query = " OR ".join(f'"{t}"' for t in terms if t)
        if q.competitors:
            query = f"({query}) OR ({' OR '.join(chr(34) + c + chr(34) for c in q.competitors[:3])})"
        resp = await self._get(
            client,
            "https://newsapi.org/v2/everything",
            params={
                "q": query[:480],
                "sortBy": "publishedAt",
                "pageSize": min(q.limit, 25),
                "language": "en",
                "from": q.since.date().isoformat(),
            },
            headers={"X-Api-Key": self.api_key},
        )
        payload = resp.json()
        if payload.get("status") == "error":
            raise SourceError(payload.get("message", "NewsAPI error"), retryable=False)
        return [
            _news_item(
                self,
                title=a.get("title") or "",
                url=a.get("url") or "",
                text=" ".join(filter(None, [a.get("description"), a.get("content")])),
                outlet=(a.get("source") or {}).get("name") or "",
                published=a.get("publishedAt"),
            )
            for a in payload.get("articles", []) or []
            if a.get("title")
        ]

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class GNewsConnector(SourceConnector):
    name = "gnews"
    source_type = "news"
    label = "GNews"
    requires_key = True
    rate_limit_per_min = 20
    docs_url = "https://gnews.io/docs/v4"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:2] or [q.query]
        resp = await self._get(
            client,
            "https://gnews.io/api/v4/search",
            params={
                "q": " OR ".join(f'"{t}"' for t in terms if t)[:200],
                "max": min(q.limit, 10),
                "lang": "en",
                "from": q.since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "apikey": self.api_key,
            },
        )
        payload = resp.json()
        if payload.get("errors"):
            raise SourceError(str(payload["errors"]), retryable=False)
        return [
            _news_item(
                self,
                title=a.get("title") or "",
                url=a.get("url") or "",
                text=" ".join(filter(None, [a.get("description"), a.get("content")])),
                outlet=(a.get("source") or {}).get("name") or "",
                published=a.get("publishedAt"),
            )
            for a in payload.get("articles", []) or []
            if a.get("title")
        ]

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class RssConnector(SourceConnector):
    """Curated RSS — no key, and the most reliable news path in a demo."""

    name = "rss"
    source_type = "news"
    label = "Curated RSS"
    requires_key = False
    rate_limit_per_min = 60
    timeout_seconds = 10.0

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        import feedparser

        terms = [t.lower() for t in (q.keywords or [q.query]) if t]
        competitors = [c.lower() for c in q.competitors]
        # Token sets let "solid-state battery" match "solid state batteries" in a
        # headline. Requiring *all* significant tokens keeps it from going loose.
        token_sets = [
            {tok for tok in _tokens(t) if len(tok) > 3} for t in terms
        ]
        token_sets = [ts for ts in token_sets if ts]

        async def pull(outlet: str, url: str) -> list[RawItem]:
            try:
                resp = await client.get(url, timeout=self.timeout_seconds)
                if resp.status_code >= 400:
                    return []
                parsed = await asyncio.to_thread(feedparser.parse, resp.content)
            except Exception:  # noqa: BLE001 — one dead feed must not fail the batch
                return []

            out: list[RawItem] = []
            for entry in (parsed.entries or [])[:30]:
                title = getattr(entry, "title", "") or ""
                body = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                haystack = f"{title} {body}".lower()
                # Keyword gate: curated feeds are broad, the profile is not.
                if terms and not _matches(haystack, terms, token_sets, competitors):
                    continue
                out.append(
                    _news_item(
                        self,
                        title=title,
                        url=getattr(entry, "link", "") or "",
                        text=body,
                        outlet=outlet,
                        published=getattr(entry, "published", None)
                        or getattr(entry, "updated", None),
                    )
                )
            return out

        batches = await asyncio.gather(
            *(pull(name, url) for name, url in CURATED_FEEDS), return_exceptions=True
        )
        items: list[RawItem] = []
        for batch in batches:
            if isinstance(batch, list):
                items.extend(batch)
        if not items and all(isinstance(b, Exception) for b in batches):
            raise SourceError("every curated feed failed", retryable=True)
        items.sort(key=lambda i: i.published_at or _EPOCH, reverse=True)
        return items[: q.limit]

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class HackerNewsConnector(SourceConnector):
    """HN via Algolia — free, no key. Practitioner reaction, often the earliest signal."""

    name = "hackernews"
    source_type = "news"
    label = "Hacker News"
    requires_key = False
    rate_limit_per_min = 60
    timeout_seconds = 10.0
    docs_url = "https://hn.algolia.com/api"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        query = q.query or " ".join(q.keywords[:2])
        resp = await self._get(
            client,
            "https://hn.algolia.com/api/v1/search_by_date",
            params={
                "query": query[:200],
                "tags": "story",
                "hitsPerPage": min(q.limit, 20),
                "numericFilters": f"created_at_i>{int(q.since.timestamp())}",
            },
        )
        payload = resp.json()
        items: list[RawItem] = []
        for hit in payload.get("hits", []) or []:
            title = (hit.get("title") or hit.get("story_title") or "").strip()
            if not title:
                continue
            hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=hit.get("url") or hn_url,
                    raw_text=(hit.get("story_text") or hit.get("comment_text") or title),
                    author=hit.get("author") or "",
                    published_at=self._parse_date(hit.get("created_at")),
                    external_id=str(hit.get("objectID") or ""),
                    credibility="standard",
                    meta={
                        "points": hit.get("points") or 0,
                        "num_comments": hit.get("num_comments") or 0,
                        "discussion_url": hn_url,
                        "outlet": "Hacker News",
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


def _tokens(text: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def _matches(
    haystack: str,
    terms: list[str],
    token_sets: list[set[str]],
    competitors: list[str],
) -> bool:
    if any(t in haystack for t in terms):
        return True
    if any(c in haystack for c in competitors):
        return True
    hay_tokens = set(_tokens(haystack))
    return any(ts <= hay_tokens for ts in token_sets)


def _news_item(
    connector: SourceConnector,
    *,
    title: str,
    url: str,
    text: str,
    outlet: str,
    published: object,
) -> RawItem:
    return RawItem(
        source_type=connector.source_type,
        source_name=connector.name,
        title=title,
        url=url,
        raw_text=text or title,
        author=outlet,
        published_at=connector._parse_date(published),
        external_id=url,
        credibility=classify(url, connector.source_type),
        meta={"outlet": outlet, "domain": url.split("/")[2] if "://" in url else ""},
    )
