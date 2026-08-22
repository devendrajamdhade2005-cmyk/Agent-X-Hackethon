"""News connectors: NewsAPI, GNews, NewsData.io, curated RSS, Hacker News."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery
from .credibility import classify, is_non_editorial, is_redirect_wrapper

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

    # The free Developer plan only serves roughly one month of history. Asking
    # for more returns HTTP 426 `parameterInvalid` and the whole call is lost —
    # and the default tool window is 45 days, so this clamp is load-bearing
    # rather than defensive. Verified against the live API.
    MAX_HISTORY_DAYS = 29

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:3] or [q.query]
        query = " OR ".join(f'"{t}"' for t in terms if t)
        if q.competitors:
            query = f"({query}) OR ({' OR '.join(chr(34) + c + chr(34) for c in q.competitors[:3])})"
        floor = datetime.now(UTC) - timedelta(days=self.MAX_HISTORY_DAYS)
        since = q.since
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        resp = await self._get(
            client,
            "https://newsapi.org/v2/everything",
            params={
                "q": query[:480],
                "sortBy": "publishedAt",
                "pageSize": min(q.limit, 25),
                "language": "en",
                "from": max(since, floor).date().isoformat(),
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
            and not is_non_editorial(a.get("url") or "")
            and not is_redirect_wrapper(a.get("url") or "")
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
            and not is_non_editorial(a.get("url") or "")
            and not is_redirect_wrapper(a.get("url") or "")
        ]

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class NewsDataConnector(SourceConnector):
    """newsdata.io — broad global aggregator, generous free tier.

    Free-tier constraints verified against the live API, not guessed:
      * ``size`` above 10 returns HTTP 422 — so the page size is clamped.
      * ``from_date`` is rejected on ``/1/news`` (paid-only), so the recency
        window is applied client-side against ``q.since``.
      * ``content`` comes back as the literal string "ONLY AVAILABLE IN PAID
        PLANS", so it is discarded and ``description`` is used as the body.
    """

    name = "newsdata"
    source_type = "news"
    label = "NewsData.io"
    requires_key = True
    rate_limit_per_min = 30
    docs_url = "https://newsdata.io/documentation"

    # Free tier hard-caps the page size; larger values are an HTTP 422.
    MAX_PAGE_SIZE = 10
    _PAID_PLACEHOLDER = "ONLY AVAILABLE IN PAID PLANS"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:2] or [q.query]
        query = " OR ".join(f'"{t}"' for t in terms if t)
        if q.competitors:
            query = " OR ".join(f'"{c}"' for c in q.competitors[:3])
        # newsdata caps the query string; over-long queries return HTTP 422.
        query = query[:100]

        # Headline matching first. Measured against the live API, `qInTitle`
        # returned 345 focused results where a full-text `q` returned 1783 with
        # 8 of 10 flagged as syndicated duplicates. Broaden only if the tighter
        # query finds nothing, which mirrors how the other connectors degrade.
        payload = await self._search(client, q, query, in_title=True)
        items = self._parse(payload, q)
        if not items:
            payload = await self._search(client, q, query, in_title=False)
            items = self._parse(payload, q)
        return items

    async def _search(
        self,
        client: httpx.AsyncClient,
        q: SourceQuery,
        query: str,
        *,
        in_title: bool,
    ) -> dict:
        resp = await self._get(
            client,
            "https://newsdata.io/api/1/news",
            params={
                "apikey": self.api_key,
                ("qInTitle" if in_title else "q"): query,
                "language": "en",
                # Server-side dedup: this tier syndicates the same wire story
                # across dozens of local outlets.
                "removeduplicate": 1,
                "size": min(q.limit, self.MAX_PAGE_SIZE),
            },
        )
        payload = resp.json()
        if payload.get("status") != "success":
            detail = payload.get("results") or payload.get("message") or "newsdata error"
            # 429 = free-tier throttle: worth retrying, unlike a rejected key.
            raise SourceError(str(detail)[:200], retryable=resp.status_code == 429)
        return payload

    def _parse(self, payload: dict, q: SourceQuery) -> list[RawItem]:
        cutoff = _naive_utc(getattr(q, "since", None))
        items: list[RawItem] = []
        for a in payload.get("results", []) or []:
            title = (a.get("title") or "").strip()
            url = a.get("link") or ""
            if not title or a.get("duplicate"):
                continue
            if is_non_editorial(url) or is_redirect_wrapper(url):
                continue
            published = self._parse_date(a.get("pubDate"))
            # The API cannot filter by date on this tier, so enforce the window
            # here. `_parse_date` returns naive UTC by convention, so the cutoff
            # is normalised the same way before comparing.
            if published and cutoff and published < cutoff:
                continue
            body = a.get("description") or ""
            content = a.get("content") or ""
            if content and self._PAID_PLACEHOLDER not in content.upper():
                body = f"{body} {content}".strip()
            outlet = a.get("source_name") or a.get("source_id") or ""
            creators = a.get("creator") or []
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=url,
                    raw_text=body or title,
                    author=(creators[0] if isinstance(creators, list) and creators else outlet),
                    published_at=published,
                    external_id=str(a.get("article_id") or url),
                    credibility=classify(url, self.source_type),
                    meta={
                        "outlet": outlet,
                        "domain": url.split("/")[2] if "://" in url else "",
                        "categories": [c for c in (a.get("category") or []) if c][:3],
                    },
                )
            )
        return items

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


def _naive_utc(value: datetime | None) -> datetime | None:
    """Match `SourceConnector._parse_date`, which yields naive-UTC datetimes.

    Mixing the two raises `TypeError: can't compare offset-naive and
    offset-aware datetimes`, so any cutoff is normalised before use.
    """
    if value is None:
        return None
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


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
