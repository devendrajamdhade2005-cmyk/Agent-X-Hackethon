"""Social connector: Reddit.

Everything from here is tagged `unverified` and carries a hard score ceiling.
Forum chatter is a genuine early-warning signal, but it is not evidence, and the
UI has to say so.
"""

from __future__ import annotations

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery


class RedditConnector(SourceConnector):
    name = "reddit"
    source_type = "social"
    label = "Reddit"
    requires_key = False  # public JSON search works; OAuth just raises the rate limit
    rate_limit_per_min = 30
    timeout_seconds = 12.0
    docs_url = "https://www.reddit.com/dev/api/"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:2] or [q.query]
        query = " OR ".join(f'"{t}"' for t in terms if t)
        if q.competitors:
            query = f"({query}) OR ({' OR '.join(chr(34) + c + chr(34) for c in q.competitors[:2])})"

        headers = {
            "User-Agent": self.credentials.get("reddit_user_agent") or "insightpulse/2.0",
            "Accept": "application/json",
        }
        params = {
            "q": query[:400],
            "sort": "new",
            "limit": min(q.limit, 25),
            "t": "month",
            "type": "link",
            "raw_json": 1,
        }
        # api.reddit.com answers a descriptive User-Agent; www.reddit.com serves an
        # HTML block page to most server IPs. Try the working host first, keep the
        # other as a fallback so a change on Reddit's side degrades instead of dying.
        payload: dict | None = None
        last_error: SourceError | None = None
        for host in ("https://api.reddit.com/search", "https://www.reddit.com/search.json"):
            try:
                resp = await self._get(client, host, params=params, headers=headers)
                payload = resp.json()
                break
            except SourceError as exc:
                last_error = exc
            except ValueError as exc:
                last_error = SourceError(
                    "non-JSON response (likely IP-blocked)", retryable=True
                )
                del exc
        if payload is None:
            raise last_error or SourceError("reddit unreachable", retryable=True)

        items: list[RawItem] = []
        for child in (payload.get("data") or {}).get("children", []) or []:
            data = child.get("data") or {}
            title = (data.get("title") or "").strip()
            if not title:
                continue
            score = int(data.get("score") or 0)
            comments = int(data.get("num_comments") or 0)
            permalink = data.get("permalink") or ""
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=f"https://reddit.com{permalink}" if permalink else (data.get("url") or ""),
                    raw_text=(data.get("selftext") or "")[:4000] or title,
                    author=f"u/{data.get('author', 'unknown')}",
                    published_at=self._parse_date(
                        _epoch_to_iso(data.get("created_utc"))
                    ),
                    external_id=str(data.get("id") or ""),
                    credibility="unverified",
                    meta={
                        "subreddit": data.get("subreddit") or "",
                        "score": score,
                        "num_comments": comments,
                        "upvote_ratio": data.get("upvote_ratio"),
                        "signal_strength": _strength(score, comments),
                        "external_link": data.get("url") or "",
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


def _epoch_to_iso(value: object) -> str | None:
    try:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(float(value), UTC).isoformat()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _strength(score: int, comments: int) -> str:
    weighted = score + comments * 2
    if weighted > 400:
        return "high"
    if weighted > 80:
        return "medium"
    return "low"
