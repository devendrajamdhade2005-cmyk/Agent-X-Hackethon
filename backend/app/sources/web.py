"""Web intelligence connector: Tavily Search API.

Tavily is an LLM-oriented search API: it returns ranked, cleaned page content
rather than a list of blue links, which makes it the right tool for "what is
happening right now" questions that curated feeds and academic indexes miss —
company announcements, press releases, product launches, market moves.

Complements rather than replaces the existing providers:
  * arXiv / OpenAlex  → peer-reviewed and preprint research
  * RSS / Hacker News → a curated, deliberately narrow slice of tech press
  * Tavily            → the open web, ranked for relevance and freshness
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery
from .credibility import classify

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyConnector(SourceConnector):
    name = "tavily"
    source_type = "web"
    label = "Tavily Web Search"
    requires_key = True
    rate_limit_per_min = 30
    timeout_seconds = 25.0
    docs_url = "https://docs.tavily.com/documentation/api-reference/endpoint/search"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        query = self._build_query(q)
        # `news` topic gives recency-ranked results with publication dates, which is
        # what competitor and announcement tracking needs. `general` is better for
        # broad background, so pick from what the planner asked for.
        topic = str(q.extra.get("topic") or "news").lower()
        if topic not in {"news", "general"}:
            topic = "news"

        body: dict[str, object] = {
            "query": query[:390],
            "max_results": min(max(q.limit, 3), 20),
            "search_depth": str(q.extra.get("search_depth") or "basic"),
            "topic": topic,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if topic == "news":
            body["days"] = max(1, min(q.since_days, 365))

        try:
            resp = await client.post(
                TAVILY_ENDPOINT,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise SourceError(f"timeout after {self.timeout_seconds}s", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise SourceError(f"transport error: {exc}", retryable=True) from exc

        if resp.status_code in (401, 403):
            raise SourceError(
                f"Tavily rejected the API key ({resp.status_code})", retryable=False,
                status=resp.status_code,
            )
        if resp.status_code == 429:
            raise SourceError("Tavily rate limit / quota reached (429)", retryable=True, status=429)
        if resp.status_code == 432:
            # Tavily uses 432 for plan/credit exhaustion.
            raise SourceError("Tavily credits exhausted", retryable=False, status=432)
        if resp.status_code >= 500:
            raise SourceError(f"Tavily upstream {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("detail") or resp.text)[:160]
            except ValueError:
                detail = resp.text[:160]
            raise SourceError(f"Tavily error {resp.status_code}: {detail}", retryable=False)

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError("Tavily returned a non-JSON body", retryable=True) from exc

        items: list[RawItem] = []
        for row in payload.get("results") or []:
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if not title or not url:
                continue

            published = _parse_web_date(row.get("published_date"))
            score = row.get("score")
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=url,
                    raw_text=row.get("content") or row.get("raw_content") or title,
                    author=_domain(url),
                    published_at=published,
                    external_id=str(row.get("id") or url),
                    # Domain-based tiering still applies: a Reuters hit outranks a blog.
                    credibility=classify(url, self.source_type),
                    meta={
                        "outlet": _domain(url),
                        "domain": _domain(url),
                        "tavily_score": round(float(score), 4) if _is_num(score) else None,
                        "topic": topic,
                        "search_query": query,
                    },
                )
            )
        return items

    def _build_query(self, q: SourceQuery) -> str:
        """Natural-language query — Tavily ranks better on prose than boolean syntax."""
        explicit = str(q.extra.get("web_query") or "").strip()
        if explicit:
            return explicit

        parts: list[str] = []
        if q.competitors:
            parts.append(" OR ".join(q.competitors[:3]))
        if q.keywords:
            parts.append(" ".join(q.keywords[:2]))
        elif q.query:
            parts.append(q.query)

        base = " ".join(p for p in parts if p).strip() or q.query or "technology news"
        # Steer away from evergreen explainer pages toward datable developments.
        if q.competitors:
            return f"{base} latest announcement news development"
        return f"{base} latest news development"

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


# ─────────────────────────────────────────────────────────────
def _parse_web_date(value: object) -> datetime | None:
    """Tavily returns RFC 1123 dates ("Wed, 19 Aug 2026 11:19:12 GMT")."""
    if not value:
        return None
    text = str(value).strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt is None:
        return None
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _is_num(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True
