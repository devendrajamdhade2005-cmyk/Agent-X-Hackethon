"""Patent connectors.

Two paths, tried in order of what the deployment actually has:
  1. SerpAPI Google Patents  — richest, needs SERPAPI_KEY
  2. PatentsView Search API  — free, US grants, needs a free PATENTSVIEW_API_KEY
Neither configured → simulated, clearly labelled.

Assignee matching against tracked competitors happens here in `meta` and is
turned into an explicit score modifier by the reasoning node, because a
competitor showing up as an assignee is the single highest-value patent signal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from html import unescape
from typing import Any

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery


class GooglePatentsConnector(SourceConnector):
    """Google Patents via SerpAPI."""

    name = "serpapi"
    source_type = "patent"
    label = "Google Patents (SerpAPI)"
    requires_key = True
    rate_limit_per_min = 20
    timeout_seconds = 18.0
    docs_url = "https://serpapi.com/google-patents-api"

    # Patents publish roughly 18 months after filing, so the tool's default
    # 45-day news-style window returns almost nothing. The patent horizon is
    # therefore floored at two years regardless of the requested recency —
    # a two-year-old filing is still current intelligence in IP terms.
    MIN_HISTORY_DAYS = 730

    # SerpAPI reports "no results" as an `error` string rather than an empty
    # list. Raising on it would look like a provider outage and push the tool
    # into simulated data, so this one phrase is treated as a legitimate zero.
    _EMPTY_MARKER = "hasn't returned any results"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        query = q.query or " ".join(q.keywords[:3])
        if q.competitors:
            query = f"{query} ({' OR '.join(q.competitors[:3])})"
        floor = datetime.now(UTC) - timedelta(days=self.MIN_HISTORY_DAYS)
        since = q.since
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        # Relevance ordering, bounded by a publication-date floor.
        #
        # `sort=new` looks right for an intelligence feed but discards Google
        # Patents' relevance ranking entirely: measured against the live API,
        # "generative AI model compression" returned "Laundry machine and laundry
        # machine operating method" as its top hit. Default (relevance) ordering
        # plus `after=publication:` keeps the results on-topic *and* recent.
        resp = await self._get(
            client,
            "https://serpapi.com/search",
            params={
                "engine": "google_patents",
                "q": query,
                # google_patents rejects `num` outside 10..100 with HTTP 400.
                # The agent often asks for fewer than 10, so the floor matters;
                # the caller trims the surplus.
                "num": max(10, min(q.limit, 100)),
                "after": f"publication:{min(since, floor).strftime('%Y%m%d')}",
                "api_key": self.api_key,
            },
            # SerpAPI serves "no results" as HTTP 400 with an explanatory body,
            # so the status guard has to stand aside for us to read it.
            tolerate_status=(400,),
        )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceError(
                f"unparseable response ({resp.status_code})", retryable=False
            ) from exc
        if payload.get("error"):
            detail = str(payload["error"])
            if self._EMPTY_MARKER in detail.lower():
                return []
            raise SourceError(detail, retryable=False)

        items: list[RawItem] = []
        for row in payload.get("organic_results", []) or []:
            # SerpAPI passes Google's HTML through, so "AT&T" arrives as
            # "At&amp;T" and would be rendered literally in the PDF report.
            title = unescape(row.get("title") or "").strip()
            if not title:
                continue
            assignee = unescape(row.get("assignee") or "")
            pub_num = row.get("publication_number") or row.get("patent_id") or ""
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=row.get("patent_link")
                    or (f"https://patents.google.com/patent/{pub_num}/en" if pub_num else ""),
                    raw_text=unescape(row.get("snippet") or row.get("abstract") or ""),
                    author=assignee,
                    published_at=self._parse_date(
                        row.get("publication_date") or row.get("grant_date")
                    ),
                    external_id=str(pub_num),
                    credibility="high",
                    meta={
                        "assignee": assignee,
                        "patent_number": pub_num,
                        "filing_date": row.get("filing_date", ""),
                        "inventors": unescape(row.get("inventor", "") or ""),
                        "competitor_match": _match_competitor(assignee, q.competitors),
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class PatentsViewConnector(SourceConnector):
    """USPTO PatentsView Search API (free key from patentsview.org)."""

    name = "patentsview"
    source_type = "patent"
    label = "PatentsView (USPTO)"
    requires_key = True
    rate_limit_per_min = 40
    timeout_seconds = 18.0
    docs_url = "https://search.patentsview.org/docs/"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:3] or [q.query]
        text_clauses: list[dict[str, Any]] = [
            {"_text_any": {"patent_title": t}} for t in terms if t
        ]
        criteria: list[dict[str, Any]] = [
            {"_gte": {"patent_date": q.since.date().isoformat()}}
        ]
        if text_clauses:
            criteria.append({"_or": text_clauses})
        if q.competitors:
            criteria.append(
                {
                    "_or": [
                        {"_text_any": {"assignees.assignee_organization": c}}
                        for c in q.competitors[:4]
                    ]
                }
            )

        resp = await self._get(
            client,
            "https://search.patentsview.org/api/v1/patent/",
            params={
                "q": _json(criteria and {"_and": criteria}),
                "f": _json(
                    [
                        "patent_id",
                        "patent_title",
                        "patent_abstract",
                        "patent_date",
                        "assignees.assignee_organization",
                        "inventors.inventor_name_last",
                    ]
                ),
                "o": _json({"size": min(q.limit, 25)}),
                "s": _json([{"patent_date": "desc"}]),
            },
            headers={"X-Api-Key": self.api_key},
        )
        payload = resp.json()
        if payload.get("error"):
            raise SourceError(str(payload.get("error")), retryable=False)

        items: list[RawItem] = []
        for row in payload.get("patents", []) or []:
            title = (row.get("patent_title") or "").strip()
            if not title:
                continue
            assignees = [
                (a or {}).get("assignee_organization") or "" for a in row.get("assignees") or []
            ]
            assignee = next((a for a in assignees if a), "")
            pid = str(row.get("patent_id") or "")
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=f"https://patents.google.com/patent/US{pid}/en" if pid else "",
                    raw_text=row.get("patent_abstract") or "",
                    author=assignee,
                    published_at=self._parse_date(row.get("patent_date")),
                    external_id=pid,
                    credibility="high",
                    meta={
                        "assignee": assignee,
                        "all_assignees": [a for a in assignees if a],
                        "patent_number": f"US{pid}",
                        "competitor_match": _match_competitor(assignee, q.competitors),
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


def _json(value: Any) -> str:
    import json

    return json.dumps(value)


def _match_competitor(assignee: str, competitors: list[str]) -> str:
    """Return the matched competitor name, or empty string."""
    a = (assignee or "").lower()
    if not a:
        return ""
    for c in competitors or []:
        token = c.lower().split()[0] if c else ""
        if token and (token in a or a in c.lower()):
            return c
    return ""
