"""Repo connector: GitHub.

A v2 addition. A competitor shipping an open-source release is frequently the
earliest *public* evidence of a strategy shift — it lands weeks before the press
release and months before the patent publishes. It is also free to monitor.
"""

from __future__ import annotations

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery


class GitHubConnector(SourceConnector):
    name = "github"
    source_type = "repo"
    label = "GitHub"
    requires_key = False  # unauthenticated search works at 10 req/min
    rate_limit_per_min = 10
    timeout_seconds = 14.0
    docs_url = "https://docs.github.com/en/rest/search"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:2] or [q.query]
        query = " ".join(t for t in terms if t)
        # Competitor-scoped repo sweeps are issued by the planner as their own
        # query (extra["org"]); combining org: with keywords here returns nothing
        # almost every time, which is how this was originally wrong.
        org = str(q.extra.get("org") or "").strip()
        if org:
            query = f"org:{_org_slug(org)}"

        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = await self._get(
            client,
            "https://api.github.com/search/repositories",
            params={
                "q": f"{query} pushed:>{q.since.date().isoformat()}"[:250],
                "sort": "updated",
                "order": "desc",
                "per_page": min(q.limit, 20),
            },
            headers=headers,
        )
        if resp.status_code == 422:
            raise SourceError("GitHub rejected the search query", retryable=False, status=422)
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("message") and "items" not in payload:
            raise SourceError(str(payload["message"]), retryable="rate limit" in str(payload["message"]).lower())

        items: list[RawItem] = []
        for repo in payload.get("items", []) or []:
            full_name = repo.get("full_name") or ""
            if not full_name:
                continue
            owner = (repo.get("owner") or {}).get("login") or ""
            topics = repo.get("topics") or []
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=f"{full_name} — {repo.get('description') or 'no description'}"[:300],
                    url=repo.get("html_url") or "",
                    raw_text=(
                        f"{repo.get('description') or ''} "
                        f"Primary language {repo.get('language') or 'unknown'}, "
                        f"{repo.get('stargazers_count') or 0} stars, "
                        f"last pushed {repo.get('pushed_at') or 'unknown'}. "
                        f"Topics: {', '.join(topics) or 'none'}."
                    ),
                    author=owner,
                    published_at=self._parse_date(repo.get("pushed_at") or repo.get("updated_at")),
                    external_id=str(repo.get("id") or full_name),
                    credibility="standard",
                    meta={
                        "owner": owner,
                        "stars": repo.get("stargazers_count") or 0,
                        "forks": repo.get("forks_count") or 0,
                        "language": repo.get("language") or "",
                        "topics": topics[:8],
                        "open_issues": repo.get("open_issues_count") or 0,
                        "license": ((repo.get("license") or {}) or {}).get("spdx_id") or "",
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


def _org_slug(name: str) -> str:
    """"Samsung SDI" → "samsungsdi" — GitHub orgs have no spaces."""
    return "".join(c for c in name.lower() if c.isalnum() or c == "-")
