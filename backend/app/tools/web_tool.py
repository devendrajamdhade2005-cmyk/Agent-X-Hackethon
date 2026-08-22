"""Web intelligence tool — live open-web search via Tavily.

Distinct from `news_search` on purpose. `news_search` reads a deliberately narrow
curated set (tier-1 RSS, Hacker News), which is precise but blind to anything
outside those feeds. This tool searches the open web, so it is the right choice
for company announcements, press releases, funding, launches and "what happened
this week" questions — and the right fallback when the curated feeds come back
empty.

The agent decides when to reach for it; see `DecisionEngine._gate`.
"""

from __future__ import annotations

from .base import Tool, ToolContext, ToolInput, ToolResult

_CREDIBILITY_RANK = {"high": 3, "standard": 2, "low": 1, "unverified": 0}


class WebIntelligenceTool(Tool):
    name = "web_search"
    display_name = "Live Web Intelligence"
    source_label = "web"
    provider_names = ("tavily",)

    description = (
        "Searches the live open web via the Tavily Search API and returns ranked, "
        "cleaned page content with publication dates and relevance scores. Covers "
        "company announcements, press releases, product launches, funding, "
        "partnerships and current industry developments."
    )
    when_to_use = (
        "Use for current, real-world web intelligence: what a company just announced, "
        "recent industry developments, market moves, or anything needing up-to-date "
        "coverage beyond curated feeds. Also the right fallback when research or "
        "curated-news searches returned little. Not the right tool for peer-reviewed "
        "papers (use research_search) or patent filings (use patent_search)."
    )

    input_schema = {
        **Tool.input_schema,
        "extra.topic": "'news' for datable developments (default) or 'general' for background",
        "extra.web_query": "optional natural-language query to use verbatim",
    }

    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        extra = dict(tool_input.extra or {})
        extra.setdefault("topic", "news")

        items = await self._collect(
            self.provider_names, tool_input, ctx, result, extra=extra
        )

        # Tavily already ranks by relevance; keep that order but drop duplicates and
        # break ties toward more credible domains.
        seen: set[str] = set()
        unique = []
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            unique.append(item)

        unique.sort(
            key=lambda i: (
                float(i.meta.get("tavily_score") or 0),
                _CREDIBILITY_RANK.get(i.credibility, 1),
            ),
            reverse=True,
        )
        result.items = unique[: tool_input.limit]

        if result.items:
            domains = {str(i.meta.get("domain") or "") for i in result.items}
            result.note = result.note or (
                f"{len(result.items)} live web result(s) across "
                f"{len([d for d in domains if d]) or 1} domain(s)"
            )
