"""News tool — industry and technology news."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolInput, ToolResult

_CREDIBILITY_RANK = {"high": 3, "standard": 2, "low": 1, "unverified": 0}


class NewsTool(Tool):
    name = "news_search"
    display_name = "Industry News Search"
    source_label = "news"
    provider_names = ("rss", "hackernews", "newsapi", "gnews")

    description = (
        "Searches industry and technology news plus practitioner discussion "
        "(curated RSS from tier-1 outlets, Hacker News, NewsAPI, GNews). Returns "
        "headline, outlet, date and summary."
    )
    when_to_use = (
        "Use for market context, product launches, funding, partnerships and "
        "regulatory moves. Good for confirming whether a technical development has "
        "reached the market. Not the right tool for company-specific tracking — use "
        "competitor_search for that."
    )

    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        items = await self._collect(self.provider_names, tool_input, ctx, result)

        seen: set[str] = set()
        unique = []
        # Rank by outlet credibility first, then recency: a Reuters piece beats a
        # press-release wire covering the same event.
        for item in sorted(
            items,
            key=lambda i: (
                _CREDIBILITY_RANK.get(i.credibility, 1),
                i.published_date or "",
            ),
            reverse=True,
        ):
            if item.id in seen:
                continue
            seen.add(item.id)
            unique.append(item)

        result.items = unique[: tool_input.limit]
        if result.items:
            outlets = {i.author or str(i.meta.get("outlet") or "") for i in result.items}
            result.note = result.note or (
                f"{len(result.items)} stories from {len([o for o in outlets if o]) or 1} outlet(s)"
            )
