"""Patent tool — IP filings and assignee tracking.

The agent should NOT call this on every run. A patent sweep is expensive and only
pays off when there is a reason: the goal mentions IP, or an earlier observation
hinted at a filing, or a competitor is active enough that its IP posture matters.
The decision engine enforces that; this module just does the search well.
"""

from __future__ import annotations

from .base import Tool, ToolContext, ToolInput, ToolResult


class PatentTool(Tool):
    name = "patent_search"
    display_name = "Patent Search"
    source_label = "patent"
    provider_names = ("patentsview", "serpapi")

    description = (
        "Searches patent filings and grants (USPTO PatentsView, Google Patents). "
        "Returns title, assignee, filing/publication date and abstract, and flags when "
        "a tracked competitor is the assignee."
    )
    when_to_use = (
        "Use when the goal explicitly involves patents or IP, when an earlier "
        "observation referenced a patent or filing, or when you need to check whether "
        "a competitor's announced capability is protected. Skip it if the goal is "
        "purely about news sentiment or market chatter."
    )

    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        items = await self._collect(self.provider_names, tool_input, ctx, result)

        tracked = [c.lower() for c in tool_input.competitors]
        seen: set[str] = set()
        unique = []
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)

            assignee = str(item.meta.get("assignee") or item.author or "")
            matched = _match_assignee(assignee, tool_input.competitors)
            if matched:
                item.competitor = matched
                item.signals = ["competitor-assignee"]
            unique.append(item)

        # A tracked competitor as assignee is the highest-value patent signal, so
        # surface those first regardless of date.
        unique.sort(
            key=lambda i: (bool(i.competitor), i.published_date or ""), reverse=True
        )
        result.items = unique[: tool_input.limit]

        if result.items:
            matches = sorted({i.competitor for i in result.items if i.competitor})
            result.note = result.note or (
                f"{len(result.items)} filings; tracked assignee match: {', '.join(matches)}"
                if matches
                else f"{len(result.items)} filings, no tracked competitor as assignee"
            )
        del tracked


def _match_assignee(assignee: str, competitors: list[str]) -> str:
    a = (assignee or "").lower()
    if not a:
        return ""
    for competitor in competitors:
        c = competitor.strip().lower()
        if not c:
            continue
        if c in a or a in c:
            return competitor
        head = c.split()[0]
        if len(head) > 3 and head in a:
            return competitor
    return ""
