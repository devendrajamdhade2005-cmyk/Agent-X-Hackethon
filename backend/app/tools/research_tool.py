"""Research tool — scientific publications and preprints."""

from __future__ import annotations

from .base import Tool, ToolContext, ToolInput, ToolResult


class ResearchTool(Tool):
    name = "research_search"
    display_name = "Research Search"
    source_label = "research"
    provider_names = ("arxiv", "openalex", "semantic_scholar")

    description = (
        "Searches scientific publications and preprints (arXiv, OpenAlex, Semantic "
        "Scholar) for papers matching the topic keywords. Returns titles, abstracts, "
        "authors, venue and citation counts."
    )
    when_to_use = (
        "Use when the goal involves research trends, scientific/technical progress, new "
        "methods or benchmarks. Use first for research-led goals, since papers often "
        "reveal what a company will announce months later."
    )

    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        items = await self._collect(self.provider_names, tool_input, ctx, result)

        # Prefer recent, well-cited work and drop cross-provider duplicates.
        seen: set[str] = set()
        unique = []
        for item in sorted(
            items,
            key=lambda i: (
                int(i.meta.get("citation_count") or 0) > 0,
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
            venues = {str(i.meta.get("venue") or "").strip() for i in result.items}
            result.note = result.note or (
                f"{len(result.items)} papers across {len([v for v in venues if v]) or 1} venue(s)"
            )
