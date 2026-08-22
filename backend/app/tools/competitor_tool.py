"""Competitor tool — company-scoped intelligence.

Deliberately separate from the news tool. A competitor sweep is a different
question ("what is *this company* doing?") and needs a different provider mix:
news for announcements, repos for shipped engineering, forums for practitioner
reaction. Results are attributed to a named competitor and verified to actually
mention that company, so the agent can reason about per-competitor coverage.
"""

from __future__ import annotations

from dataclasses import replace

from .base import FindingRecord, Tool, ToolContext, ToolInput, ToolResult
from .signals import detect_signals


class CompetitorTool(Tool):
    name = "competitor_search"
    display_name = "Competitor Intelligence"
    source_label = "competitor"
    provider_names = (
        "rss", "hackernews", "newsapi", "newsdata", "gnews", "github", "reddit",
    )

    description = (
        "Searches for activity by a specific named company across news, open-source "
        "repositories and practitioner forums. Returns company-attributed items with "
        "announcement signals (launches, funding, partnerships, acquisitions)."
    )
    when_to_use = (
        "Use only when the goal names competitors to track, or when another tool "
        "surfaced a company that clearly matters to the goal. Call it per company. "
        "Do not use it for generic topic news."
    )

    input_schema = {
        **Tool.input_schema,
        "competitors": "string[] — REQUIRED. The company names to investigate.",
    }

    async def _execute(
        self, tool_input: ToolInput, ctx: ToolContext, result: ToolResult
    ) -> None:
        if not tool_input.competitors:
            result.ok = False
            result.error = "competitor_search requires at least one competitor name"
            return

        # One scoped sweep per company so results stay attributable.
        for company in tool_input.competitors[:3]:
            scoped = replace(
                tool_input,
                query=f"{company} {' '.join(tool_input.keywords[:2])}".strip(),
                competitors=[company],
                limit=max(4, tool_input.limit // max(1, len(tool_input.competitors[:3]))),
            )

            news_items = await self._collect(
                ("rss", "hackernews", "newsapi", "newsdata", "gnews"),
                scoped,
                ctx,
                result,
                competitor=company,
            )
            repo_items = await self._collect(
                ("github",),
                scoped,
                ctx,
                result,
                competitor=company,
                extra={"org": company},
            )
            social_items = await self._collect(
                ("reddit",), scoped, ctx, result, competitor=company
            )

            for item in news_items + repo_items + social_items:
                if not _mentions(item, company):
                    continue
                item.signals = _announcement_signals(item)
                result.items.append(item)

        # Dedupe across companies, keeping whichever copy carries more signal.
        best: dict[str, FindingRecord] = {}
        for item in result.items:
            existing = best.get(item.id)
            if existing is None or len(item.signals) > len(existing.signals):
                best[item.id] = item

        # Round-robin by company before truncating. Taking the first N outright
        # would spend the whole budget on the first company in the list and leave
        # the others with zero coverage.
        result.items = _balance_by_company(
            list(best.values()), max(tool_input.limit, 6), tool_input.competitors[:3]
        )

        if result.items:
            companies = sorted({i.competitor for i in result.items if i.competitor})
            result.note = result.note or f"activity found for: {', '.join(companies)}"
        elif not result.error:
            result.note = (
                f"no recent activity found for {', '.join(tool_input.competitors[:3])}"
            )


def _balance_by_company(
    items: list[FindingRecord], limit: int, companies: list[str]
) -> list[FindingRecord]:
    """Interleave results so every requested company gets represented."""
    if not companies:
        return items[:limit]

    buckets: dict[str, list[FindingRecord]] = {c: [] for c in companies}
    spare: list[FindingRecord] = []
    for item in items:
        if item.competitor in buckets:
            buckets[item.competitor].append(item)
        else:
            spare.append(item)

    for bucket in buckets.values():
        bucket.sort(key=lambda i: (len(i.signals), i.published_date or ""), reverse=True)

    out: list[FindingRecord] = []
    index = 0
    while len(out) < limit and any(len(b) > index for b in buckets.values()):
        for company in companies:
            bucket = buckets[company]
            if len(bucket) > index and len(out) < limit:
                out.append(bucket[index])
        index += 1

    for item in spare:
        if len(out) >= limit:
            break
        out.append(item)
    return out


def _mentions(item, company: str) -> bool:
    """Guard against generic results leaking into a company-scoped sweep."""
    company = company.strip().lower()
    if not company:
        return False
    haystack = " ".join(
        [
            item.title or "",
            item.raw_text or "",
            item.author or "",
            str(item.meta.get("owner") or ""),
            str(item.meta.get("company") or ""),
        ]
    ).lower()
    if company in haystack:
        return True
    # "Samsung SDI" should still match a headline that only says "Samsung".
    head = company.split()[0]
    return len(head) > 3 and head in haystack


def _announcement_signals(item: FindingRecord) -> list[str]:
    signals = detect_signals(item.title, item.raw_text)
    if item.provider == "github":
        # A public release is the most concrete evidence a company actually shipped.
        signals.append("shipped-code")
    return sorted(set(signals))[:6]
