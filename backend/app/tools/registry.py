"""Tool registry — the agent's action space.

The decision engine reads `catalog()` to decide what it *could* do next. Adding a
tool means adding one line here; no agent code changes.
"""

from __future__ import annotations

from typing import Any

from ..sources.registry import SourceRegistry, registry as source_registry
from .base import Tool, ToolAvailability
from .competitor_tool import CompetitorTool
from .news_tool import NewsTool
from .patent_tool import PatentTool
from .research_tool import ResearchTool
from .web_tool import WebIntelligenceTool

TOOL_CLASSES: list[type[Tool]] = [
    ResearchTool,
    NewsTool,
    WebIntelligenceTool,
    CompetitorTool,
    PatentTool,
]


class ToolRegistry:
    def __init__(self, sources: SourceRegistry | None = None) -> None:
        self.sources = sources or source_registry
        self._tools: dict[str, Tool] = {
            cls.name: cls(self.sources) for cls in TOOL_CLASSES
        }

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def catalog(self) -> list[dict[str, Any]]:
        return [tool.catalog_entry() for tool in self._tools.values()]

    def availability(self) -> dict[str, ToolAvailability]:
        return {name: tool.availability() for name, tool in self._tools.items()}

    def usable_names(self) -> list[str]:
        return [name for name, av in self.availability().items() if av.available]


tool_registry = ToolRegistry()
