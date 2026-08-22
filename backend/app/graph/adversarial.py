"""Adversarial test mode — deterministic, controlled fault injection.

The adversarial demo must be *repeatable*, so it cannot depend on real external
APIs failing on cue. Instead the graph injects faults at points it controls and
then genuinely recovers through the same routing, retry, fallback, verification and
replanning logic a real failure would exercise. Nothing about the final success is
faked: the recovery path is the production path.

What can be injected (section 27):
  1. tool failure            — a targeted tool's primary attempt fails
  2. tool timeout            — the failure is a timeout, and it fails twice
  3. fallback source success — the fallback attempt (real tool call) succeeds
  4. conflicting evidence    — two contradictory findings enter the evidence set
  5. low-confidence evidence — a weak, single-source item lowers confidence
  6. budget constraint       — a tightened resource budget forces triage
  7. replanning requirement  — the conflict/failure forces a plan revision
  8. deadlock condition       — (optional) repeated no-progress action

The controller is intentionally one-shot per fault: once a tool's injected failure
has been "recovered", later calls to that tool pass, so the fallback genuinely
returns data and the run can complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Primary → fallback provider labels per tool. The fallback is a real second tool
# call; these labels describe which source recovered, matching the project's actual
# provider chains (research: arXiv→OpenAlex→Semantic Scholar; competitive: live web
# →curated news).
FALLBACK_PROVIDERS = {
    "research_search": ("arxiv", "openalex"),
    "patent_search": ("serpapi", "patentsview"),
    "web_search": ("tavily", "news_fallback"),
    "news_search": ("newsapi", "rss"),
    "competitor_search": ("newsapi", "rss"),
}


@dataclass
class ToolFault:
    tool: str
    provider: str = ""
    fallback_provider: str = ""
    timeout: bool = False          # if True, fails twice (retry also fails) before fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "provider": self.provider,
            "fallback_provider": self.fallback_provider,
            "timeout": self.timeout,
        }


@dataclass
class AdversarialConfig:
    """Declarative description of the faults to inject. Serialisable into state."""

    enabled: bool = False
    scenario: str = "custom"
    tool_faults: list[ToolFault] = field(default_factory=list)
    # Which tools a run actually calls depends on the *dynamic* plan, which is
    # LLM-influenced and varies between runs. With faults pinned to specific tool
    # names, a plan that never reaches those tools would silently produce a
    # fault-free run — and the demo has to be repeatable. When `adaptive` is on, an
    # unfired fault is applied to whichever tool the run does call, so the
    # failure → retry → fallback path is always exercised.
    adaptive: bool = True
    inject_conflict: bool = False
    conflict_subject: str = ""
    conflict_claim_a: str = ""
    conflict_claim_b: str = ""
    inject_low_confidence: bool = False
    budget_override: dict[str, Any] = field(default_factory=dict)
    force_replan: bool = False
    inject_deadlock: bool = False

    # ── serialisation ───────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scenario": self.scenario,
            "adaptive": self.adaptive,
            "tool_faults": [f.to_dict() for f in self.tool_faults],
            "inject_conflict": self.inject_conflict,
            "conflict_subject": self.conflict_subject,
            "conflict_claim_a": self.conflict_claim_a,
            "conflict_claim_b": self.conflict_claim_b,
            "inject_low_confidence": self.inject_low_confidence,
            "budget_override": dict(self.budget_override),
            "force_replan": self.force_replan,
            "inject_deadlock": self.inject_deadlock,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdversarialConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            scenario=str(data.get("scenario", "custom")),
            adaptive=bool(data.get("adaptive", True)),
            tool_faults=[
                ToolFault(
                    tool=f.get("tool", ""),
                    provider=f.get("provider", ""),
                    fallback_provider=f.get("fallback_provider", ""),
                    timeout=bool(f.get("timeout", False)),
                )
                for f in data.get("tool_faults", [])
                if f.get("tool")
            ],
            inject_conflict=bool(data.get("inject_conflict", False)),
            conflict_subject=str(data.get("conflict_subject", "")),
            conflict_claim_a=str(data.get("conflict_claim_a", "")),
            conflict_claim_b=str(data.get("conflict_claim_b", "")),
            inject_low_confidence=bool(data.get("inject_low_confidence", False)),
            budget_override=dict(data.get("budget_override", {})),
            force_replan=bool(data.get("force_replan", False)),
            inject_deadlock=bool(data.get("inject_deadlock", False)),
        )

    # ── presets ─────────────────────────────────────────────
    @classmethod
    def full_scenario(cls, competitor: str = "OpenAI") -> "AdversarialConfig":
        """The mandated adversarial scenario (section 28): a research tool fails, a
        competitive tool times out, evidence conflicts, budget is constrained, and
        the agent must replan and still complete."""
        return cls(
            enabled=True,
            scenario="full_adversarial",
            tool_faults=[
                ToolFault(tool="research_search", provider="arxiv",
                          fallback_provider="openalex", timeout=False),
                ToolFault(tool="web_search", provider="tavily",
                          fallback_provider="news_search", timeout=True),
            ],
            inject_conflict=True,
            conflict_subject=f"{competitor} agent capability X",
            conflict_claim_a=f"{competitor} announced capability X is generally available.",
            conflict_claim_b=f"{competitor} has not publicly released capability X.",
            inject_low_confidence=True,
            budget_override={"max_tool_calls": 8, "max_replans": 2, "usd_ceiling": 0.20},
            force_replan=True,
            inject_deadlock=False,
        )

    @classmethod
    def named(cls, scenario: str, competitor: str = "OpenAI") -> "AdversarialConfig":
        if scenario in ("full", "full_adversarial", "default"):
            return cls.full_scenario(competitor)
        if scenario == "tool_failure":
            return cls(
                enabled=True, scenario="tool_failure",
                tool_faults=[ToolFault("research_search", "arxiv", "openalex")],
            )
        if scenario == "conflict":
            return cls(
                enabled=True, scenario="conflict", inject_conflict=True,
                conflict_subject=f"{competitor} capability X",
                conflict_claim_a=f"{competitor} shipped capability X.",
                conflict_claim_b=f"{competitor} has not shipped capability X.",
                inject_low_confidence=True,
            )
        if scenario == "budget":
            return cls(
                enabled=True, scenario="budget",
                budget_override={"max_tool_calls": 4, "usd_ceiling": 0.10},
            )
        return cls.full_scenario(competitor)


class AdversarialController:
    """Runtime behaviour for a config. Lives on the engine (not checkpointed).

    One-shot faults are tracked here so a fault fires once and the recovery path is
    real: the second (fallback) attempt is an ordinary, successful tool call.
    """

    def __init__(self, config: AdversarialConfig) -> None:
        self.config = config
        self._recovered: set[str] = set()
        self._conflict_emitted = False
        self._low_conf_emitted = False
        # Faults already spent, so the total number injected never exceeds what was
        # configured no matter which tools the dynamic plan reaches for.
        self._fired = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def budget(self) -> int:
        return len(self.config.tool_faults)

    def fault_for(self, tool_name: str) -> ToolFault | None:
        """Return a fault to inject on this tool call, or None.

        Exact tool matches win. Otherwise, when `adaptive` is set and fault budget
        remains, an unfired configured fault is retargeted onto this tool — that is
        what makes the demo repeatable across different dynamic plans.
        """
        if not self.config.enabled or tool_name in self._recovered:
            return None
        # One cap for both paths, so the number of injected faults is exactly what
        # was configured however the plan turns out.
        if self._fired >= self.budget:
            return None
        exact = next((f for f in self.config.tool_faults if f.tool == tool_name), None)
        if exact is not None:
            return exact
        if not self.config.adaptive:
            return None
        # Retarget the next unfired fault, keeping its timeout characteristic and
        # using this tool's real provider chain for the failure/fallback labels.
        template = self.config.tool_faults[self._fired]
        primary, fallback = FALLBACK_PROVIDERS.get(tool_name, ("primary source", "fallback source"))
        return ToolFault(
            tool=tool_name,
            provider=primary,
            fallback_provider=fallback,
            timeout=template.timeout,
        )

    def mark_recovered(self, tool_name: str) -> None:
        self._recovered.add(tool_name)
        self._fired += 1

    def take_conflict(self) -> bool:
        if self.config.enabled and self.config.inject_conflict and not self._conflict_emitted:
            self._conflict_emitted = True
            return True
        return False

    def take_low_confidence(self) -> bool:
        if (
            self.config.enabled
            and self.config.inject_low_confidence
            and not self._low_conf_emitted
        ):
            self._low_conf_emitted = True
            return True
        return False
