"""Run the agent from the terminal and watch the loop live.

Examples:
    python -m scripts.run_agent
    python -m scripts.run_agent --goal "Monitor patents related to Generative AI"
    python -m scripts.run_agent --goal "Track AI Agents" --competitors OpenAI,Anthropic
    python -m scripts.run_agent --sim          # fully offline, no network
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.agent import run_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the InsightPulse agent")
    p.add_argument(
        "--goal",
        default="Track important developments in AI Agents and monitor OpenAI and Anthropic",
    )
    p.add_argument("--keywords", default="", help="comma separated")
    p.add_argument("--competitors", default="", help="comma separated")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--sim", action="store_true", help="force simulation mode (no network)")
    p.add_argument("--json", action="store_true", help="dump the full JSON result")
    return p.parse_args()


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


async def main() -> int:
    args = parse_args()
    print("=" * 78)
    print("InsightPulse — autonomous research & competitor intelligence agent")
    print("=" * 78)

    result = await run_agent(
        args.goal,
        keywords=_split(args.keywords),
        competitors=_split(args.competitors),
        max_iterations=args.max_iterations,
        simulation_mode=True if args.sim else None,
        echo=True,
    )

    print("\n" + "=" * 78)
    print("PRIORITIZED INSIGHTS")
    print("=" * 78)
    print(result.insights_text or "(none)")

    print("\n" + "=" * 78)
    print("EXECUTIVE SUMMARY")
    print("=" * 78)
    print(result.summary or "(none)")

    m = result.metrics
    print("\n" + "-" * 78)
    print(
        f"status={result.status}  iterations={m['iterations']}/{m['max_iterations']}  "
        f"tool_calls={m['tool_calls']}  findings={m['findings_total']} "
        f"(relevant {m['findings_relevant']}, dupes suppressed {m['duplicates_suppressed']})"
    )
    print(
        f"insights={m['insights']} {m['priority_counts']}  errors={m['errors']}  "
        f"reasoner={m['reasoner']}  duration={m['duration_ms']}ms"
    )
    print(f"tools used in order: {' → '.join(m['tools_used']) or 'none'}")

    if args.json:
        import json

        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
