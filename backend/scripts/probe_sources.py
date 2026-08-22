"""Dev utility: probe every connector independently (live + simulated).

Usage:
    python -m scripts.probe_sources            # live where possible
    python -m scripts.probe_sources --sim      # force simulation
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sources.base import SourceQuery  # noqa: E402
from app.sources.registry import build_http_client, registry  # noqa: E402
from app.sources.resilience import collect_from_source  # noqa: E402


async def main() -> int:
    force_sim = "--sim" in sys.argv
    query = SourceQuery(
        source="probe",
        source_type="research",
        query="solid-state battery electrolyte",
        keywords=["solid-state battery", "sulfide electrolyte"],
        competitors=["QuantumScape", "Toyota", "Samsung SDI"],
        limit=5,
        since_days=45,
    )

    print(f"{'source':<18}{'type':<10}{'items':<7}{'ms':<8}{'mode':<12}note")
    print("-" * 88)
    failures = 0
    async with build_http_client() as client:
        for connector in registry.all():
            q = replace(query, source=connector.name, source_type=connector.source_type)
            outcome = await collect_from_source(connector, client, q, simulation_mode=force_sim)
            mode = "simulated" if outcome.simulated else ("skipped" if outcome.skipped else "live")
            note = outcome.error or outcome.note or "ok"
            if not outcome.ok and not outcome.simulated:
                failures += 1
            print(
                f"{connector.name:<18}{connector.source_type:<10}{len(outcome.items):<7}"
                f"{outcome.latency_ms:<8}{mode:<12}{note[:44]}"
            )
            if outcome.items:
                sample = outcome.items[0]
                print(f"{'':<18}└─ {sample.title[:70]}")

    print("-" * 88)
    print(f"connectors with hard failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
