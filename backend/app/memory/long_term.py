"""Long-term memory — what survives a run and can inform a later one.

Storage note: this project has no database. Runs (`api/agent.py::_RUNS`) and reports
(`reports/store.py`) are module-global `OrderedDict`s that die with the process, and
`config.database_url` is vestigial — nothing reads it. Memory that vanished on
restart would not be long-term in any meaningful sense, so this store keeps the same
module-global + bounded-`OrderedDict` shape as the existing stores and adds a JSON
file under the already-configured `DATA_DIR`. That is a deliberate, minimal first
persistence pattern: no new dependency, no new service, no schema migration.

Two rules the rest of the system relies on:

  * **Selective.** Only items at or above an importance floor are accepted. A
    transient HTTP error, a duplicate search hit or a low-relevance result has no
    future value and is rejected at the door.
  * **Never raises.** Every public method degrades to a safe default and records
    why in `degraded`. A memory subsystem must not be able to fail a completed
    intelligence run.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DATA_DIR

# ── memory categories ───────────────────────────────────────
TRACKED_TOPIC = "TRACKED_TOPIC"
TRACKED_COMPETITOR = "TRACKED_COMPETITOR"
USER_MONITORING_PREFERENCE = "USER_MONITORING_PREFERENCE"
IMPORTANT_FINDING = "IMPORTANT_FINDING"
RUN_SUMMARY = "RUN_SUMMARY"
HISTORICAL_BASELINE = "HISTORICAL_BASELINE"
RESEARCH_CONTEXT = "RESEARCH_CONTEXT"
COMPETITIVE_CONTEXT = "COMPETITIVE_CONTEXT"
UNRESOLVED_QUESTION = "UNRESOLVED_QUESTION"

MEMORY_TYPES = (
    TRACKED_TOPIC, TRACKED_COMPETITOR, USER_MONITORING_PREFERENCE,
    IMPORTANT_FINDING, RUN_SUMMARY, HISTORICAL_BASELINE,
    RESEARCH_CONTEXT, COMPETITIVE_CONTEXT, UNRESOLVED_QUESTION,
)

MEMORY_TYPE_LABELS = {
    TRACKED_TOPIC: "Tracked topic",
    TRACKED_COMPETITOR: "Tracked competitor",
    USER_MONITORING_PREFERENCE: "Monitoring preference",
    IMPORTANT_FINDING: "Important finding",
    RUN_SUMMARY: "Previous run summary",
    HISTORICAL_BASELINE: "Historical baseline",
    RESEARCH_CONTEXT: "Research context",
    COMPETITIVE_CONTEXT: "Competitive context",
    UNRESOLVED_QUESTION: "Unresolved question",
}

IMPORTANCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Nothing below HIGH is persisted. Run-local detail stays run-local.
PERSIST_FLOOR = IMPORTANCE_ORDER["HIGH"]

MAX_ITEMS = 400
STORE_VERSION = 1
_WORD = re.compile(r"[a-z0-9\-+]+")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if len(t) > 2}


def _age_days(iso: str) -> float:
    try:
        then = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return 9_999.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - then).total_seconds() / 86_400.0)


@dataclass
class LongTermMemoryItem:
    memory_id: str
    memory_type: str
    content: str
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    importance: str = "HIGH"
    source_run_id: str = ""
    source_goal: str = ""
    url: str = ""
    created_at: str = field(default_factory=_now)
    last_accessed_at: str = ""
    access_count: int = 0
    # How many separate runs have produced this same memory. A recurring item is
    # more trustworthy than a one-off, and recurrence feeds the relevance score.
    recurrence: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    # Optional stable identity. Some memories restate the same thing in different
    # prose every run — a run summary, a baseline — so keying them on their text
    # would accumulate one near-duplicate per run. Giving them an explicit scope
    # makes a later run refresh the existing memory instead.
    dedup_scope: str = ""

    @property
    def rank(self) -> int:
        return IMPORTANCE_ORDER.get(self.importance, 1)

    def match_terms(self) -> set[str]:
        return _tokens(" ".join([
            self.content, self.summary, *self.topics, *self.entities, *self.competitors,
        ]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "summary": self.summary,
            "topics": self.topics,
            "entities": self.entities,
            "competitors": self.competitors,
            "importance": self.importance,
            "source_run_id": self.source_run_id,
            "source_goal": self.source_goal,
            "url": self.url,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "recurrence": self.recurrence,
            "metadata": self.metadata,
            "active": self.active,
            "dedup_scope": self.dedup_scope,
        }

    def public(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "type_label": MEMORY_TYPE_LABELS.get(self.memory_type, self.memory_type),
            "content": self.content,
            "summary": self.summary,
            "topics": self.topics,
            "competitors": self.competitors,
            "importance": self.importance,
            "source_run_id": self.source_run_id,
            "source_goal": self.source_goal,
            "url": self.url,
            "created_at": self.created_at,
            "recurrence": self.recurrence,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "LongTermMemoryItem | None":
        """Rebuild from persisted JSON, or return None if the record is unusable.

        Persisted data is treated as untrusted: a hand-edited or partially written
        file must be quarantined record-by-record, not crash the load.
        """
        if not isinstance(raw, dict):
            return None
        memory_id = str(raw.get("memory_id") or "").strip()
        memory_type = str(raw.get("memory_type") or "").strip()
        content = str(raw.get("content") or "").strip()
        if not memory_id or memory_type not in MEMORY_TYPES or not content:
            return None

        def _strs(key: str) -> list[str]:
            value = raw.get(key)
            if not isinstance(value, list):
                return []
            return [str(v)[:120] for v in value if str(v).strip()][:12]

        importance = str(raw.get("importance") or "HIGH").upper()
        if importance not in IMPORTANCE_ORDER:
            importance = "HIGH"
        try:
            access_count = max(0, int(raw.get("access_count") or 0))
            recurrence = max(1, int(raw.get("recurrence") or 1))
        except (TypeError, ValueError):
            access_count, recurrence = 0, 1

        return cls(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content[:1000],
            summary=str(raw.get("summary") or "")[:400],
            topics=_strs("topics"),
            entities=_strs("entities"),
            competitors=_strs("competitors"),
            importance=importance,
            source_run_id=str(raw.get("source_run_id") or "")[:40],
            source_goal=str(raw.get("source_goal") or "")[:400],
            url=str(raw.get("url") or "")[:500],
            created_at=str(raw.get("created_at") or _now()),
            last_accessed_at=str(raw.get("last_accessed_at") or ""),
            access_count=access_count,
            recurrence=recurrence,
            metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            active=bool(raw.get("active", True)),
            dedup_scope=str(raw.get("dedup_scope") or "")[:160],
        )


def dedup_key(memory_type: str, content: str, scope: str = "") -> str:
    """Identity of a memory: its type plus a normalised discriminator.

    Re-running the same monitoring goal must refresh the existing memory rather
    than accumulate near-identical copies. `scope` overrides the content when the
    memory's prose legitimately changes every run.
    """
    basis = scope or content or ""
    norm = " ".join(_WORD.findall(basis.lower()))[:160]
    return f"{memory_type}::{norm}"


class LongTermStore:
    """Bounded, relevance-searchable memory persisted as JSON."""

    def __init__(self, path: Path | None = None, *, max_items: int = MAX_ITEMS) -> None:
        self.path = path if path is not None else DATA_DIR / "long_term_memory.json"
        self.max_items = max_items
        self._items: "OrderedDict[str, LongTermMemoryItem]" = OrderedDict()
        self._by_key: dict[str, str] = {}
        self._loaded = False
        # Non-empty when the store could not read or write. Surfaced to the UI as an
        # honest status rather than pretending memory is healthy.
        self.degraded = ""
        self.quarantined = 0

    # ── persistence ─────────────────────────────────────────
    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if not self.path.exists():
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — a corrupt file must not break a run
            self.degraded = f"could not read memory file: {type(exc).__name__}"
            return

        records = raw.get("items") if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            self.degraded = "memory file has an unexpected shape"
            return

        for record in records:
            item = LongTermMemoryItem.from_dict(record)
            if item is None:
                self.quarantined += 1
                continue
            self._items[item.memory_id] = item
            self._by_key[dedup_key(item.memory_type, item.content, item.dedup_scope)] = item.memory_id
        self._evict()

    def flush(self) -> bool:
        """Write atomically. Returns False (never raises) when it cannot."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": STORE_VERSION,
                "updated_at": _now(),
                "items": [i.to_dict() for i in self._items.values()],
            }
            # Temp file + replace so a crash mid-write cannot truncate the store.
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return True
        except Exception as exc:  # noqa: BLE001
            self.degraded = f"could not write memory file: {type(exc).__name__}"
            return False

    # ── writes ──────────────────────────────────────────────
    def save_many(self, items: list[LongTermMemoryItem]) -> dict[str, Any]:
        """Persist items above the importance floor. Returns a safe summary."""
        self.load()
        stored: list[LongTermMemoryItem] = []
        refreshed: list[LongTermMemoryItem] = []
        rejected = 0

        for item in items:
            if item.rank < PERSIST_FLOOR:
                rejected += 1
                continue
            key = dedup_key(item.memory_type, item.content, item.dedup_scope)
            existing_id = self._by_key.get(key)
            if existing_id and existing_id in self._items:
                existing = self._items[existing_id]
                existing.recurrence += 1
                existing.last_accessed_at = _now()
                # Importance can only be upgraded; a later weaker signal must not
                # downgrade something already judged critical.
                if item.rank > existing.rank:
                    existing.importance = item.importance
                for value in item.competitors:
                    if value not in existing.competitors:
                        existing.competitors.append(value)
                existing.metadata.setdefault("also_seen_in", [])
                if isinstance(existing.metadata["also_seen_in"], list):
                    if item.source_run_id and item.source_run_id not in existing.metadata["also_seen_in"]:
                        existing.metadata["also_seen_in"].append(item.source_run_id)
                # A baseline describes "what was known as of the last run", so when a
                # later run re-states it the fingerprints must advance. Without this
                # the baseline froze at its first run and every later comparison was
                # made against stale history. Safe ordering-wise: comparison happens
                # during orchestration, consolidation only at the end of the run.
                incoming_prints = item.metadata.get("fingerprints")
                if isinstance(incoming_prints, list) and incoming_prints:
                    existing.metadata["fingerprints"] = incoming_prints
                    existing.metadata["finding_count"] = item.metadata.get("finding_count")
                    existing.metadata["refreshed_from_run"] = item.source_run_id
                    existing.created_at = item.created_at
                    existing.summary = item.summary or existing.summary
                refreshed.append(existing)
                continue
            self._items[item.memory_id] = item
            self._by_key[key] = item.memory_id
            stored.append(item)

        self._evict()
        persisted = self.flush()
        return {
            "stored": len(stored),
            "refreshed": len(refreshed),
            "rejected": rejected,
            "persisted": persisted,
            "total": len(self._items),
            "types": sorted({i.memory_type for i in [*stored, *refreshed]}),
        }

    def _evict(self) -> None:
        """Drop the least valuable items once over capacity.

        Insertion order alone would discard a critical memory in favour of a recent
        trivial one, so eviction is by (importance, recurrence, recency) ascending.
        """
        while len(self._items) > self.max_items:
            victim_id = min(
                self._items,
                key=lambda mid: (
                    self._items[mid].rank,
                    self._items[mid].recurrence,
                    self._items[mid].created_at,
                ),
            )
            victim = self._items.pop(victim_id)
            key = dedup_key(victim.memory_type, victim.content, victim.dedup_scope)
            if self._by_key.get(key) == victim_id:
                self._by_key.pop(key, None)

    def mark_accessed(self, memory_ids: list[str]) -> None:
        self.load()
        touched = False
        for memory_id in memory_ids:
            item = self._items.get(memory_id)
            if item is None:
                continue
            item.access_count += 1
            item.last_accessed_at = _now()
            touched = True
        if touched:
            self.flush()

    # ── reads ───────────────────────────────────────────────
    def search(
        self,
        *,
        terms: list[str] | None = None,
        topics: list[str] | None = None,
        competitors: list[str] | None = None,
        limit: int = 5,
        exclude_run_id: str = "",
        min_score: float = 0.30,
    ) -> list[tuple[LongTermMemoryItem, float]]:
        """Relevance-ranked lookup.

        `min_score` is the guard against the failure mode the brief calls out: a
        quantum-computing goal must not pull in AI-agent memory just because that is
        all the store holds. No match clears the floor, so nothing is returned.
        """
        self.load()
        query_terms = _tokens(" ".join(terms or []))
        topic_terms = _tokens(" ".join(topics or []))
        company_set = {c.strip().lower() for c in (competitors or []) if c.strip()}
        if not query_terms and not company_set:
            return []

        scored: list[tuple[LongTermMemoryItem, float]] = []
        for item in self._items.values():
            if not item.active or (exclude_run_id and item.source_run_id == exclude_run_id):
                continue
            score = self._score(item, query_terms, topic_terms, company_set)
            if score >= min_score:
                scored.append((item, score))

        scored.sort(key=lambda pair: (pair[1], pair[0].rank), reverse=True)
        return scored[:limit]

    @staticmethod
    def _score(
        item: LongTermMemoryItem,
        query_terms: set[str],
        topic_terms: set[str],
        company_set: set[str],
    ) -> float:
        """Topic + entity + competitor + importance + recency + recurrence.

        Topical overlap dominates. Importance and recency can promote among things
        that already match, but cannot make an unrelated memory relevant.
        """
        item_terms = item.match_terms()
        if not item_terms:
            return 0.0

        overlap = query_terms & item_terms
        topical = len(overlap) / max(1, len(query_terms)) if query_terms else 0.0
        score = 0.45 * topical

        item_topics = _tokens(" ".join(item.topics))
        if topic_terms and (topic_terms & item_topics):
            score += 0.15

        item_companies = {c.strip().lower() for c in item.competitors}
        if company_set and (company_set & item_companies):
            score += 0.20

        score += {0: 0.0, 1: 0.03, 2: 0.08, 3: 0.12}.get(item.rank, 0.0)

        age = _age_days(item.created_at)
        if age <= 7:
            score += 0.08
        elif age <= 30:
            score += 0.05
        elif age <= 90:
            score += 0.02

        if item.recurrence > 1:
            score += min(0.06, 0.02 * (item.recurrence - 1))

        return round(min(1.0, score), 3)

    def baseline_for(
        self, *, terms: list[str], competitors: list[str] | None = None
    ) -> LongTermMemoryItem | None:
        """Most recent relevant baseline usable for historical comparison.

        A baseline is only useful if it carries the fingerprints of what was known
        last time. Ranking purely by recency picked whichever relevant item was
        newest — and a `RUN_SUMMARY` is written on every run and carries no
        fingerprints, so from the third run onward it always won and change
        detection silently reported "no baseline available".
        """
        candidates = [
            item
            for item, _score in self.search(
                terms=terms, competitors=competitors, limit=10, min_score=0.30
            )
            if item.memory_type in {HISTORICAL_BASELINE, RUN_SUMMARY}
            and isinstance(item.metadata.get("fingerprints"), list)
            and item.metadata["fingerprints"]
        ]
        if not candidates:
            return None
        # Prefer a purpose-built baseline over any other carrier, then most recent.
        return max(
            candidates,
            key=lambda i: (i.memory_type == HISTORICAL_BASELINE, i.created_at),
        )

    # Which memory types define the *subject* of monitoring, in priority order.
    # A run summary describes a past run; a tracked topic says what the user cares
    # about. When restoring a subjectless "continue monitoring" goal the latter must
    # win, or accumulating run summaries crowd the subject out entirely.
    CONTINUATION_PRIORITY = {
        TRACKED_TOPIC: 0,
        TRACKED_COMPETITOR: 0,
        USER_MONITORING_PREFERENCE: 1,
        HISTORICAL_BASELINE: 2,
        RUN_SUMMARY: 3,
    }

    def continuation_items(self, *, exclude_run_id: str = "", limit: int = 5
                           ) -> list[LongTermMemoryItem]:
        """Best stored monitoring context when the goal names no subject."""
        self.load()
        items = [
            i for i in self._items.values()
            if i.active
            and i.memory_type in self.CONTINUATION_PRIORITY
            and i.source_run_id != exclude_run_id
        ]
        items.sort(key=lambda i: (
            self.CONTINUATION_PRIORITY[i.memory_type],
            -i.rank,
            _age_days(i.created_at),
        ))
        return items[:limit]

    def all_items(self) -> list[LongTermMemoryItem]:
        self.load()
        return list(self._items.values())

    def stats(self) -> dict[str, Any]:
        self.load()
        by_type: dict[str, int] = {}
        for item in self._items.values():
            by_type[item.memory_type] = by_type.get(item.memory_type, 0) + 1
        return {
            "total": len(self._items),
            "by_type": by_type,
            "path": str(self.path),
            "degraded": self.degraded,
            "quarantined": self.quarantined,
        }

    def reset(self, *, delete_file: bool = False) -> None:
        """Clear in-memory state. Used by tests, which must not share memory."""
        self._items.clear()
        self._by_key.clear()
        self._loaded = False
        self.degraded = ""
        self.quarantined = 0
        if delete_file:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


# Module-global singleton, matching how `_RUNS` and `_REPORTS` are exposed.
long_term_store = LongTermStore()
