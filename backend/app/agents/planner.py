"""Planner: turn a natural-language goal into a structured plan.

The plan is not a script. It declares *information needs* — what the agent must
learn to satisfy the goal — and the decision engine then works out how to satisfy
them, in what order, and whether to add needs it discovers along the way.

Two implementations behind one call:
  * LLM planner (Claude) when a key is configured.
  * Heuristic planner otherwise — reads intent from the goal text.
Both produce the same `Plan`, so the rest of the agent never branches on it.
"""

from __future__ import annotations

import re
from typing import Any

from .llm import LLMClient, clamp_int, coerce_str_list
from .sanitize import sanitize
from .state import InformationNeed, Plan

VALID_NEEDS = ("research", "news", "competitor", "patent")

# ── intent lexicons ─────────────────────────────────────────
_RESEARCH_HINTS = (
    "research", "paper", "papers", "publication", "publications", "scientific",
    "study", "studies", "arxiv", "preprint", "benchmark", "method", "methods",
    "technique", "state of the art", "sota", "academic", "literature", "advance",
    "advances", "development", "developments", "breakthrough",
)
_PATENT_HINTS = (
    "patent", "patents", "ip", "intellectual property", "filing", "filings",
    "uspto", "epo", "assignee", "prior art", "portfolio",
)
_COMPETITOR_HINTS = (
    "competitor", "competitors", "competitive", "rival", "rivals", "monitor",
    "track", "versus", "vs", "market share", "player", "players",
)
_NEWS_HINTS = (
    "news", "industry", "market", "launch", "launches", "announcement",
    "announcements", "funding", "raise", "partnership", "acquisition",
    "regulation", "regulatory", "adoption", "commercial", "product",
)

_STOPWORDS = {
    "a", "an", "and", "any", "are", "around", "as", "at", "about", "be", "been",
    "but", "by", "can", "developments", "do", "for", "from", "had", "has", "have",
    "how", "i", "important", "in", "into", "is", "it", "its", "keep", "latest",
    "me", "monitor", "monitoring", "my", "new", "of", "on", "or", "our", "over",
    "related", "relating", "should", "so", "such", "than", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "track", "tracking",
    "up", "us", "was", "we", "were", "what", "when", "which", "who", "why",
    "will", "with", "within", "would", "eye", "recent", "want", "need", "also",
    "please", "across", "like", "top", "key", "major", "some", "all", "get",
}

# Words that describe *what kind of tracking* is wanted rather than the subject.
# They drive the plan's information needs, so they must not pollute the search
# keywords — "monitor patents related to Generative AI" should search for
# "Generative AI", not for "patents".
_INTENT_WORDS = {
    "research", "researches", "paper", "papers", "publication", "publications",
    "patent", "patents", "ip", "filing", "filings", "uspto", "news", "industry",
    "market", "competitor", "competitors", "competitive", "rival", "rivals",
    "study", "studies", "literature", "preprint", "preprints", "trend", "trends",
    "activity", "activities", "update", "updates", "signal", "signals",
    "announcement", "announcements", "space", "sector", "landscape",
}

# Short tokens that are meaningful subjects despite their length.
_KEEP_SHORT = {"ai", "ml", "llm", "llms", "nlp", "iot", "ev", "evs", "5g", "6g", "ar", "vr", "hpc", "gpu", "tpu"}

# "competitors such as OpenAI, Anthropic and Google"
_COMPETITOR_PHRASE = re.compile(
    r"(?:competitors?|rivals?|companies|players)\s*(?:such as|like|including|:)?\s*"
    r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*(?:\s*,\s*[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)*"
    r"(?:\s+and\s+[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)?)",
)
_PROPER_RUN = re.compile(r"\b([A-Z][a-zA-Z0-9&.\-]{1,}(?:\s+[A-Z][a-zA-Z0-9&.\-]{1,}){0,2})\b")

# Words that look like proper nouns in a sentence but are never companies.
_NOT_COMPANIES = {
    # domain words that get capitalised in goals
    "ai", "llm", "llms", "research", "patent", "patents", "news", "generative",
    "multi", "agent", "agents", "competitor", "competitors", "industry", "market",
    "technology", "tech", "developments", "development", "trends", "papers",
    # imperative verbs that start a goal sentence and look like proper nouns
    "track", "monitor", "keep", "follow", "watch", "find", "show", "tell", "give",
    "compare", "analyse", "analyze", "summarise", "summarize", "report",
    "identify", "investigate", "check", "look", "review", "scan", "alert",
    # function words
    "i", "the", "and", "for", "with", "what", "which", "how", "is", "are", "do",
}


class Planner:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    async def build(
        self,
        goal: str,
        *,
        keywords: list[str] | None = None,
        competitors: list[str] | None = None,
        topics: list[str] | None = None,
    ) -> Plan:
        goal_clean, _ = sanitize(goal, max_chars=1200)
        supplied_keywords = coerce_str_list(keywords, limit=10)
        supplied_competitors = coerce_str_list(competitors, limit=8)
        supplied_topics = coerce_str_list(topics, limit=8)

        if self.llm is not None and self.llm.available:
            plan = await self._llm_plan(
                goal_clean, supplied_keywords, supplied_competitors, supplied_topics
            )
            if plan is not None:
                return plan

        return self._heuristic_plan(
            goal_clean, supplied_keywords, supplied_competitors, supplied_topics
        )

    # ── LLM path ────────────────────────────────────────────
    async def _llm_plan(
        self,
        goal: str,
        keywords: list[str],
        competitors: list[str],
        topics: list[str],
    ) -> Plan | None:
        assert self.llm is not None
        system = (
            "You are the planning module of an autonomous competitive-intelligence "
            "agent. You convert a user's tracking goal into a structured research "
            "plan. You do not perform the research yourself.\n\n"
            "Available information needs (use only these keys):\n"
            "  research  — scientific papers and preprints\n"
            "  news      — industry/technology news and market moves\n"
            "  competitor— activity by specific named companies\n"
            "  patent    — patent filings and IP posture\n\n"
            "Mark a need `required: true` only when the goal genuinely cannot be "
            "satisfied without it. Do not mark `patent` required unless the goal "
            "concerns IP, or protection of a capability. Do not mark `competitor` "
            "required if no company is named.\n"
            "Reply with ONLY a JSON object."
        )
        user = (
            f"USER GOAL: {goal}\n"
            f"KEYWORDS SUPPLIED: {keywords or 'none'}\n"
            f"COMPETITORS SUPPLIED: {competitors or 'none'}\n"
            f"TOPICS SUPPLIED: {topics or 'none'}\n\n"
            "Return JSON with this exact shape:\n"
            "{\n"
            '  "interpretation": "one sentence restating what the user wants tracked",\n'
            '  "topics": ["..."],\n'
            '  "keywords": ["3-6 concrete search phrases"],\n'
            '  "competitors": ["company names, [] if none"],\n'
            '  "needs": [{"key":"research","reason":"why this is needed",'
            '"required":true,"min_items":2}],\n'
            '  "opening_move": "which need to satisfy first and why",\n'
            '  "success_criteria": "how you will know the goal is satisfied"\n'
            "}"
        )
        data = await self.llm.complete_json(
            purpose="plan", system=system, user=user, max_tokens=1200
        )
        if not data:
            return None

        needs = self._coerce_needs(data.get("needs"))
        if not needs:
            return None

        plan_competitors = coerce_str_list(data.get("competitors"), limit=8) or competitors
        plan = Plan(
            objective=goal,
            interpretation=str(data.get("interpretation") or "")[:400],
            needs=needs,
            opening_move=str(data.get("opening_move") or "")[:300],
            success_criteria=str(data.get("success_criteria") or "")[:300],
            author=self.llm.reasoner_name,
        )
        # Guardrail: a competitor need makes no sense with no companies to check.
        if not plan_competitors:
            for need in plan.needs:
                if need.key == "competitor":
                    need.required = False
                    need.reason = "no companies named — skipped unless one surfaces"
        plan.revisions.append("initial plan")
        self._llm_extras = {
            "topics": coerce_str_list(data.get("topics"), limit=8) or topics,
            "keywords": coerce_str_list(data.get("keywords"), limit=8) or keywords,
            "competitors": plan_competitors,
        }
        return plan

    def _coerce_needs(self, raw: Any) -> list[InformationNeed]:
        needs: list[InformationNeed] = []
        seen: set[str] = set()
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip().lower()
            if key not in VALID_NEEDS or key in seen:
                continue
            seen.add(key)
            needs.append(
                InformationNeed(
                    key=key,
                    reason=str(item.get("reason") or "")[:240],
                    required=bool(item.get("required", True)),
                    min_items=clamp_int(item.get("min_items"), 1, 8, 2),
                )
            )
        return needs

    # ── heuristic path ──────────────────────────────────────
    def _heuristic_plan(
        self,
        goal: str,
        keywords: list[str],
        competitors: list[str],
        topics: list[str],
    ) -> Plan:
        text = goal.lower()
        competitors = competitors or extract_competitors(goal)
        keywords = keywords or extract_keywords(goal)
        topics = topics or keywords[:3]

        wants_research = _hit(text, _RESEARCH_HINTS)
        wants_patent = _hit(text, _PATENT_HINTS)
        wants_news = _hit(text, _NEWS_HINTS)
        wants_competitor = bool(competitors) or _hit(text, _COMPETITOR_HINTS)

        # A goal with no explicit lean still needs a starting point: research plus
        # news is the widest-coverage, lowest-assumption opening.
        if not any((wants_research, wants_patent, wants_news, wants_competitor)):
            wants_research = wants_news = True

        needs: list[InformationNeed] = []
        if wants_research:
            needs.append(
                InformationNeed(
                    key="research",
                    reason="the goal refers to research, technical progress or new methods",
                    required=True,
                    min_items=2,
                )
            )
        if wants_news:
            needs.append(
                InformationNeed(
                    key="news",
                    reason="market and industry context is needed to judge impact",
                    required=True,
                    min_items=2,
                )
            )
        if wants_competitor:
            needs.append(
                InformationNeed(
                    key="competitor",
                    reason=(
                        f"the goal names companies to track: {', '.join(competitors)}"
                        if competitors
                        else "the goal asks for competitive monitoring"
                    ),
                    required=bool(competitors),
                    min_items=2 if competitors else 1,
                )
            )
        needs.append(
            InformationNeed(
                key="patent",
                reason=(
                    "the goal explicitly concerns patents or IP"
                    if wants_patent
                    else "held back — only worth a call if a filing or IP signal appears"
                ),
                required=wants_patent,
                min_items=1,
            )
        )
        if not needs:
            needs.append(
                InformationNeed(key="research", reason="default starting point", required=True)
            )

        first = next((n.key for n in needs if n.required), needs[0].key)
        opening = {
            "research": "start with research — papers lead announcements by months",
            "news": "start with industry news to establish current market state",
            "competitor": "start with the named competitors, since they define the goal",
            "patent": "start with patents, since the goal is explicitly about IP",
        }[first]

        required = [n.key for n in needs if n.required]
        plan = Plan(
            objective=goal,
            interpretation=_interpretation(goal, keywords, competitors),
            needs=needs,
            opening_move=opening,
            success_criteria=(
                f"at least {len(required)} information need(s) satisfied "
                f"({', '.join(required)}) with enough relevant items to prioritize"
            ),
            author="heuristic-planner",
        )
        plan.revisions.append("initial plan")
        self._llm_extras = {
            "topics": topics,
            "keywords": keywords,
            "competitors": competitors,
        }
        return plan

    # ── what the agent should load into state after planning ─
    def derived(self) -> dict[str, list[str]]:
        return getattr(self, "_llm_extras", {"topics": [], "keywords": [], "competitors": []})


# ─────────────────────────────────────────────────────────────
# extraction helpers
# ─────────────────────────────────────────────────────────────
def _hit(text: str, hints: tuple[str, ...]) -> bool:
    return any(h in text for h in hints)


def _interpretation(goal: str, keywords: list[str], competitors: list[str]) -> str:
    parts = []
    if keywords:
        parts.append(f"topics: {', '.join(keywords[:4])}")
    if competitors:
        parts.append(f"companies: {', '.join(competitors[:4])}")
    tail = "; ".join(parts) if parts else "no explicit topics detected"
    return f"Track and prioritize developments for — {tail}."


def extract_competitors(goal: str) -> list[str]:
    """Pull company names out of a free-text goal."""
    found: list[str] = []

    match = _COMPETITOR_PHRASE.search(goal)
    if match:
        for chunk in re.split(r",|\band\b", match.group(1)):
            name = chunk.strip(" .,:;")
            if _looks_like_company(name):
                found.append(name)

    # Only guess at proper nouns when the goal actually asks for company tracking.
    # Otherwise "Keep an eye on multi-agent frameworks" invents a competitor
    # called "Keep", and the agent wastes a tool call chasing it.
    if not found and _hit(goal.lower(), _COMPETITOR_HINTS):
        for candidate in _PROPER_RUN.findall(goal):
            name = candidate.strip(" .,:;")
            if _looks_like_company(name) and name not in found:
                found.append(name)

    out: list[str] = []
    seen: set[str] = set()
    for name in found:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out[:6]


def _looks_like_company(name: str) -> bool:
    if not name or len(name) < 3:
        return False
    if name.lower() in _NOT_COMPANIES:
        return False
    if not name[0].isupper():
        return False
    words = name.split()
    return len(words) <= 3 and all(w.lower() not in _NOT_COMPANIES for w in words)


def extract_keywords(goal: str, limit: int = 6) -> list[str]:
    """Derive search phrases from a goal sentence."""
    competitors = {c.lower() for c in extract_competitors(goal)}
    cleaned = re.sub(r"[^\w\s\-]", " ", goal)
    words = [w for w in cleaned.split() if w]

    phrases: list[str] = []
    buffer: list[str] = []
    for word in words:
        low = word.lower()
        too_short = len(low) < 3 and low not in _KEEP_SHORT
        if low in _STOPWORDS or low in _INTENT_WORDS or low in competitors or too_short:
            if buffer:
                phrases.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(word)
        if len(buffer) == 3:
            phrases.append(" ".join(buffer))
            buffer = []
    if buffer:
        phrases.append(" ".join(buffer))

    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        p = phrase.strip()
        if len(p) < 3:
            continue
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
    return out[:limit] or [goal.strip()[:80]]
