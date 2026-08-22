"""Strategic signal detection.

Shared by the tools (which tag items), the decision engine (which reacts to
signals by choosing a different tool) and the insight generator (which uses them
to set priority). One definition, one place.

Precision matters more than recall here. A loose substring match reads
"active-learning acquisition function" in a machine-learning abstract as an
acquisition, which then promotes a routine paper to HIGH priority. Every pattern
below is word-bounded, and the ambiguous ones require commercial context.
"""

from __future__ import annotations

import re

# Signals that represent a real strategic move rather than general activity.
# `benchmark` and `hiring` are informative but do not on their own justify HIGH.
STRATEGIC_SIGNALS: frozenset[str] = frozenset(
    {"launch", "acquisition", "funding", "partnership", "patent", "regulatory"}
)

SIGNAL_LABELS: dict[str, str] = {
    "patent": "a patent or IP filing was referenced",
    "launch": "a product launch or commercial release was referenced",
    "funding": "funding or investment activity was referenced",
    "partnership": "a partnership or collaboration was referenced",
    "acquisition": "an acquisition or merger was referenced",
    "regulatory": "regulatory or legal action was referenced",
    "benchmark": "a benchmark or state-of-the-art claim was made",
    "hiring": "hiring or team movement was referenced",
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    "patent": re.compile(
        r"\b(patent(?:s|ed|ing)?|uspto|epo\b|intellectual property|prior art|"
        r"patent (?:filing|application|portfolio)|assignee)\b",
        re.I,
    ),
    # Commercial launch, not "we release our code".
    "launch": re.compile(
        r"\b(launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|general availability|"
        r"now generally available|now available to|goes on sale|"
        r"(?:product|commercial|public) (?:release|rollout|debut)|"
        r"begins shipping|starts shipping|enters production)\b",
        re.I,
    ),
    # Money moving, not "funding was provided by grant NSF-123".
    "funding": re.compile(
        r"(\braise[sd]?\b[^.]{0,40}\b(?:million|billion|\$)|"
        r"\bseries [a-e]\b|\bfunding round\b|\bventure round\b|"
        r"\b(?:pre-)?seed round\b|\bvaluation of\b|"
        r"\binvest(?:s|ed|ment)\b[^.]{0,30}\$|\$\s?\d+(?:\.\d+)?\s?(?:m|bn|million|billion)\b)",
        re.I,
    ),
    "partnership": re.compile(
        r"\b(partner(?:s|ed|ship|ships)\b|joint venture|strategic alliance|"
        r"co-develop(?:s|ed|ment)?|teams? up with|signed an agreement)\b",
        re.I,
    ),
    # Corporate M&A only. Excludes "acquisition function", "data acquisition",
    # "image acquisition", "language acquisition".
    "acquisition": re.compile(
        r"\b(acquire[sd]?\b(?!\s+(?:data|images?|signals?|knowledge))|"
        r"acquisition of\s+[A-Z]|has acquired|to acquire|"
        r"merger|merges with|buyout|takeover bid)\b",
    ),
    "regulatory": re.compile(
        r"\b(regulator(?:s|y)?\b|antitrust|competition authority|"
        r"opens? (?:an? )?(?:review|inquiry|investigation)|"
        r"lawsuit|sue[sd]?\b|litigation|consent decree|compliance order|"
        r"export controls?)\b",
        re.I,
    ),
    "benchmark": re.compile(
        r"\b(benchmark(?:s|ed|ing)?|state of the art|\bsota\b|outperform(?:s|ed|ing)?|"
        r"leaderboard|evaluation suite)\b",
        re.I,
    ),
    "hiring": re.compile(
        r"\b(hir(?:es|ed|ing)\b|poach(?:es|ed|ing)?|recruit(?:s|ed|ing)?\b[^.]{0,30}\bteam\b|"
        r"joins? (?:as|from)\b|head(?:ed)? of [a-z ]{3,20} (?:joins|leaves|departs)|"
        r"talent war)\b",
        re.I,
    ),
}

# Contexts that specifically negate a match, checked after the pattern hits.
_NEGATIONS: dict[str, re.Pattern[str]] = {
    "acquisition": re.compile(
        r"\b(acquisition function|active[- ]learning acquisition|data acquisition|"
        r"image acquisition|signal acquisition|language acquisition|"
        r"knowledge acquisition|acquisition time)\b",
        re.I,
    ),
    "funding": re.compile(r"\b(grant number|funded by (?:the )?(?:nsf|nih|erc|darpa))\b", re.I),
}


def detect_signals(*texts: str) -> list[str]:
    """Return the strategic signals present in the supplied text."""
    blob = " ".join(t for t in texts if t)
    if not blob:
        return []
    found: list[str] = []
    for name, pattern in _PATTERNS.items():
        if not pattern.search(blob):
            continue
        negation = _NEGATIONS.get(name)
        if negation is not None and negation.search(blob) and not _has_clean_hit(
            pattern, negation, blob
        ):
            continue
        found.append(name)
    return found


def _has_clean_hit(
    pattern: re.Pattern[str], negation: re.Pattern[str], blob: str
) -> bool:
    """True when at least one match sits outside every negated span."""
    negated = [m.span() for m in negation.finditer(blob)]
    for match in pattern.finditer(blob):
        start, end = match.span()
        if not any(ns <= start and end <= ne for ns, ne in negated):
            return True
    return False


def label(signal: str) -> str:
    return SIGNAL_LABELS.get(signal, signal)
