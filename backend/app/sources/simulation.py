"""Deterministic synthetic source data — the reason this demo cannot die.

Simulation Mode is not a mock for tests: it is a product feature. Rate limits,
expired keys and flaky Wi-Fi are the most common cause of a dead hackathon demo,
so every connector can serve plausible, deterministic, clearly-labelled items
instead. Seeded by (source, query, day) so the same run reproduces exactly, and
consecutive runs on the same day produce overlapping items — which conveniently
also exercises the dedup path.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime, timedelta

from .base import RawItem, SourceQuery

# ── vocabularies ────────────────────────────────────────────
_METHODS = [
    "sparse mixture-of-experts routing",
    "self-distilled curriculum pretraining",
    "physics-informed surrogate modelling",
    "differentiable simulation",
    "retrieval-augmented planning",
    "low-rank adapter fine-tuning",
    "quantized inference kernels",
    "active-learning acquisition",
    "graph neural surrogate search",
    "Bayesian experimental design",
]

_CLAIMS = [
    "reports a 3.4x throughput gain at equal accuracy",
    "cuts required training data by 62%",
    "reaches state of the art on three public benchmarks",
    "reduces inference cost per query by 41%",
    "closes 80% of the gap to the supervised ceiling",
    "improves cycle life by 27% in accelerated testing",
    "halves the calibration effort needed for deployment",
    "generalizes to two unseen material families",
]

_INSTITUTIONS = [
    "MIT CSAIL",
    "Stanford SAIL",
    "ETH Zurich",
    "Tsinghua University",
    "KAIST",
    "Max Planck Institute",
    "University of Toronto",
    "Oxford Engineering Science",
    "Berkeley BAIR",
    "National University of Singapore",
]

_SURNAMES = [
    "Okafor",
    "Lindqvist",
    "Nakamura",
    "Rahman",
    "Delgado",
    "Fitzgerald",
    "Sokolova",
    "Mwangi",
    "Bhattacharya",
    "Haugen",
    "Oyelaran",
    "Vasquez",
]

_OUTLETS = [
    ("Reuters", "reuters.com"),
    ("TechCrunch", "techcrunch.com"),
    ("The Verge", "theverge.com"),
    ("Ars Technica", "arstechnica.com"),
    ("Bloomberg", "bloomberg.com"),
    ("VentureBeat", "venturebeat.com"),
    ("IEEE Spectrum", "spectrum.ieee.org",),
    ("PR Newswire", "prnewswire.com"),
]

# (headline template, body template). Bodies differ per shape so the generated
# corpus does not read as one sentence repeated — which also keeps the dedup and
# relevance logic honest during offline demos.
_NEWS_SHAPES: list[tuple[str, str]] = [
    (
        "{company} commits ${n}M to scale {kw}",
        "{company} has approved ${n}M of additional funding to move its {kw} programme "
        "from prototype to volume. The company said the investment covers tooling and a "
        "second production line, with first output targeted for next year.",
    ),
    (
        "{company} opens a dedicated {kw} research unit",
        "{company} is standing up a separate research group focused on {kw}, reporting "
        "directly to its CTO. Around 40 researchers are being reassigned internally, "
        "which analysts read as a long-horizon bet rather than a product push.",
    ),
    (
        "{company} and a tier-1 supplier partner on {kw}",
        "{company} has signed a multi-year supply and co-development agreement covering "
        "{kw}. Neither party disclosed terms, but the partnership gives {company} "
        "capacity it would otherwise have had to build itself.",
    ),
    (
        "{company} moves {kw} from pilot into production",
        "After two years of pilot deployments, {company} says its {kw} work is now "
        "running in production. Executives cited internal reliability figures that have "
        "not been independently verified.",
    ),
    (
        "{company} hires a {kw} team away from a rival",
        "{company} has recruited a group of senior engineers who previously led {kw} work "
        "at a competitor. Talent movement of this size usually precedes a visible product "
        "shift by two to three quarters.",
    ),
    (
        "Regulators open a review of {company}'s {kw} claims",
        "A regulator has opened a review into performance claims {company} made about its "
        "{kw} products. The company says it stands by the figures and is cooperating.",
    ),
    (
        "{company} publishes benchmark results for its {kw} stack",
        "{company} released benchmark numbers for its {kw} stack, claiming a material lead "
        "over the nearest alternative. The methodology is published but the test harness "
        "is not, so the results are not yet reproducible.",
    ),
    (
        "{company} acquires a {kw} startup",
        "{company} has acquired a small team working on {kw}, folding the technology into "
        "its existing platform. The deal appears to be talent-and-IP rather than revenue.",
    ),
]

_SUBREDDITS = ["MachineLearning", "singularity", "hardware", "engineering", "science", "startups", "patents"]

_SOCIAL_SHAPES = [
    "Has anyone actually reproduced the {kw} numbers?",
    "{company}'s {kw} claims look inflated — here's why",
    "Six months running {kw} in production: what broke",
    "Why is nobody talking about {company}'s {kw} filing?",
    "{kw} benchmark thread — post your configs",
    "Interviewed at {company}, the {kw} roadmap surprised me",
]

_REPO_SHAPES = [
    "{slug}-core v{maj}.{min}.0 — {feature}",
    "{slug}-toolkit v{maj}.{min}.0 adds {feature}",
    "open-sourcing {slug}: {feature}",
]

_FEATURES = [
    "streaming evaluation harness",
    "ONNX + TensorRT export path",
    "distributed checkpoint sharding",
    "a reference dataset loader",
    "8-bit quantization support",
    "a deterministic replay mode",
]

_SENTIMENT_TAIL = [
    "Commenters are broadly positive but want independent replication.",
    "The thread is skeptical about the methodology.",
    "Mixed reception: strong engineering, unclear economics.",
    "Strongly negative — several practitioners report contradictory results.",
]


def _rng(source: str, q: SourceQuery, salt: str = "") -> random.Random:
    """Deterministic per (source, query, day) so runs are reproducible."""
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    key = f"{source}|{q.query}|{','.join(sorted(q.keywords))}|{day}|{salt}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def _topic(rng: random.Random, q: SourceQuery) -> str:
    pool = q.keywords or [q.query] or ["emerging technology"]
    return rng.choice([p for p in pool if p]) or "emerging technology"


def _company(rng: random.Random, q: SourceQuery) -> str:
    pool = q.competitors or ["Northwind Labs", "Helios Dynamics", "Kite Systems"]
    return rng.choice(pool)


def _slug(text: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in text.lower()).split())[:40] or "topic"


def _when(rng: random.Random, q: SourceQuery) -> datetime:
    span = max(1, min(q.since_days, 60))
    return (
        datetime.now(UTC).replace(tzinfo=None)
        - timedelta(days=rng.randint(0, span), hours=rng.randint(0, 23))
    )


# ── generators ──────────────────────────────────────────────
def simulate_research(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    rng = _rng(source, q)
    items: list[RawItem] = []
    for i in range(count):
        kw = _topic(rng, q)
        method = rng.choice(_METHODS)
        claim = rng.choice(_CLAIMS)
        inst = rng.choice(_INSTITUTIONS)
        authors = ", ".join(
            f"{rng.choice('ABCDEFGHJKLMNPRSTV')}. {rng.choice(_SURNAMES)}" for _ in range(rng.randint(2, 4))
        )
        ident = f"{rng.randint(2401, 2612)}.{rng.randint(10000, 99999)}"
        cites = rng.choice([0, 0, 1, 3, 8, 17, 44])
        items.append(
            RawItem(
                source_type="research",
                source_name=source,
                title=f"{method.capitalize()} for {kw}",
                url=f"https://arxiv.org/abs/{ident}",
                raw_text=(
                    f"We study {kw} through {method}. Across three benchmark suites the "
                    f"approach {claim}. Work conducted at {inst}. We release code and the "
                    f"evaluation harness, and discuss failure modes under distribution shift."
                ),
                author=authors,
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{ident}",
                credibility="high",
                is_simulated=True,
                meta={
                    "citation_count": cites,
                    "venue": rng.choice(["NeurIPS", "ICML", "ICLR", "Nature Energy", "preprint"]),
                    "institution": inst,
                    "categories": ["cs.LG", "cs.AI"],
                },
            )
        )
    return items


def simulate_patent(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    rng = _rng(source, q)
    items: list[RawItem] = []
    for _ in range(count):
        kw = _topic(rng, q)
        method = rng.choice(_METHODS)
        # Bias toward tracked competitors: assignee matches are the high-value signal.
        assignee = _company(rng, q) if rng.random() < 0.7 else rng.choice(_INSTITUTIONS)
        num = f"US{rng.randint(11000000, 12999999)}B2"
        filed = _when(rng, q) - timedelta(days=rng.randint(200, 700))
        items.append(
            RawItem(
                source_type="patent",
                source_name=source,
                title=f"System and method for {kw} using {method}",
                url=f"https://patents.google.com/patent/{num}/en",
                raw_text=(
                    f"A system for {kw} is disclosed. The described embodiment applies {method} "
                    f"to reduce latency and improve yield relative to conventional pipelines. "
                    f"Claims cover the control loop, the data representation, and the "
                    f"manufacturing process window."
                ),
                author=assignee,
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{num}",
                credibility="high",
                is_simulated=True,
                meta={
                    "assignee": assignee,
                    "patent_number": num,
                    "filing_date": filed.date().isoformat(),
                    "cpc": rng.choice(["G06N 3/08", "H01M 10/0562", "G06F 9/50", "B33Y 50/02"]),
                    "claim_count": rng.randint(8, 31),
                },
            )
        )
    return items


def simulate_news(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    rng = _rng(source, q)
    items: list[RawItem] = []
    used: set[int] = set()
    for _ in range(count):
        kw = _topic(rng, q)
        company = _company(rng, q)
        outlet, domain = rng.choice(_OUTLETS)
        # Prefer an unused headline shape so a single run does not repeat itself.
        choices = [i for i in range(len(_NEWS_SHAPES)) if i not in used] or list(
            range(len(_NEWS_SHAPES))
        )
        idx = rng.choice(choices)
        used.add(idx)
        shape, body = _NEWS_SHAPES[idx]
        amount = rng.choice([12, 40, 75, 120, 400])
        headline = shape.format(company=company, kw=kw, n=amount)
        items.append(
            RawItem(
                source_type="news",
                source_name=source,
                title=headline,
                url=f"https://{domain}/{_slug(headline)}-{rng.randint(1000, 9999)}",
                raw_text=body.format(company=company, kw=kw, n=amount),
                author=outlet,
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{_slug(headline)}",
                is_simulated=True,
                meta={"outlet": outlet, "domain": domain, "category": "industry"},
            )
        )
    return items


def simulate_social(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    rng = _rng(source, q)
    items: list[RawItem] = []
    for _ in range(count):
        kw = _topic(rng, q)
        company = _company(rng, q)
        sub = rng.choice(_SUBREDDITS)
        title = rng.choice(_SOCIAL_SHAPES).format(kw=kw, company=company)
        score = rng.choice([4, 18, 63, 140, 380, 910])
        items.append(
            RawItem(
                source_type="social",
                source_name=source,
                title=title,
                url=f"https://reddit.com/r/{sub}/comments/{rng.randint(10**6, 10**7)}/{_slug(title)}",
                raw_text=(
                    f"Posting in r/{sub}. {title} We ran a small internal evaluation on {kw} and "
                    f"the numbers did not line up with the published figures. "
                    f"{rng.choice(_SENTIMENT_TAIL)}"
                ),
                author=f"u/{_slug(rng.choice(_SURNAMES))}{rng.randint(11, 99)}",
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{_slug(title)}",
                credibility="unverified",
                is_simulated=True,
                meta={
                    "subreddit": sub,
                    "score": score,
                    "num_comments": rng.randint(2, 260),
                    "signal_strength": "high" if score > 300 else "medium" if score > 60 else "low",
                },
            )
        )
    return items


def simulate_repo(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    rng = _rng(source, q)
    items: list[RawItem] = []
    for _ in range(count):
        kw = _topic(rng, q)
        company = _company(rng, q)
        owner = _slug(company)
        feature = rng.choice(_FEATURES)
        title = rng.choice(_REPO_SHAPES).format(
            slug=_slug(kw), feature=feature, maj=rng.randint(0, 3), min=rng.randint(1, 24)
        )
        stars = rng.choice([12, 87, 340, 1200, 5600])
        items.append(
            RawItem(
                source_type="repo",
                source_name=source,
                title=f"{owner}/{title}",
                url=f"https://github.com/{owner}/{_slug(kw)}/releases",
                raw_text=(
                    f"Release notes: {feature}. This repository implements {kw} tooling maintained "
                    f"by engineers at {company}. The changelog references an internal deployment "
                    f"and adds a migration guide, which usually signals real production use."
                ),
                author=owner,
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{owner}:{_slug(title)}",
                is_simulated=True,
                meta={
                    "owner": owner,
                    "stars": stars,
                    "language": rng.choice(["Python", "Rust", "C++", "TypeScript"]),
                    "company": company,
                },
            )
        )
    return items


def simulate_web(source: str, q: SourceQuery, count: int) -> list[RawItem]:
    """Open-web results: same event shapes as news, but attributed to live pages."""
    rng = _rng(source, q, salt="web")
    items: list[RawItem] = []
    used: set[int] = set()
    for _ in range(count):
        kw = _topic(rng, q)
        company = _company(rng, q)
        outlet, domain = rng.choice(_OUTLETS)
        choices = [i for i in range(len(_NEWS_SHAPES)) if i not in used] or list(
            range(len(_NEWS_SHAPES))
        )
        idx = rng.choice(choices)
        used.add(idx)
        shape, body = _NEWS_SHAPES[idx]
        amount = rng.choice([15, 60, 90, 250, 500])
        headline = shape.format(company=company, kw=kw, n=amount)
        items.append(
            RawItem(
                source_type="web",
                source_name=source,
                title=headline,
                url=f"https://{domain}/{_slug(headline)}",
                raw_text=body.format(company=company, kw=kw, n=amount),
                author=domain,
                published_at=_when(rng, q),
                external_id=f"sim:{source}:{_slug(headline)}",
                is_simulated=True,
                meta={
                    "outlet": outlet,
                    "domain": domain,
                    "tavily_score": round(rng.uniform(0.55, 0.95), 4),
                    "topic": "news",
                },
            )
        )
    return items


_DISPATCH = {
    "research": simulate_research,
    "patent": simulate_patent,
    "news": simulate_news,
    "social": simulate_social,
    "repo": simulate_repo,
    "web": simulate_web,
}


def simulate(source: str, source_type: str, q: SourceQuery, count: int | None = None) -> list[RawItem]:
    """Entry point used by every connector's `simulate()`."""
    generator = _DISPATCH.get(source_type, simulate_news)
    n = count if count is not None else max(2, min(q.limit, 8))
    return generator(source, q, n)
