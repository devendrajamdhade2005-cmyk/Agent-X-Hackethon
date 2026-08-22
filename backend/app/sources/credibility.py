"""Source credibility tiering — the cheapest noise reducer available.

A peer-reviewed paper and an anonymous forum post are not equal evidence, and
scoring them identically is what makes an intel feed feel noisy. Tier feeds into
the reasoning step as an explicit, visible score modifier.
"""

from __future__ import annotations

from urllib.parse import urlparse

HIGH = "high"
STANDARD = "standard"
LOW = "low"
UNVERIFIED = "unverified"

# Tier-1 outlets and primary sources.
_HIGH_DOMAINS = {
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "arxiv.org",
    "semanticscholar.org",
    "openalex.org",
    "patentsview.org",
    "patents.google.com",
    "uspto.gov",
    "epo.org",
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "nasa.gov",
    "who.int",
}

# Credible trade press.
_STANDARD_DOMAINS = {
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
    "venturebeat.com",
    "theinformation.com",
    "cnbc.com",
    "forbes.com",
    "zdnet.com",
    "protocol.com",
    "axios.com",
    "semianalysis.com",
    "spectrum.ieee.org",
    "github.com",
    "news.ycombinator.com",
    "theregister.com",
    "nikkei.com",
}

# Aggregators and content farms.
_LOW_DOMAINS = {
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "einpresswire.com",
    "newsbreak.com",
}

# Not editorial content at all. Broad news aggregators index these and return
# automated release notes, listings and directory pages for any technical query
# — e.g. a search for "AI agents" yields PyPI pages like "agent2win 1.0.7".
# They are not low-credibility news, they are *not news*, so they are dropped
# rather than down-weighted. Applied only to the aggregator APIs: a Hacker News
# story that happens to link a package release is still a genuine signal.
_NON_EDITORIAL_DOMAINS = {
    "pypi.org",
    "npmjs.com",
    "crates.io",
    "rubygems.org",
    "packagist.org",
    "nuget.org",
    "hex.pm",
    "pkg.go.dev",
    "metacpan.org",
    "sourceforge.net",
    "freshports.org",
    "libraries.io",
    "changelog.md",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "linkedin.com",
    "coursera.org",
    "udemy.com",
}


# Aggregator wrappers, not sources. A `news.google.com/rss/articles/<base64>`
# link is an opaque redirect, so it cannot be shown as a citation and the real
# article is usually indexed separately anyway. Dropping these keeps the report's
# source provenance honest and removes a whole class of duplicate.
_REDIRECT_WRAPPER_DOMAINS = {
    "news.google.com",
    "news.url.google.com",
    "flipboard.com",
    "headtopics.com",
    "newsnow.co.uk",
    "biztoc.com",
}


def is_redirect_wrapper(url: str) -> bool:
    """True for aggregator links that redirect instead of hosting the article."""
    host = domain_of(url)
    if not host:
        return False
    return any(
        host == domain or host.endswith("." + domain)
        for domain in _REDIRECT_WRAPPER_DOMAINS
    )


def is_non_editorial(url: str) -> bool:
    """True for package registries, job boards and course listings.

    Used to keep automated directory pages out of the news feed.
    """
    host = domain_of(url)
    if not host:
        return False
    return any(
        host == domain or host.endswith("." + domain)
        for domain in _NON_EDITORIAL_DOMAINS
    )


# Score modifier applied per tier (points added to the composite score).
TIER_MODIFIER = {HIGH: 6, STANDARD: 0, LOW: -8, UNVERIFIED: -12}

# Hard ceiling: unverified social chatter can never present as a top-tier signal.
TIER_CEILING = {HIGH: 100, STANDARD: 100, LOW: 88, UNVERIFIED: 78}

_TYPE_DEFAULT = {
    "research": HIGH,
    "patent": HIGH,
    "news": STANDARD,
    "repo": STANDARD,
    # Open-web results are domain-tiered: the URL decides, not the source type.
    "web": STANDARD,
    "social": UNVERIFIED,
}


def domain_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify(url: str, source_type: str) -> str:
    """Tier a single item. Source type sets the floor, domain refines it."""
    default = _TYPE_DEFAULT.get(source_type, STANDARD)
    host = domain_of(url)
    if not host:
        return default
    for domain in _HIGH_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return HIGH if source_type != "social" else UNVERIFIED
    for domain in _LOW_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return LOW if source_type != "social" else UNVERIFIED
    for domain in _STANDARD_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return UNVERIFIED if source_type == "social" else STANDARD
    return default


def label(tier: str) -> str:
    return {
        HIGH: "primary / peer-reviewed source",
        STANDARD: "credible trade source",
        LOW: "low-signal aggregator",
        UNVERIFIED: "unverified social signal",
    }.get(tier, "credible trade source")
