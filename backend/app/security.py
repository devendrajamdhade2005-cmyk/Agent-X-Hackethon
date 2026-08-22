"""Input hygiene and outbound-URL safety.

Everything a user types reaches an external API or a model prompt, so it is
cleaned here first. Kept deliberately small: the agent endpoints are guarded by a
shared-secret header (`app/api/agent.py::require_token`), not by user accounts.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

# Control characters, zero-width and bidi-override characters. These are the
# building blocks of prompt-injection and homoglyph tricks, so they never survive
# into a query or a prompt.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2028\u2029\ufeff\u202a-\u202e]")


def clean_text(value: str, *, max_len: int = 500) -> str:
    """Strip control/zero-width characters and clamp length."""
    if not value:
        return ""
    return _CONTROL.sub("", str(value)).strip()[:max_len]


def clean_terms(values: list[str] | None, *, max_items: int = 40, max_len: int = 120) -> list[str]:
    """Clean a list of user terms, dropping blanks and case-insensitive duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for v in values or []:
        c = clean_text(str(v), max_len=max_len)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
        if len(out) >= max_items:
            break
    return out


_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "169.254.169.254"}


def validate_outbound_url(url: str) -> str:
    """SSRF guard for any user-supplied outbound target (e.g. a webhook)."""
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must be http(s)")
    host = (parsed.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTS:
        raise ValueError("host is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url  # hostname, not a literal IP
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError("URL may not target a private network")
    return url
