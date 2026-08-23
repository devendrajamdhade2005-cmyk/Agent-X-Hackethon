"""Redaction — the gate everything passes through before persistence or export.

Two separate concerns, deliberately not conflated:

  * **Secrets** must never leave the process. Keys, tokens and auth headers are
    matched by name *and* by shape, because a value can arrive under an unexpected
    key (a provider error body that quotes the request URL, for example).
  * **Chain-of-thought** must never be stored. The project already has a rule that
    the activity log carries safe decision *summaries* rather than private
    deliberation; traces follow the same rule, so prompt spans record a template id,
    a version and token counts — never prompt or completion text.

Redaction is applied at write time in the tracer, not at read time in the API, so a
secret cannot sit in memory waiting for someone to forget to filter it.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

# Field names whose values are never recorded, matched case-insensitively on a
# substring so `x_api_token`, `GEMINI_API_KEY` and `authorization` all hit.
_SENSITIVE_KEYS = (
    "api_key", "apikey", "secret", "token", "password", "passwd", "credential",
    "authorization", "auth", "bearer", "cookie", "session", "private_key",
    "access_key", "client_secret", "signature", "database_url", "dsn",
)

# Keys that carry model input/output text. Prompt *metadata* is kept; prompt
# *content* is not, because that is where chain-of-thought and user data live.
_PROMPT_CONTENT_KEYS = (
    "prompt", "system", "user", "messages", "completion", "response_text",
    "raw_response", "reasoning", "thought", "chain_of_thought", "deliberation",
    "raw_text", "content",
)

# Value shapes that look like credentials regardless of the key they arrived under.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),                 # OpenAI-style
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),                   # GitHub classic
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),           # GitHub fine-grained
    re.compile(r"\btvly-[A-Za-z0-9\-]{10,}\b"),                # Tavily
    re.compile(r"\bs2k-[A-Za-z0-9]{10,}\b"),                   # Semantic Scholar
    re.compile(r"\bpub_[0-9a-f]{20,}\b"),                      # newsdata.io
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),                # Google API key
    re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b"),                # Google short-lived
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),                       # long hex secrets
    # `key=value` credential assignments. Stops at a separator so redacting one
    # query parameter does not swallow the rest of the URL and destroy useful
    # trace context.
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret)\s*[=:]\s*[^&\s,;)\]}\"']+"),
)

# Query parameters that carry credentials in URLs.
_URL_SECRET_PARAM = re.compile(
    r"(?i)([?&](?:api_?key|key|token|access_token|apikey|auth)=)[^&\s]+"
)

MAX_TEXT = 400


def scrub_text(text: Any, *, limit: int = MAX_TEXT) -> str:
    """Redact credential-shaped substrings and truncate. Safe on any input."""
    if text is None:
        return ""
    out = str(text)
    out = _URL_SECRET_PARAM.sub(rf"\1{REDACTED}", out)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    if len(out) > limit:
        out = out[: limit - 1] + "…"
    return out


def scrub_value(key: str, value: Any, *, depth: int = 0) -> Any:
    """Redact one attribute by key and by value shape."""
    low = str(key).lower()
    if any(s in low for s in _SENSITIVE_KEYS):
        return REDACTED
    if any(p == low or low.endswith("_" + p) for p in _PROMPT_CONTENT_KEYS):
        # Keep the fact that content existed, never the content itself.
        length = len(str(value)) if value is not None else 0
        return f"<omitted: {length} chars>"
    return scrub_attributes(value, depth=depth + 1)


def scrub_attributes(value: Any, *, depth: int = 0) -> Any:
    """Recursively redact a JSON-ish structure. Bounded depth and width."""
    if depth > 6:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {
            str(k): scrub_value(str(k), v, depth=depth)
            for k, v in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple, set)):
        return [scrub_attributes(v, depth=depth + 1) for v in list(value)[:40]]
    # Unknown object: record its type, not its repr, which could embed anything.
    return f"<{type(value).__name__}>"


def safe_error_message(exc: BaseException | str) -> str:
    """A short, secret-free description of a failure."""
    raw = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return scrub_text(raw, limit=240)


def contains_secret(text: str) -> bool:
    """Test hook: does this text still look like it carries a credential?"""
    probe = str(text)
    return any(p.search(probe) for p in _SECRET_PATTERNS) or bool(
        _URL_SECRET_PARAM.search(probe)
    )
