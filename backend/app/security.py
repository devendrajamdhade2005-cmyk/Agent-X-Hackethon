"""Auth: bcrypt password hashing + JWT bearer tokens, enforced server-side."""

from __future__ import annotations

import ipaddress
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)


# ── passwords ───────────────────────────────────────────────
def hash_password(raw: str) -> str:
    return bcrypt.hashpw(raw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── tokens ──────────────────────────────────────────────────
def create_access_token(user_id: str, email: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
        "iss": "insightpulse",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM], issuer="insightpulse")


def resolve_user_from_token(token: str, db: Session) -> User | None:
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        return None
    return db.get(User, payload.get("sub", ""))


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = resolve_user_from_token(creds.credentials, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ── input hygiene ───────────────────────────────────────────
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2028\u2029\ufeff]")


def clean_text(value: str, *, max_len: int = 500) -> str:
    """Strip control/zero-width chars and clamp length. Applied to every user field."""
    if not value:
        return ""
    return _CONTROL.sub("", str(value)).strip()[:max_len]


def clean_terms(values: list[str] | None, *, max_items: int = 40, max_len: int = 120) -> list[str]:
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
    """SSRF guard for user-supplied webhook targets."""
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Webhook URL must be http(s)")
    host = (parsed.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTS:
        raise ValueError("Webhook host is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url  # hostname, not a literal IP
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError("Webhook URL may not target a private network")
    return url
