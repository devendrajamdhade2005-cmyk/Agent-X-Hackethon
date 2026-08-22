"""LLM layer: provider-agnostic reasoning with a deterministic fallback.

Providers
---------
* **gemini** (default) — Google Generative Language API over plain httpx. No SDK
  dependency, and `responseMimeType: application/json` gives us structured output
  natively instead of prompt-and-pray JSON parsing.
* **anthropic** — optional, kept behind the same interface.
* **none** — no credential, so the agent uses its heuristic reasoner.

Two hard requirements shape this module:

1. The agent must run with zero API keys. `LLMClient.available` is False in that
   case and every caller has a heuristic path. The agent still plans, decides and
   prioritizes — it just reports `reasoner: heuristic` instead of pretending a
   model was involved.

2. A bad or rejected credential must degrade, not crash. `verify()` probes the
   provider once; on rejection the client marks itself unavailable and records
   why, and the activity log says so in plain language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import settings

# Rough blended $/1M tokens for the run cost estimate shown in the UI.
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini": (0.30, 2.50),
    "opus": (15.0, 75.0),
    "sonnet": (3.0, 15.0),
    "haiku": (0.80, 4.0),
}
_DEFAULT_PRICE = (0.30, 2.50)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

MAX_LLM_ATTEMPTS = 2
TRANSIENT_STATUS = frozenset({408, 409, 500, 502, 503, 504})

# Latency guards. A hackathon demo that hangs is a demo that failed.
VERIFY_TIMEOUT_SECONDS = 15.0   # startup credential probe
CALL_TIMEOUT_SECONDS = 30.0     # any single model call
RUN_TIME_BUDGET_SECONDS = 75.0  # total model time per agent run

# Errors that will not fix themselves — the client stops using the provider.
_FATAL_MARKERS = (
    "401",
    "403",
    "credential",
    "permission_denied",
    "denied access",
    "api key not valid",
    "api_key_invalid",
)

# Quota exhaustion is not transient within a run. Retrying it just adds seconds of
# backoff to every remaining step, so treat it as "stop using the model now" and
# let the deterministic reasoner finish the job.
_QUOTA_MARKERS = (
    "exceeded your current quota",
    "check your plan and billing",
    "resource_exhausted",
    "quota exceeded",
    "rate limit",
)


def _is_fatal(error: str) -> bool:
    low = (error or "").lower()
    return any(marker in low for marker in _FATAL_MARKERS + _QUOTA_MARKERS)


def _is_quota(error: str) -> bool:
    low = (error or "").lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


async def _backoff(attempt: int) -> None:
    import asyncio
    import random

    await asyncio.sleep(0.5 * attempt + random.uniform(0, 0.25))


_SUGGESTED_MODEL = re.compile(r"use\s+models/([A-Za-z0-9._\-]+)", re.I)


def _suggested_model(error_text: str) -> str:
    """Extract 'Please update your code to use models/X' from an API error."""
    match = _SUGGESTED_MODEL.search(error_text or "")
    return match.group(1) if match else ""


@dataclass
class LlmUsageTotals:
    calls: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    by_purpose: dict[str, int] = field(default_factory=dict)

    def add(self, purpose: str, tin: int, tout: int, model: str) -> None:
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        rate_in, rate_out = _DEFAULT_PRICE
        for key, price in _PRICES.items():
            if key in model.lower():
                rate_in, rate_out = price
                break
        self.cost_usd += (tin / 1_000_000) * rate_in + (tout / 1_000_000) * rate_out
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "by_purpose": self.by_purpose,
        }


# ─────────────────────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────────────────────
class _Provider:
    name = "none"

    async def verify(self) -> tuple[bool, str]:
        return False, "no provider configured"

    async def complete_json(
        self, *, system: str, user: str, max_tokens: int, temperature: float
    ) -> tuple[dict[str, Any] | None, int, int, str]:
        """Returns (parsed_json, tokens_in, tokens_out, error)."""
        return None, 0, 0, "no provider configured"


class GeminiProvider(_Provider):
    """Google Generative Language API via httpx."""

    name = "gemini"

    # Tried in order when the configured model is not usable by this key.
    # Ordered by measured latency and availability, not by version number.
    # Benchmarked on the same prompt: 3.5-flash ~1.1s, 3.1-flash-lite ~1.4s,
    # while 3.7-flash returned 503 "high demand" repeatedly and took 9-26s when it
    # did answer. The `-latest` aliases route to the busiest pool, so they sit last.
    PREFERRED = (
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash",
    )

    # Listed models that cannot do plain text reasoning for us.
    _EXCLUDE = (
        "image", "tts", "audio", "robotics", "lyria", "banana", "gemma",
        "computer-use", "deep-research", "antigravity", "omni", "embedding",
    )

    def __init__(self, api_key: str, model: str, *, timeout: float = 45.0) -> None:
        self.api_key = api_key
        self.model = model or self.PREFERRED[0]
        self.timeout = timeout
        self._verified = False
        self._available_models: list[str] = []
        self._dead_models: set[str] = set()
        self.model_switches: list[str] = []
        self.supports_thinking_config = True

    def _model_candidates(self) -> list[str]:
        """Current model first, then healthy alternatives, newest-first."""
        out = [self.model]
        pool = [
            m
            for m in self.PREFERRED
            if m not in out
            and m not in self._dead_models
            and (not self._available_models or m in self._available_models)
        ]
        out.extend(pool[:2])  # two failovers is enough; beyond that, use heuristics
        return out

    # ── auth ────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        """API-key auth only.

        Verified against the live endpoint: `x-goog-api-key` is accepted, while
        sending the same key as `Authorization: Bearer` is rejected with 401
        ("Expected OAuth 2 access token"). So we must not add a Bearer header.
        """
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    async def verify(self) -> tuple[bool, str]:
        """List models once to confirm the credential and pin a usable model."""
        if self._verified:
            return True, ""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{GEMINI_BASE}/models", headers=self._headers(), params={"pageSize": 100}
                )
        except httpx.HTTPError as exc:
            return False, f"network error reaching Gemini: {exc}"

        if resp.status_code in (401, 403):
            return False, f"credential rejected by Gemini ({resp.status_code})"
        if resp.status_code >= 400:
            return False, f"Gemini models endpoint returned {resp.status_code}"

        try:
            payload = resp.json()
        except ValueError:
            return False, "Gemini models endpoint returned a non-JSON body"

        ids = [
            str(m.get("name", "")).split("/")[-1]
            for m in payload.get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]
        if not ids:
            return False, "no models available to this credential"

        self._available_models = ids
        if self.model not in ids:
            self.model = self._pick_model(ids)

        # Listing models is not proof of generation access: a key can enumerate
        # models and still be denied generateContent (project not enabled for the
        # API). Probe generation for real so the agent knows before it plans.
        #
        # Hard-bounded: a slow or overloaded model must not stall startup. If the
        # probe runs long we proceed optimistically and let the first real call
        # deal with retries and model failover.
        import asyncio as _asyncio

        try:
            parsed, _, _, error = await _asyncio.wait_for(
                self.complete_json(
                    system="Reply with ONLY a JSON object.",
                    user='Return exactly: {"ready": true}',
                    max_tokens=256,
                    temperature=0.0,
                    retry_on_truncation=False,
                ),
                timeout=VERIFY_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return True, (
                f"generation probe timed out after {VERIFY_TIMEOUT_SECONDS:.0f}s; "
                f"proceeding and will retry in use"
            )

        if parsed is None:
            # Distinguish "this key will never work" from "the service is busy".
            # Disabling the provider over a 503 would needlessly drop us to
            # heuristics for the whole run.
            if _is_quota(error):
                return False, f"quota exhausted — {error}"
            if _is_fatal(error):
                return False, f"generation denied — {error}"
            return True, f"generation probe was inconclusive ({error}); will retry in use"

        self._verified = True
        return True, ""

    def _pick_model(self, ids: list[str]) -> str:
        """Choose a text-reasoning model, newest-first."""
        text_models = [
            i for i in ids if not any(bad in i.lower() for bad in self._EXCLUDE)
        ]
        for preferred in self.PREFERRED:
            if preferred in text_models:
                return preferred
        flash = [i for i in text_models if "flash" in i]
        return (flash or text_models or ids)[0]

    # ── generation ──────────────────────────────────────────
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        retry_on_truncation: bool = True,
    ) -> tuple[dict[str, Any] | None, int, int, str]:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            # Native structured output — no fenced-code cleanup needed.
            "responseMimeType": "application/json",
        }
        # Extended thinking is the dominant latency cost and buys nothing for
        # schema-constrained extraction: measured 2.3s → 1.1s on the same prompt
        # with identical output quality. Disabled unless a model rejects the field.
        if self.supports_thinking_config:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        payload: dict[str, Any] | None = None
        last_error = "no response from Gemini"

        # Outer loop rotates models, inner loop retries the current one. Popular
        # aliases like gemini-flash-latest return 503 under load far more often
        # than a specific pinned model, so failing over is worth more than
        # retrying the same endpoint harder.
        for model in self._model_candidates():
            if model != self.model:
                self.model_switches.append(f"{self.model} → {model}")
                self.model = model

            for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
                url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(url, headers=self._headers(), json=body)
                except httpx.TimeoutException:
                    last_error = f"timeout after {self.timeout:.0f}s"
                    if attempt < MAX_LLM_ATTEMPTS:
                        await _backoff(attempt)
                        continue
                    break
                except httpx.HTTPError as exc:
                    return None, 0, 0, f"network error: {exc}"

                if resp.status_code < 400:
                    try:
                        payload = resp.json()
                    except ValueError:
                        return None, 0, 0, "non-JSON response body"
                    break

                detail = ""
                try:
                    detail = str((resp.json().get("error") or {}).get("message", ""))[:250]
                except ValueError:
                    detail = resp.text[:250]
                last_error = f"HTTP {resp.status_code}: {detail}"

                if _is_fatal(last_error):
                    return None, 0, 0, last_error

                # Older/other models reject thinkingConfig outright. Drop it and
                # retry rather than losing the model over an optional field.
                if (
                    resp.status_code == 400
                    and self.supports_thinking_config
                    and "thinking" in detail.lower()
                ):
                    self.supports_thinking_config = False
                    generation_config.pop("thinkingConfig", None)
                    continue

                # Retired-but-listed model: Google names the replacement.
                if resp.status_code == 404:
                    suggested = _suggested_model(detail)
                    if suggested and suggested not in self._dead_models:
                        self._dead_models.add(self.model)
                        self.model_switches.append(f"{self.model} → {suggested}")
                        self.model = suggested
                        continue
                    self._dead_models.add(self.model)
                    break

                if resp.status_code == 429:
                    return None, 0, 0, last_error  # quota — no point retrying
                if resp.status_code in TRANSIENT_STATUS and attempt < MAX_LLM_ATTEMPTS:
                    await _backoff(attempt)
                    continue
                break

            if payload is not None:
                break

        if payload is None:
            return None, 0, 0, last_error

        usage = payload.get("usageMetadata") or {}
        tin = int(usage.get("promptTokenCount") or 0)
        tout = int(usage.get("candidatesTokenCount") or 0)

        candidates = payload.get("candidates") or []
        if not candidates:
            blocked = (payload.get("promptFeedback") or {}).get("blockReason")
            return None, tin, tout, f"no candidates returned{f' ({blocked})' if blocked else ''}"

        text = "".join(
            part.get("text", "")
            for part in ((candidates[0].get("content") or {}).get("parts") or [])
        )
        parsed = extract_json(text)
        if parsed is not None:
            return parsed, tin, tout, ""

        finish = str(candidates[0].get("finishReason") or "unknown")
        thoughts = int(usage.get("thoughtsTokenCount") or 0)
        # Reasoning models can spend the entire output budget on internal thinking
        # and return an empty answer. Retry once with real headroom rather than
        # reporting a malformed response.
        if retry_on_truncation and (finish.upper() == "MAX_TOKENS" or (thoughts and not text)):
            bumped = min(max_tokens * 4, 8192)
            if bumped > max_tokens:
                return await self.complete_json(
                    system=system,
                    user=user,
                    max_tokens=bumped,
                    temperature=temperature,
                    retry_on_truncation=False,
                )
        return None, tin, tout, f"response was not valid JSON (finishReason={finish})"


class AnthropicProvider(_Provider):
    """Optional Claude path, same interface."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key, max_retries=1)
        return self._client

    async def verify(self) -> tuple[bool, str]:
        try:
            listing = await self._ensure().models.list(limit=40)
        except Exception as exc:  # noqa: BLE001
            return False, f"credential rejected by Anthropic: {type(exc).__name__}"
        ids = [m.id for m in getattr(listing, "data", [])]
        if ids and self.model not in ids:
            self.model = next(
                (i for i in ids if "sonnet" in i.lower()),
                next((i for i in ids if "haiku" in i.lower()), ids[0]),
            )
        return True, ""

    async def complete_json(
        self, *, system: str, user: str, max_tokens: int, temperature: float
    ) -> tuple[dict[str, Any] | None, int, int, str]:
        try:
            resp = await self._ensure().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "{"},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return None, 0, 0, f"{type(exc).__name__}: {exc}"

        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)
        text = "{" + "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        )
        parsed = extract_json(text)
        return parsed, tin, tout, "" if parsed else "response was not valid JSON"


# ─────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────
class LLMClient:
    """Safe to construct with no credentials at all."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.usage = LlmUsageTotals()
        self.last_error: str = ""
        self.disabled_reason: str = ""
        self.seconds_spent: float = 0.0
        self.budget_usd = budget_usd if budget_usd is not None else settings.llm_run_budget_usd
        self._verified = False

        chosen = (provider or settings.llm_provider or "auto").strip().lower()
        gemini_key = (api_key if provider == "gemini" else None) or settings.gemini_api_key
        anthropic_key = (api_key if provider == "anthropic" else None) or settings.anthropic_api_key

        if chosen == "auto":
            chosen = "gemini" if gemini_key.strip() else (
                "anthropic" if anthropic_key.strip() else "none"
            )

        if chosen == "gemini" and gemini_key.strip():
            self.provider: _Provider = GeminiProvider(
                gemini_key.strip(), (model or settings.gemini_model).strip()
            )
        elif chosen == "anthropic" and anthropic_key.strip():
            self.provider = AnthropicProvider(
                anthropic_key.strip(), (model or settings.anthropic_model).strip()
            )
        else:
            self.provider = _Provider()
            self.disabled_reason = "no LLM credential configured"

    # ── capability ──────────────────────────────────────────
    @property
    def model(self) -> str:
        return getattr(self.provider, "model", "insightpulse-heuristic-v2")

    @property
    def configured(self) -> bool:
        return self.provider.name != "none"

    @property
    def available(self) -> bool:
        return self.configured and not self.disabled_reason and not self.budget_exhausted

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_usd > 0 and self.usage.cost_usd >= self.budget_usd

    @property
    def reasoner_name(self) -> str:
        if not self.available:
            return "heuristic"
        return f"{self.provider.name}:{self.model}"

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "model": self.model,
            "available": self.available,
            "verified": self._verified,
            "disabled_reason": self.disabled_reason,
            "last_error": self.last_error,
            "budget_usd": self.budget_usd,
            "seconds_spent": round(self.seconds_spent, 2),
            "usage": self.usage.to_dict(),
        }

    # ── lifecycle ───────────────────────────────────────────
    async def verify(self) -> tuple[bool, str]:
        """Probe the credential once. Marks the client unavailable on rejection."""
        if not self.configured:
            return False, self.disabled_reason
        if self._verified:
            return True, ""
        ok, reason = await self.provider.verify()
        if ok:
            self._verified = not reason  # reason present = inconclusive probe
            if reason:
                self.last_error = reason
            return True, reason
        self.disabled_reason = reason
        self.last_error = reason
        return False, reason

    # ── generation ──────────────────────────────────────────
    async def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any] | None:
        """Ask for one JSON object. Returns None on any failure — callers fall back."""
        if not self.available:
            return None

        import asyncio as _asyncio
        import time as _time

        started = _time.perf_counter()
        try:
            parsed, tin, tout, error = await _asyncio.wait_for(
                self.provider.complete_json(
                    system=system,
                    user=user,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            parsed, tin, tout = None, 0, 0
            error = f"call exceeded {CALL_TIMEOUT_SECONDS:.0f}s"

        self.seconds_spent += _time.perf_counter() - started
        if tin or tout or parsed is not None:
            self.usage.add(purpose, tin, tout, self.model)
        if parsed is None:
            self.usage.failures += 1
            self.last_error = error or "unknown LLM error"
            # An auth failure will not fix itself: stop using the provider.
            # Capacity errors will, so those just fall back for this one call.
            if _is_fatal(self.last_error):
                self.disabled_reason = self.last_error
            elif self.usage.failures >= 3 and self.usage.calls == 0:
                self.disabled_reason = (
                    f"{self.usage.failures} consecutive LLM failures — "
                    f"switching to the heuristic reasoner ({self.last_error})"
                )

        # A run has to finish in a demoable amount of time. Once the model has
        # consumed its share of the wall clock, the rest of the run completes on
        # heuristics rather than making the user wait.
        if self.seconds_spent > RUN_TIME_BUDGET_SECONDS and not self.disabled_reason:
            self.disabled_reason = (
                f"model spent {self.seconds_spent:.0f}s of the "
                f"{RUN_TIME_BUDGET_SECONDS:.0f}s reasoning budget — finishing on heuristics"
            )
        return parsed


# ─────────────────────────────────────────────────────────────
# JSON extraction / repair
# ─────────────────────────────────────────────────────────────
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from a model response."""
    if not text:
        return None
    candidate = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    for attempt in (candidate, _balanced_object(candidate)):
        if not attempt:
            continue
        for repaired in (attempt, _TRAILING_COMMA.sub(r"\1", attempt)):
            try:
                value = json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"items": value}
    return None


def _balanced_object(text: str) -> str:
    """Slice out the first balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def coerce_str_list(value: Any, *, limit: int = 12, max_len: int = 120) -> list[str]:
    """Model output hygiene: accept a list or a delimited string, return clean strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = list(re.split(r"[,;\n]", value))
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()[:max_len]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
        if len(out) >= limit:
            break
    return out


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(float(value))))
    except (TypeError, ValueError):
        return default
