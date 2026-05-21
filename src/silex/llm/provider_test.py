"""
Ephemeral connectivity checks for LLM providers (setup wizard + aria doctor --ping).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from silex.llm.catalog import find_model, get_provider_defaults, list_providers
from silex.llm.factory import build_provider
from silex.runtime.settings import RuntimeSettingsStore


class _Ping(BaseModel):
    ok: bool = True


def _known_provider_ids() -> set[str]:
    return {p["id"] for p in list_providers()}


def humanize_llm_error(exc: BaseException, provider_id: str) -> tuple[str, str, str]:
    """
    Map raw exceptions to a short message, remediation hint, and stable code.
    """
    raw = str(exc)
    low = raw.lower()
    class_name = exc.__class__.__name__.lower()
    status_code = getattr(exc, "status_code", getattr(exc, "code", None))

    if "install aria" in low or "install aria[" in low:
        return (
            "Missing optional dependency for this provider.",
            "Install provider extras: pip install \"aria[providers]\"",
            "missing_dependency",
        )
    if (
        status_code in (401, 403)
        or "authentication" in class_name
        or "unauthorized" in class_name
        or "401" in raw
        or "403" in raw
        or ("incorrect api key" in low)
        or ("invalid" in low and "api" in low)
    ):
        return (
            "Authentication was rejected.",
            f"Open the {provider_id} dashboard, create a fresh API key, and paste it again.",
            "invalid_api_key",
        )
    if "429" in raw or "rate limit" in low or "too many requests" in low:
        return (
            "Rate limited by the provider.",
            "Wait a minute and retry, or check quota and billing in the provider dashboard.",
            "rate_limited",
        )
    if "resource exhausted" in low or "quota" in low or "exceeded" in low and "token" in low:
        return (
            "Quota or spending limit may be exceeded.",
            "Check the provider billing page and usage caps.",
            "quota",
        )
    if "model" in low and ("not found" in low or "does not exist" in low or "unknown model" in low):
        return (
            "This model name was not accepted.",
            "Pick another model from `aria models` or the setup list for this provider.",
            "model_not_found",
        )
    if "connection refused" in low or "failed to establish" in low:
        if provider_id == "ollama":
            return (
                "Could not reach the local Ollama endpoint.",
                "Start Ollama or set OLLAMA_HOST / verify http://127.0.0.1:11434 is listening.",
                "ollama_unreachable",
            )
        return (
            "Network connection to the provider failed.",
            "Check internet access, VPN, firewall, and corporate SSL inspection.",
            "network",
        )
    if "timed out" in low or "timeout" in low:
        return (
            "The provider did not respond in time.",
            "Retry when the network is stable; local Ollama may be cold-starting a large model.",
            "timeout",
        )
    if "500" in raw or "502" in raw or "503" in raw or "unavailable" in low:
        return (
            "The provider returned a server-side error.",
            "Retry later; if it persists, check provider status pages.",
            "provider_error",
        )
    # Shorten huge HTML error pages
    trimmed = raw if len(raw) < 320 else raw[:300] + "…"
    return (
        trimmed,
        "If this persists, run `aria doctor --ping` from the same machine and compare the raw error.",
        "unknown",
    )


async def ping_provider(
    provider_id: str,
    api_key: str,
    model_id: str | None = None,
    *,
    base_url: str | None = None,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """
    Run a tiny structured completion using a throwaway settings directory.

    Returns a dict with ok, message, and optionally hint, code, model.
    """
    if provider_id not in _known_provider_ids():
        return {
            "ok": False,
            "message": "Unknown provider.",
            "hint": "Choose a provider from the setup wizard list.",
            "code": "unknown_provider",
        }
    if provider_id != "ollama" and not (api_key or "").strip():
        return {
            "ok": False,
            "message": "API key is required for this provider.",
            "hint": "Paste your key from the provider's developer console.",
            "code": "missing_key",
        }

    defaults = get_provider_defaults(provider_id)
    model_use = (model_id or "").strip() or defaults["fast_model"]
    if model_id and provider_id != "custom" and find_model(provider_id, model_use) is None:
        return {
            "ok": False,
            "message": f"Model `{model_use}` is not in ARIA's catalog for `{provider_id}`.",
            "hint": "Pick a model from the dropdown or run `aria models` to see supported ids.",
            "code": "unknown_model",
        }

    with tempfile.TemporaryDirectory(prefix="aria-provider-test-") as tmp:
        tpath = Path(tmp)
        store = RuntimeSettingsStore(settings_path=tpath / "settings.json", secrets_path=tpath / "secrets.json")
        store.save_settings(
            {
                "setup_completed": True,
                "provider": provider_id,
                "model": model_use,
                "fast_model": model_use,
                "reasoning_model": defaults["reasoning_model"],
                "base_url": base_url or "",
            }
        )
        if provider_id != "ollama":
            store.set_provider_secret(provider_id, api_key.strip())

        client = build_provider(store, usage_tracker=None)
        try:
            client.connect()
        except Exception as exc:
            msg, hint, code = humanize_llm_error(exc, provider_id)
            return {"ok": False, "message": msg, "hint": hint, "code": code, "model": model_use}

        async def _call():
            return await client.complete_json(
                schema=_Ping,
                system_prompt='You must reply with JSON only: {"ok": true} — connectivity check, no prose.',
                user_input='Return exactly: {"ok": true}',
                model_override=model_use,
                temperature=0.0,
                request_kind="connectivity_test",
            )

        try:
            await asyncio.wait_for(_call(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "message": "Provider did not respond before the timeout.",
                "hint": "For Ollama, ensure the model is pulled (`ollama pull ...`). For cloud APIs, check network latency.",
                "code": "timeout",
                "model": model_use,
            }
        except Exception as exc:
            msg, hint, code = humanize_llm_error(exc, provider_id)
            return {"ok": False, "message": msg, "hint": hint, "code": code, "model": model_use}

    return {
        "ok": True,
        "message": f"Connected successfully ({provider_id} / {model_use}).",
        "model": model_use,
    }
