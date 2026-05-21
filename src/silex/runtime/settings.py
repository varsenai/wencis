from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from silex.utils.config import VYN_CONFIG, VYN_SECRETS

SETTINGS_PATH = VYN_CONFIG
SECRETS_PATH = VYN_SECRETS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_private_file(path: Path) -> None:
    """Create a private file with restricted permissions.

    Security note (Windows):
      Windows does not support Unix file permissions (0o600). On Windows,
      the secrets file (data/secrets.json) is readable by any process running
      as the current user. For maximum security on Windows, use environment
      variables (GEMINI_API_KEY, OPENAI_API_KEY, etc.) instead of storing
      API keys in the secrets file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    else:
        # Windows: no chmod equivalent.
        pass


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(default))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return json.loads(json.dumps(default))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def default_settings() -> dict[str, Any]:
    return {
        "version": 1,
        "setup_completed": False,
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "fast_model": "gemini-3.1-flash-lite",
        "reasoning_model": "gemini-3.1-pro",
        "providers": {},
        "security": {
            "require_tool_approvals": True,
            "web_api_key_enabled": False,
            "browser_actions": True,
            "terminal_execution": False,
            "code_apply": False,
            "background_actions": False,
        },
        "telegram": {
            "paired_users": [],
            "pair_codes": [],
            "public_mode": False,
        },
        "usage": {
            "soft_cap_usd": None,
            "warning_threshold_usd": None,
            "disable_expensive_models": False,
        },
        "identity": {
            "assistant_name": "VYN",
            "persona": "",
        },
        "updated_at": _now(),
    }


def default_secrets() -> dict[str, Any]:
    return {
        "version": 1,
        "web_api_key": "",
        "providers": {},
        "updated_at": _now(),
    }


class RuntimeSettingsStore:
    """Simple local JSON-backed settings and secrets store."""

    def __init__(self, settings_path: Path = SETTINGS_PATH, secrets_path: Path = SECRETS_PATH):
        self.settings_path = settings_path
        self.secrets_path = secrets_path
        _ensure_private_file(self.settings_path)
        _ensure_private_file(self.secrets_path)

    def load_settings(self) -> dict[str, Any]:
        data = _read_json(self.settings_path, default_settings())
        merged = default_settings()
        merged.update(data)
        merged["security"] = {**default_settings()["security"], **data.get("security", {})}
        merged["telegram"] = {**default_settings()["telegram"], **data.get("telegram", {})}
        merged["usage"] = {**default_settings()["usage"], **data.get("usage", {})}
        merged["identity"] = {**default_settings()["identity"], **data.get("identity", {})}
        merged["providers"] = data.get("providers", {})
        return merged

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.load_settings()
        merged = {**current, **payload}
        if "security" in payload:
            merged["security"] = {**current.get("security", {}), **payload["security"]}
        if "telegram" in payload:
            merged["telegram"] = {**current.get("telegram", {}), **payload["telegram"]}
        if "usage" in payload:
            merged["usage"] = {**current.get("usage", {}), **payload["usage"]}
        if "identity" in payload:
            merged["identity"] = {**current.get("identity", {}), **payload["identity"]}
        if "providers" in payload:
            merged["providers"] = {**current.get("providers", {}), **payload["providers"]}
        merged["updated_at"] = _now()
        _write_json(self.settings_path, merged)
        return merged

    def load_secrets(self) -> dict[str, Any]:
        data = _read_json(self.secrets_path, default_secrets())
        merged = default_secrets()
        merged.update(data)
        merged["providers"] = data.get("providers", {})
        return merged

    def save_secrets(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.load_secrets()
        merged = {**current, **payload}
        if "providers" in payload:
            merged["providers"] = {**current.get("providers", {}), **payload["providers"]}
        merged["updated_at"] = _now()
        _write_json(self.secrets_path, merged)
        return merged

    def get_provider_secret(self, provider: str, key: str = "api_key") -> str:
        secrets_payload = self.load_secrets()
        return str(secrets_payload.get("providers", {}).get(provider, {}).get(key, "") or "")

    def set_provider_secret(self, provider: str, value: str, key: str = "api_key") -> None:
        secrets_payload = self.load_secrets()
        provider_secrets = dict(secrets_payload.get("providers", {}).get(provider, {}))
        provider_secrets[key] = value
        secrets_payload.setdefault("providers", {})[provider] = provider_secrets
        self.save_secrets(secrets_payload)

    def get_web_api_key(self) -> str:
        return str(self.load_secrets().get("web_api_key", "") or "")

    def set_web_api_key(self, value: str) -> None:
        self.save_secrets({"web_api_key": value})

    def ensure_web_api_key(self) -> str:
        existing = self.get_web_api_key()
        if existing:
            return existing
        generated = secrets.token_urlsafe(24)
        self.set_web_api_key(generated)
        return generated

    def setup_status(self) -> dict[str, Any]:
        settings = self.load_settings()
        secrets_payload = self.load_secrets()
        provider = settings.get("provider", "gemini")
        provider_secret = bool(secrets_payload.get("providers", {}).get(provider, {}).get("api_key"))
        if provider == "ollama":
            provider_secret = True
        web_api_key = bool(secrets_payload.get("web_api_key"))
        return {
            "setup_completed": bool(settings.get("setup_completed")),
            "provider": provider,
            "model": settings.get("model"),
            "provider_configured": provider_secret,
            "web_api_key_configured": web_api_key,
            "paired_telegram_users": len(settings.get("telegram", {}).get("paired_users", [])),
            "telegram_public_mode": bool(settings.get("telegram", {}).get("public_mode", False)),
        }

    def create_pair_code(self, ttl_minutes: int = 10) -> str:
        settings = self.load_settings()
        code = secrets.token_hex(3).upper()
        pair_codes = list(settings.get("telegram", {}).get("pair_codes", []))
        pair_codes = [
            entry for entry in pair_codes
            if entry.get("expires_at", "") > _now() and not entry.get("consumed_at")
        ]
        pair_codes.append({
            "code": code,
            "created_at": _now(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(),
            "consumed_at": None,
            "paired_user_id": None,
        })
        settings.setdefault("telegram", {})["pair_codes"] = pair_codes
        self.save_settings(settings)
        return code

    def consume_pair_code(self, code: str, user_id: int, username: str | None = None) -> bool:
        code = code.strip().upper()
        settings = self.load_settings()
        telegram = settings.setdefault("telegram", {})
        pair_codes = telegram.get("pair_codes", [])
        now = _now()
        matched = False
        for entry in pair_codes:
            if entry.get("code") == code and not entry.get("consumed_at") and entry.get("expires_at", "") > now:
                entry["consumed_at"] = now
                entry["paired_user_id"] = user_id
                matched = True
                break
        if not matched:
            return False

        paired_users = telegram.setdefault("paired_users", [])
        if not any(int(user.get("user_id", 0)) == int(user_id) for user in paired_users):
            paired_users.append({
                "user_id": int(user_id),
                "username": username or "",
                "paired_at": now,
            })
        telegram["pair_codes"] = pair_codes
        self.save_settings(settings)
        return True

    def add_paired_telegram_user(self, user_id: int, username: str | None = None) -> None:
        """Directly add a user to the paired users list (used by magic handshake)."""
        settings = self.load_settings()
        telegram = settings.setdefault("telegram", {})
        paired_users = telegram.setdefault("paired_users", [])
        now = _now()
        
        if not any(int(user.get("user_id", 0)) == int(user_id) for user in paired_users):
            paired_users.append({
                "user_id": int(user_id),
                "username": username or "",
                "paired_at": now,
            })
            self.save_settings(settings)

    def revoke_telegram_user(self, user_id: int) -> None:
        settings = self.load_settings()
        telegram = settings.setdefault("telegram", {})
        telegram["paired_users"] = [
            user for user in telegram.get("paired_users", [])
            if int(user.get("user_id", 0)) != int(user_id)
        ]
        self.save_settings(settings)

    def is_telegram_user_allowed(self, user_id: int) -> bool:
        settings = self.load_settings()
        telegram = settings.get("telegram", {})
        if telegram.get("public_mode"):
            return True
        return any(int(user.get("user_id", 0)) == int(user_id) for user in telegram.get("paired_users", []))

    def list_telegram_users(self) -> list[dict[str, Any]]:
        return list(self.load_settings().get("telegram", {}).get("paired_users", []))
