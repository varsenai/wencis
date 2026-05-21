"""
Configuration loader for VYN.

Reads from .env file and provides typed access to all settings.
All runtime data lives under ~/.vyn/
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silex.runtime.settings import RuntimeSettingsStore

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Silex Home paths — dynamically injected by the application layer
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[5]

_env_data_dir = os.getenv("SILEX_DATA_DIR") or os.getenv("VYN_DATA_DIR")
if _env_data_dir:
    SILEX_HOME = Path(_env_data_dir).resolve()
else:
    SILEX_HOME = Path.home() / ".silex"

VYN_HOME = SILEX_HOME # Legacy alias

VYN_DB           = SILEX_HOME / "silex.db"
VYN_CONFIG       = SILEX_HOME / "config.json"
VYN_SECRETS      = SILEX_HOME / "secrets.json"
VYN_WORKSPACE    = SILEX_HOME / "workspace"
VYN_VECTOR_DB    = SILEX_HOME / "memory" / "vector_db"
VYN_SKILLS       = SILEX_HOME / "skills"
VYN_LOGS         = SILEX_HOME / "logs"
VYN_DAEMON_LOG   = SILEX_HOME / "logs" / "daemon.log"
VYN_PHANTOM      = SILEX_HOME / ".phantom"
VYN_DAEMON_LOCK  = SILEX_HOME / "daemon.lock"
VYN_MANIFEST     = SILEX_HOME / "workspace_index_manifest.json"
VYN_PROCESS_LOCK = SILEX_HOME / ".silex-process.lock"
VYN_ONTOLOGY     = SILEX_HOME / "ontology.json"
VYN_EXPORTS      = SILEX_HOME / "exports"
VYN_TRACES       = SILEX_HOME / "traces"
VYN_PENDING_EDITS = SILEX_HOME / "pending_edits.json"
VYN_BACKUPS      = SILEX_HOME / "backups"
VYN_MEMORY_MD    = SILEX_HOME / "MEMORY.md"
VYN_PLUGINS_PROVIDERS = SILEX_HOME / "plugins" / "model-providers"
# Phase 21 — Skill Lifecycle
VYN_SKILLS_ARCHIVE  = VYN_SKILLS / ".archive"
VYN_CURATOR_LAST_RUN = SILEX_HOME / ".curator_last_run"


# Legacy aliases kept so existing imports don't break
DATA_DIR   = VYN_HOME
DB_PATH    = VYN_DB


# WORKSPACE: VYN_WORKSPACE env > ARIA_WORKSPACE env (backwards compat) > ~/.vyn/workspace
_workspace_env = os.getenv("VYN_WORKSPACE") or os.getenv("ARIA_WORKSPACE")
if _workspace_env:
    WORKSPACE_DIR = Path(_workspace_env).resolve()
    if not str(WORKSPACE_DIR).startswith(str(VYN_WORKSPACE)):
        logging.getLogger("vyn.init").warning(f"SECURITY WARNING: Workspace directory resolved to {WORKSPACE_DIR} outside {VYN_WORKSPACE}")
else:
    WORKSPACE_DIR = VYN_WORKSPACE

VYN_DIRECTIVES_FILE = WORKSPACE_DIR / "vyn_core_directives.md"

_vyn_home_ensured = False

def ensure_vyn_home() -> None:
    global _vyn_home_ensured
    if _vyn_home_ensured:
        return
    _vyn_home_ensured = True

    # 1. Create directories
    VYN_HOME.mkdir(exist_ok=True)
    VYN_WORKSPACE.mkdir(exist_ok=True)
    VYN_VECTOR_DB.mkdir(parents=True, exist_ok=True)
    VYN_SKILLS.mkdir(exist_ok=True)
    VYN_LOGS.mkdir(exist_ok=True)
    VYN_TRACES.mkdir(exist_ok=True)
    VYN_BACKUPS.mkdir(exist_ok=True)
    VYN_PLUGINS_PROVIDERS.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("vyn.init")

    # 2. Skills README
    readme_path = VYN_SKILLS / "README.md"
    if not any(VYN_SKILLS.iterdir()) or not readme_path.exists():
        readme_path.write_text(
            "Add .md files to this directory to extend VYN with new skills.\n"
            "Each file should describe a workflow or capability.\n"
            "Restart VYN after adding a skill for it to take effect.\n",
            encoding="utf-8"
        )

    if not VYN_DIRECTIVES_FILE.exists():
        VYN_DIRECTIVES_FILE.write_text(
            "# VYN Core Directives\n\n"
            "This file contains unbreakable rules and behavioral guidelines. "
            "Any instructions here override general knowledge and normal operating procedures.\n",
            encoding="utf-8"
        )

    # 3. Phantom Cleanup
    if VYN_PHANTOM.exists():
        try:
            shutil.rmtree(VYN_PHANTOM)
            log.info("Cleaned up leftover phantom directory from previous crash")
        except Exception as e:
            log.error(f"Failed to clean up phantom directory: {e}")

    # 4. Migrate old data
    old_data_dir = PROJECT_ROOT / "data"
    old_db = old_data_dir / "silex.db"
    
    if old_db.exists() and not VYN_DB.exists():
        shutil.copy2(old_db, VYN_DB)
        log.info("Migrated existing database to ~/.vyn/vyn.db")
    elif old_db.exists() and VYN_DB.exists():
        log.warning("WARNING: Both old database (data/silex.db) and new database (~/.vyn/vyn.db) exist. Using new database.")

    old_vector_db1 = old_data_dir / "vector_db"
    old_vector_db2 = VYN_HOME / "vector_db"
    for old_v in [old_vector_db1, old_vector_db2]:
        if old_v.exists() and old_v.is_dir():
            if not any(VYN_VECTOR_DB.iterdir()):
                shutil.copytree(old_v, VYN_VECTOR_DB, dirs_exist_ok=True)
                log.info(f"Migrated existing ChromaDB from {old_v} to {VYN_VECTOR_DB}")
            break

    # Migrate settings/secrets
    old_settings = [old_data_dir / "settings.json", VYN_HOME / "settings.json"]
    for osg in old_settings:
        if osg.exists() and not VYN_CONFIG.exists():
            shutil.copy2(osg, VYN_CONFIG)
            log.info(f"Migrated settings from {osg} to {VYN_CONFIG}")
            break
            
    old_secrets = [old_data_dir / "secrets.json", VYN_HOME / "secrets.json"]
    for os_sec in old_secrets:
        if os_sec.exists() and not VYN_SECRETS.exists():
            shutil.copy2(os_sec, VYN_SECRETS)
            log.info(f"Migrated secrets from {os_sec} to {VYN_SECRETS}")
            break

    # Secrets permission
    if not VYN_SECRETS.exists():
        VYN_SECRETS.write_text("{}", encoding="utf-8")
        
    if os.name != "nt":
        try:
            os.chmod(VYN_SECRETS, 0o600)
        except OSError:
            pass
    else:
        log.warning("WARNING: secrets.json has no file permission protection on Windows. Store API keys as environment variables for better security.")

ensure_vyn_home()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load .env from project root
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

_settings_store = None

def get_settings_store() -> "RuntimeSettingsStore":
    global _settings_store
    if _settings_store is None:
        from silex.runtime.settings import RuntimeSettingsStore
        _settings_store = RuntimeSettingsStore()
    return _settings_store


def get_provider_settings(settings_store = None) -> dict:
    store = settings_store or get_settings_store()
    saved = store.load_settings()
    
    # Raw saved settings without default merging
    raw_saved = {}
    try:
        import json
        path = store.settings_path if hasattr(store, "settings_path") else VYN_CONFIG
        if path.exists():
            raw_saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    
    # Priority: Env Var > Saved Settings > Hardcoded Default
    # BUT: We only want Env Var to win if it's NOT the "dummy" default from a pre-filled .env
    provider = os.getenv("ARIA_PROVIDER") or saved.get("provider", "gemini")
    model = os.getenv("ARIA_MODEL") or saved.get("model", "gemini-3.1-flash-lite")
    
    # If the user is using custom, we MUST respect the saved model/base_url
    if provider == "custom":
        model = saved.get("model", model)
        
    # Determine family-aligned defaults for gemini provider
    default_fast = "gemini-3.1-flash-lite"
    default_reasoning = "gemini-3.1-pro"
    default_critic = "gemini-3.1-pro"
    
    if provider == "gemini":
        if model.startswith("gemini-2.5"):
            default_fast = "gemini-2.5-flash"
            default_reasoning = "gemini-2.5-pro"
            default_critic = "gemini-2.5-pro"
        elif model.startswith("gemini-1.5"):
            default_fast = "gemini-1.5-flash"
            default_reasoning = "gemini-1.5-pro"
            default_critic = "gemini-1.5-pro"
        
    fast_model = os.getenv("ARIA_FAST_MODEL") or raw_saved.get("fast_model") or default_fast
    reasoning_model = os.getenv("ARIA_REASONING_MODEL") or raw_saved.get("reasoning_model") or default_reasoning
    critic_model = os.getenv("ARIA_CRITIC_MODEL") or raw_saved.get("critic_model") or default_critic
    
    return {
        "provider": provider,
        "model": model,
        "fast_model": fast_model,
        "reasoning_model": reasoning_model,
        "critic_model": critic_model,
        "base_url": saved.get("base_url", ""),
    }


def get_provider_secret(provider_id: str, settings_store = None) -> str | None:
    store = settings_store or get_settings_store()
    stored = store.get_provider_secret(provider_id)
    if stored:
        return stored

    from silex.llm.catalog import MODEL_CATALOG
    payload = MODEL_CATALOG.get(provider_id, {})
    env_name = payload.get("env_key", "")
    if not env_name:
        return ""
    value = os.getenv(env_name, "")
    if not value or value.endswith("_here"):
        return ""
    return value


def get_api_key() -> str:
    """Backward-compatible provider key lookup."""
    provider = get_provider_settings()["provider"]
    key = get_provider_secret(provider)
    if key:
        return key
    raise EnvironmentError(
        f"{provider} API key is not set.\n"
        "Run `vyn setup`, use the web onboarding flow, or configure the matching env var."
    )


def get_model() -> str:
    """Get the active model."""
    return get_provider_settings()["model"]


def get_log_level() -> str:
    """Get the logging level."""
    return os.getenv("ARIA_LOG_LEVEL", "INFO").upper()


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean feature flag from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _saved_security_flag(name: str, default: bool) -> bool:
    settings = get_settings_store().load_settings()
    return bool(settings.get("security", {}).get(name, default))


def terminal_execution_enabled() -> bool:
    """Whether ARIA may run sandboxed terminal commands."""
    return env_flag("ARIA_ENABLE_TERMINAL_EXECUTION", _saved_security_flag("terminal_execution", False))


def code_apply_enabled() -> bool:
    """Whether ARIA may apply code edits without a human approval step."""
    return env_flag("ARIA_ENABLE_CODE_APPLY", _saved_security_flag("code_apply", False))


def browser_actions_enabled() -> bool:
    """Whether ARIA may use the browser automation tool."""
    return env_flag("ARIA_ENABLE_BROWSER_ACTIONS", _saved_security_flag("browser_actions", True))


def background_actions_enabled() -> bool:
    """Whether ARIA may wake itself up to work on active goals."""
    return env_flag("ARIA_ENABLE_BACKGROUND_LOOP", _saved_security_flag("background_actions", False))


def require_tool_approvals() -> bool:
    """Whether high-risk tools should enter a pending approval queue."""
    return env_flag("ARIA_REQUIRE_TOOL_APPROVALS", _saved_security_flag("require_tool_approvals", True))


def max_tool_calls_per_turn() -> int:
    """Hard ceiling for model-requested tool calls in a single turn."""
    try:
        return max(1, int(os.getenv("ARIA_MAX_TOOL_CALLS_PER_TURN", "8")))
    except ValueError:
        return 8


def get_process_role() -> str:
    """Identify this process for single-writer deployment checks."""
    return os.getenv("ARIA_PROCESS_ROLE", "standalone")


def allow_multi_writer() -> bool:
    """Whether multiple ARIA processes may share a data directory."""
    return env_flag("ARIA_ALLOW_MULTI_WRITER", False)


def get_web_host() -> str:
    return os.getenv("ARIA_WEB_HOST", "127.0.0.1")


def get_web_port() -> int:
    try:
        return int(os.getenv("ARIA_WEB_PORT", "8000"))
    except ValueError:
        return 8000


def get_web_api_key() -> str:
    env_value = os.getenv("ARIA_WEB_API_KEY", "")
    if env_value:
        return env_value
    return _settings_store.get_web_api_key()


def get_web_allowed_origins() -> list[str]:
    raw = os.getenv(
        "ARIA_WEB_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def telegram_public_mode_enabled() -> bool:
    settings_value = bool(_settings_store.load_settings().get("telegram", {}).get("public_mode", False))
    return env_flag("TELEGRAM_PUBLIC_MODE", settings_value)


def autonomy_policy_snapshot() -> dict:
    """Operator-facing summary of the active autonomy policy."""
    return {
        "terminal_execution": terminal_execution_enabled(),
        "code_apply": code_apply_enabled(),
        "browser_actions": browser_actions_enabled(),
        "background_actions": background_actions_enabled(),
        "require_tool_approvals": require_tool_approvals(),
        "max_tool_calls_per_turn": max_tool_calls_per_turn(),
        "process_role": get_process_role(),
        "provider": get_provider_settings()["provider"],
        "model": get_provider_settings()["model"],
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Memory retrieval budget per turn
MAX_RECENT_MEMORIES = 5
MAX_IMPORTANT_MEMORIES = 5
MAX_RELEVANT_MEMORIES = 5

# Conversation context
MAX_HISTORY_TURNS = 10

# Memory pruning
MEMORY_ARCHIVE_THRESHOLD = 0.1  # Importance below this gets archived eventually
MEMORY_MAX_AGE_DAYS = 365       # For future use
MEMORY_HALFLIFE_DAYS = 30.0

