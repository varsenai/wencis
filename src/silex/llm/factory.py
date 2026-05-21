from __future__ import annotations

from silex.llm.catalog import MODEL_CATALOG
from silex.runtime.settings import RuntimeSettingsStore
from silex.runtime.usage import UsageTracker
from silex.utils.config import get_provider_settings


def build_provider(
    settings_store: RuntimeSettingsStore | None = None,
    usage_tracker: UsageTracker | None = None,
):
    active = get_provider_settings(settings_store)
    provider = active["provider"]
    model = active["model"]

    # Trigger dynamic load if not loaded
    _ = MODEL_CATALOG.get(provider)
    
    from registry import get_registered_profile
    profile = get_registered_profile(provider)
    
    if not profile:
        raise ValueError(f"Provider '{provider}' is not registered.")
        
    return profile.build_client(model, settings_store, usage_tracker)
