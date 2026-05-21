from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def discover_providers():
    """Discover and load provider plugins."""
    # Only load once
    if getattr(discover_providers, "_loaded", False):
        return
    discover_providers._loaded = True

    project_root = Path(__file__).resolve().parents[5]
    bundled_dir = project_root / "products" / "vyn" / "src" / "vyn_app" / "plugins" / "model-providers"
    user_dir = Path.home() / ".vyn" / "plugins" / "model-providers"
    
    # Add plugins directory to sys.path so plugins can do `from registry import ...`
    plugins_dir_str = str(bundled_dir)
    if plugins_dir_str not in sys.path:
        sys.path.insert(0, plugins_dir_str)
        
    discovered = {}
    
    if bundled_dir.exists():
        for item in bundled_dir.iterdir():
            if item.is_dir() and (item / "plugin.yaml").exists() and (item / "__init__.py").exists():
                discovered[item.name] = item
                
    if user_dir.exists():
        for item in user_dir.iterdir():
            if item.is_dir() and (item / "plugin.yaml").exists() and (item / "__init__.py").exists():
                discovered[item.name] = item

    log = logging.getLogger("silex.llm.catalog")
    for name, path in discovered.items():
        try:
            with open(path / "plugin.yaml", "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            if manifest.get("kind") != "model-provider":
                continue
                
            module_name = f"vyn_provider_{name}"
            init_file = path / "__init__.py"
            spec = importlib.util.spec_from_file_location(module_name, str(init_file))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as e:
            log.error(f"Failed to load provider plugin {name} from {path}: {e}")

class DynamicModelCatalog(dict):
    """A dictionary subclass that lazy-loads model providers on access."""
    
    def __init__(self):
        super().__init__()
        self._loaded = False
        
    def _ensure_loaded(self):
        if not self._loaded:
            self._loaded = True
            discover_providers()
            # registry is available now because discover_providers adds it to path
            from registry import get_registered_providers
            
            for provider_id, profile in get_registered_providers().items():
                self[provider_id] = {
                    "label": profile.label,
                    "env_key": profile.env_key,
                    "base_url": profile.base_url,
                    "models": profile.models,
                }

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def items(self):
        self._ensure_loaded()
        return super().items()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()


MODEL_CATALOG = DynamicModelCatalog()


def list_providers() -> list[dict[str, Any]]:
    providers = []
    for provider_id, payload in MODEL_CATALOG.items():
        providers.append(
            {
                "id": provider_id,
                "label": payload["label"],
                "env_key": payload.get("env_key", ""),
                "base_url": payload.get("base_url", ""),
                "models": payload["models"],
            }
        )
    return providers


def get_provider_defaults(provider: str) -> dict[str, Any]:
    payload = MODEL_CATALOG.get(provider)
    if not payload:
        raise ValueError(f"Unknown provider: {provider}")
    models = payload.get("models")
    if not models:
        raise ValueError(f"No models defined for provider: {provider}")
    fast_model = next((m for m in models if m.get("tier") == "fast"), models[0])
    reasoning_model = next((m for m in models if m.get("tier") == "reasoning"), models[0])
    fast_id = fast_model["id"]
    reasoning_id = reasoning_model["id"]
    return {
        "provider": provider,
        "model": fast_id,
        "fast_model": fast_id,
        "reasoning_model": reasoning_id,
        "label": payload["label"],
        "env_key": payload.get("env_key", ""),
        "base_url": payload.get("base_url", ""),
    }


def find_model(provider: str, model_id: str) -> dict[str, Any] | None:
    payload = MODEL_CATALOG.get(provider, {})
    return next((model for model in payload.get("models", []) if model["id"] == model_id), None)
