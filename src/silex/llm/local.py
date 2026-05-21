"""
Local LLM provider scaffold.

This uses an OpenAI-compatible local endpoint (for example Ollama or llama.cpp
server) for low-risk routing once a local model passes ARIA's eval suite.
"""

from __future__ import annotations

import json
import os

from pydantic import ValidationError

from silex.models.schemas import CognitiveResponse
from silex.utils.logger import setup_logger

log = setup_logger("silex.llm.local")


class LocalLLMClient:
    """OpenAI-compatible local model client."""

    def __init__(self):
        self.base_url = os.getenv("ARIA_LOCAL_LLM_URL", "http://127.0.0.1:11434/v1")
        self.model = os.getenv("ARIA_LOCAL_MODEL", "local-aria")
        self.enabled = os.getenv("ARIA_ENABLE_LOCAL_LLM", "false").lower() == "true"

    def connect(self) -> None:
        if self.enabled:
            log.info(f"Local LLM routing enabled: {self.base_url} ({self.model})")

    async def think(
        self,
        system_prompt: str,
        user_input: str,
        images: list[dict] | None = None,
        model_override: str | None = None,
    ) -> CognitiveResponse:
        if not self.enabled:
            raise RuntimeError("Local LLM is disabled. Set ARIA_ENABLE_LOCAL_LLM=true.")
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("Install aria[local] to use LocalLLMClient.") from e

        payload = {
            "model": model_override or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        try:
            return CognitiveResponse.model_validate_json(content)
        except ValidationError:
            return CognitiveResponse.model_validate(json.loads(content))
