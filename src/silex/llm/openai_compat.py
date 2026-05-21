from __future__ import annotations

import json
import time
from base64 import b64encode

from silex.llm.base import BaseLLMProvider, SchemaT, retry_on_transient, repair_json
from silex.runtime.usage import UsageTracker
from silex.utils.logger import setup_logger

log = setup_logger("silex.llm.openai_compat")


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible provider for OpenAI, OpenRouter, DeepSeek, Mistral, Groq, and Ollama."""

    def __init__(
        self,
        *,
        provider_name: str,
        default_model: str,
        api_key: str = "",
        base_url: str,
        usage_tracker: UsageTracker | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(default_model=default_model)
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self._usage_tracker = usage_tracker
        self._client = None

    def connect(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install aria[providers] to use OpenAI-compatible providers.") from exc

        self._client = AsyncOpenAI(
            api_key=self.api_key or "local-aria",
            base_url=self.base_url,
            default_headers=self.extra_headers or None,
        )
        log.info("OpenAI-compatible provider ready: %s (%s)", self.provider_name, self.default_model)

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError(f"{self.provider_name} client not connected.")
        return self._client

    @retry_on_transient(max_retries=3, base_delay=1.5)
    async def complete_json(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_input: str,
        images: list[dict] | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        request_kind: str = "chat",
    ) -> SchemaT:
        model = model_override or self.default_model
        user_content: list[dict[str, object]] | str = user_input
        if images:
            user_content = [{"type": "text", "text": user_input}]
            for image in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image['mime']};base64,{b64encode(image['bytes']).decode('ascii')}",
                        },
                    }
                )

        started = time.perf_counter()
        error_text: str | None = None
        response = None
        try:
            response = await self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = response.choices[0].message.content or "{}"

            # Try direct parse first, then repair if needed
            try:
                return schema.model_validate(json.loads(content))
            except (json.JSONDecodeError, Exception):
                log.warning(
                    "%s returned non-parseable JSON. Attempting repair...",
                    self.provider_name,
                )
                repaired = repair_json(content)
                return schema.model_validate(json.loads(repaired))
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            if self._usage_tracker:
                usage = getattr(response, "usage", None) if response else None
                await self._usage_tracker.log_llm_call(
                    provider=self.provider_name,
                    model=model,
                    request_kind=request_kind,
                    input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                    estimated_cost_usd=None,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    success=error_text is None,
                    error=error_text,
                )
