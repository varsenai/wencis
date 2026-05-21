"""
LLM provider interfaces and shared helpers.

Shared utilities (v1.0.5):
  - retry_on_transient: exponential-backoff decorator for transient API errors.
  - repair_json: strips markdown fences and common LLM JSON formatting artifacts.
  Both are used by ALL providers, not just Gemini.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Protocol, TypeVar, Callable

from pydantic import BaseModel

from silex.models.schemas import CognitiveResponse
from silex.utils.logger import setup_logger

log = setup_logger("silex.llm.base")

SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Shared: Retry decorator for transient API errors
# ---------------------------------------------------------------------------

_TRANSIENT_ERROR_CODES = {"503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"}


def _is_transient(error: Exception) -> bool:
    """Check if an exception is a transient API error worth retrying."""
    error_str = str(error)
    return any(code in error_str for code in _TRANSIENT_ERROR_CODES)


def retry_on_transient(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator that retries async functions on transient API errors.
    Uses exponential backoff with jitter.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if _is_transient(e) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log.warning(
                            f"Transient API error (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
            raise last_error  # Should never reach here, but safety net
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Shared: JSON repair for non-compliant LLM output
# ---------------------------------------------------------------------------

_MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def repair_json(raw: str) -> str:
    """Attempt to extract clean JSON from LLM output that may include
    markdown fences, leading/trailing prose, or other formatting artifacts.

    Returns the cleaned string (still needs json.loads/pydantic validation).
    """
    from silex.utils.json_repair import repair_json as custom_repair_json
    return custom_repair_json(raw)


# ---------------------------------------------------------------------------
# Provider base class
# ---------------------------------------------------------------------------

class SupportsLLM(Protocol):
    provider_name: str
    default_model: str

    def connect(self) -> None:
        ...

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
        ...

    async def complete_json_stream(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_input: str,
        text_callback: Callable[[str], None],
        images: list[dict] | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        request_kind: str = "chat",
    ) -> SchemaT:
        ...

    async def think(
        self,
        system_prompt: str,
        user_input: str,
        images: list[dict] | None = None,
        model_override: str | None = None,
    ) -> CognitiveResponse:
        ...

    async def think_stream(
        self,
        system_prompt: str,
        user_input: str,
        text_callback: Callable[[str], None],
        images: list[dict] | None = None,
        model_override: str | None = None,
    ) -> CognitiveResponse:
        ...


class BaseLLMProvider(ABC):
    provider_name = "unknown"

    def __init__(self, default_model: str):
        self.default_model = default_model
        
        # Dynamically wrap complete_json with caching
        original_complete_json = self.complete_json
        
        async def wrapped_complete_json(
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_input: str,
            images: list[dict] | None = None,
            model_override: str | None = None,
            temperature: float = 0.7,
            request_kind: str = "chat",
        ) -> SchemaT:
            return await self._cached_complete_json(
                original_complete_json,
                schema=schema,
                system_prompt=system_prompt,
                user_input=user_input,
                images=images,
                model_override=model_override,
                temperature=temperature,
                request_kind=request_kind,
            )
        
        self.complete_json = wrapped_complete_json

        # Dynamically wrap complete_json_stream with caching
        original_complete_json_stream = getattr(self, "complete_json_stream", None)
        if original_complete_json_stream:
            async def wrapped_complete_json_stream(
                *,
                schema: type[SchemaT],
                system_prompt: str,
                user_input: str,
                text_callback: Callable[[str], None],
                images: list[dict] | None = None,
                model_override: str | None = None,
                temperature: float = 0.7,
                request_kind: str = "chat",
            ) -> SchemaT:
                return await self._cached_complete_json_stream(
                    original_complete_json_stream,
                    schema=schema,
                    system_prompt=system_prompt,
                    user_input=user_input,
                    text_callback=text_callback,
                    images=images,
                    model_override=model_override,
                    temperature=temperature,
                    request_kind=request_kind,
                )
            self.complete_json_stream = wrapped_complete_json_stream

    async def _cached_complete_json(
        self,
        original_complete_json,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_input: str,
        images: list[dict] | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        request_kind: str = "chat",
    ) -> SchemaT:
        # Resolve DB from usage tracker
        db = None
        if hasattr(self, "_usage_tracker") and self._usage_tracker:
            db = self._usage_tracker.db
            
        if not db:
            return await original_complete_json(
                schema=schema,
                system_prompt=system_prompt,
                user_input=user_input,
                images=images,
                model_override=model_override,
                temperature=temperature,
                request_kind=request_kind,
            )

        import hashlib
        from datetime import datetime, timezone

        # 3. Hashing
        hash_input = f"{system_prompt}||{user_input}||{schema.__name__}"
        query_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        # 4. Retrieval & TTL (15 minutes = 900 seconds)
        try:
            cached_row = await db.fetch_one(
                "SELECT response, created_at FROM response_cache WHERE query_hash = ?",
                (query_hash,)
            )
            if cached_row:
                cached_response = cached_row["response"]
                created_at_str = cached_row["created_at"]
                
                created_at = datetime.fromisoformat(created_at_str)
                now = datetime.now(timezone.utc)
                age = (now - created_at).total_seconds()
                
                if age <= 900:
                    log.info("Semantic Response Cache HIT! (Age: %.1fs)", age)
                    return self.parse_model_json(schema, cached_response)
                else:
                    log.debug("Cache hit but expired (Age: %.1fs)", age)
        except Exception as e:
            log.warning("Failed to check response cache: %s", e)

        # 5. Storage (on cache miss)
        result = await original_complete_json(
            schema=schema,
            system_prompt=system_prompt,
            user_input=user_input,
            images=images,
            model_override=model_override,
            temperature=temperature,
            request_kind=request_kind,
        )

        try:
            if isinstance(result, BaseModel):
                json_str = result.model_dump_json()
            else:
                json_str = json.dumps(result)

            now_str = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT OR REPLACE INTO response_cache (query_hash, response, created_at) VALUES (?, ?, ?)",
                (query_hash, json_str, now_str)
            )
            log.debug("Stored response in Semantic Response Cache.")
        except Exception as e:
            log.warning("Failed to save response to cache: %s", e)

        return result

    async def _cached_complete_json_stream(
        self,
        original_complete_json_stream,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_input: str,
        text_callback: Callable[[str], None],
        images: list[dict] | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        request_kind: str = "chat",
    ) -> SchemaT:
        # Resolve DB from usage tracker
        db = None
        if hasattr(self, "_usage_tracker") and self._usage_tracker:
            db = self._usage_tracker.db
            
        if not db:
            return await original_complete_json_stream(
                schema=schema,
                system_prompt=system_prompt,
                user_input=user_input,
                text_callback=text_callback,
                images=images,
                model_override=model_override,
                temperature=temperature,
                request_kind=request_kind,
            )

        import hashlib
        from datetime import datetime, timezone

        # 3. Hashing
        hash_input = f"{system_prompt}||{user_input}||{schema.__name__}"
        query_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        # 4. Retrieval & TTL (15 minutes = 900 seconds)
        try:
            cached_row = await db.fetch_one(
                "SELECT response, created_at FROM response_cache WHERE query_hash = ?",
                (query_hash,)
            )
            if cached_row:
                cached_response = cached_row["response"]
                created_at_str = cached_row["created_at"]
                
                created_at = datetime.fromisoformat(created_at_str)
                now = datetime.now(timezone.utc)
                age = (now - created_at).total_seconds()
                
                if age <= 900:
                    log.info("Semantic Response Cache HIT! (Age: %.1fs)", age)
                    parsed_obj = self.parse_model_json(schema, cached_response)
                    # Trigger text_callback with the "response" field if it exists
                    if hasattr(parsed_obj, "response"):
                        text_callback(parsed_obj.response)
                    elif isinstance(parsed_obj, dict) and "response" in parsed_obj:
                        text_callback(parsed_obj["response"])
                    return parsed_obj
                else:
                    log.debug("Cache hit but expired (Age: %.1fs)", age)
        except Exception as e:
            log.warning("Failed to check response cache: %s", e)

        # 5. Storage (on cache miss)
        result = await original_complete_json_stream(
            schema=schema,
            system_prompt=system_prompt,
            user_input=user_input,
            text_callback=text_callback,
            images=images,
            model_override=model_override,
            temperature=temperature,
            request_kind=request_kind,
        )

        try:
            if isinstance(result, BaseModel):
                json_str = result.model_dump_json()
            else:
                json_str = json.dumps(result)

            now_str = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT OR REPLACE INTO response_cache (query_hash, response, created_at) VALUES (?, ?, ?)",
                (query_hash, json_str, now_str)
            )
            log.debug("Stored response in Semantic Response Cache.")
        except Exception as e:
            log.warning("Failed to save response to cache: %s", e)

        return result

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    async def complete_json_stream(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_input: str,
        text_callback: Callable[[str], None],
        images: list[dict] | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        request_kind: str = "chat",
    ) -> SchemaT:
        """
        Default streaming implementation that falls back to complete_json and calls text_callback with the final response.
        Override this in subclasses for true native streaming.
        """
        res = await self.complete_json(
            schema=schema,
            system_prompt=system_prompt,
            user_input=user_input,
            images=images,
            model_override=model_override,
            temperature=temperature,
            request_kind=request_kind,
        )
        if hasattr(res, "response"):
            text_callback(res.response)
        elif isinstance(res, dict) and "response" in res:
            text_callback(res["response"])
        return res

    async def complete_text(
        self,
        prompt: str,
        model_override: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Plain-text completion without JSON schema enforcement.

        Default implementation wraps complete_json with a minimal schema.
        Providers can override this for a more efficient raw call.
        """
        from pydantic import BaseModel as _BaseModel

        class _TextResult(_BaseModel):
            text: str

        result = await self.complete_json(
            schema=_TextResult,
            system_prompt="Respond with only the requested content, no commentary.",
            user_input=prompt,
            model_override=model_override,
            temperature=temperature,
            request_kind="compression",
        )
        return result.text

    async def think(
        self,
        system_prompt: str,
        user_input: str,
        images: list[dict] | None = None,
        model_override: str | None = None,
    ) -> CognitiveResponse:
        return await self.complete_json(
            schema=CognitiveResponse,
            system_prompt=system_prompt,
            user_input=user_input,
            images=images,
            model_override=model_override,
            temperature=0.7,
            request_kind="chat",
        )

    async def think_stream(
        self,
        system_prompt: str,
        user_input: str,
        text_callback: Callable[[str], None],
        images: list[dict] | None = None,
        model_override: str | None = None,
    ) -> CognitiveResponse:
        return await self.complete_json_stream(
            schema=CognitiveResponse,
            system_prompt=system_prompt,
            user_input=user_input,
            text_callback=text_callback,
            images=images,
            model_override=model_override,
            temperature=0.7,
            request_kind="chat",
        )

    @staticmethod
    def parse_model_json(schema: type[SchemaT], payload: str | dict[str, Any]) -> SchemaT:
        if isinstance(payload, str):
            try:
                return schema.model_validate_json(payload)
            except Exception:
                try:
                    repaired = repair_json(payload)
                    return schema.model_validate_json(repaired)
                except Exception:
                    return schema.model_validate(json.loads(payload))
        return schema.model_validate(payload)
