"""
Gemini provider implementation.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from google import genai
from google.genai import types

from silex.llm.base import BaseLLMProvider, SchemaT, retry_on_transient
from silex.runtime.settings import RuntimeSettingsStore
from silex.runtime.usage import UsageTracker
from silex.utils.config import get_provider_secret, get_provider_settings
from silex.utils.logger import setup_logger

log = setup_logger("silex.llm")


# ---------------------------------------------------------------------------
# Streaming JSON Parser
# ---------------------------------------------------------------------------

class StreamingJSONParser:
    """
    A character-by-character state-machine JSON parser that progressively
    extracts the value of the "response" key as characters are streamed from the LLM,
    correctly decoding escaped sequences on the fly.
    """
    def __init__(self, text_callback: Callable[[str], None]):
        self.text_callback = text_callback
        self.depth = 0
        self.in_string = False
        self.escaped = False
        self.string_buf = []
        
        self.found_response_key = False
        self.in_response_value = False
        self.after_key_colon = False
        
        self.in_unicode_escape = False
        self.unicode_escape_buf = []
        
    def feed(self, char: str):
        if self.in_unicode_escape:
            self.unicode_escape_buf.append(char)
            if len(self.unicode_escape_buf) == 4:
                self.in_unicode_escape = False
                try:
                    decoded = chr(int("".join(self.unicode_escape_buf), 16))
                    if self.in_response_value:
                        self.text_callback(decoded)
                    else:
                        self.string_buf.append(decoded)
                except Exception:
                    raw_seq = "u" + "".join(self.unicode_escape_buf)
                    if self.in_response_value:
                        self.text_callback("\\" + raw_seq)
                    else:
                        self.string_buf.append("\\" + raw_seq)
            return

        if self.in_string:
            if self.escaped:
                self.escaped = False
                if self.in_response_value:
                    if char == 'u':
                        self.unicode_escape_buf = []
                        self.in_unicode_escape = True
                        return
                    
                    if char == 'n':
                        self.text_callback('\n')
                    elif char == 't':
                        self.text_callback('\t')
                    elif char == 'r':
                        self.text_callback('\r')
                    elif char == 'b':
                        self.text_callback('\b')
                    elif char == 'f':
                        self.text_callback('\f')
                    elif char in ('"', '\\', '/'):
                        self.text_callback(char)
                    else:
                        self.text_callback('\\' + char)
                else:
                    if char == 'u':
                        self.unicode_escape_buf = []
                        self.in_unicode_escape = True
                        return
                    self.string_buf.append(char)
                return
            
            if char == '\\':
                self.escaped = True
                return
            
            if char == '"':
                self.in_string = False
                str_val = "".join(self.string_buf)
                self.string_buf.clear()
                
                if self.in_response_value:
                    self.in_response_value = False
                    self.found_response_key = False
                    
                elif self.depth == 1 and str_val == "response":
                    self.found_response_key = True
                    self.after_key_colon = False
                return
            
            if self.in_response_value:
                self.text_callback(char)
            else:
                self.string_buf.append(char)
                
        else:
            if char == '"':
                self.in_string = True
                self.escaped = False
                self.string_buf.clear()
                if self.found_response_key and self.after_key_colon:
                    self.in_response_value = True
                return
                
            if char in ('{', '['):
                self.depth += 1
            elif char in ('}', ']'):
                self.depth -= 1
                
            if self.found_response_key:
                if char == ':':
                    self.after_key_colon = True


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMProvider):
    """Manages the connection to Google's Gemini API."""

    def __init__(
        self,
        settings_store: RuntimeSettingsStore | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        settings = get_provider_settings(settings_store)
        super().__init__(default_model=settings["model"])
        self._settings_store = settings_store
        self._usage_tracker = usage_tracker
        self._client: genai.Client | None = None
        self.provider_name = "gemini"

    def connect(self) -> None:
        """Initialize the Gemini client."""
        api_key = get_provider_secret("gemini", settings_store=self._settings_store)
        self._client = genai.Client(api_key=api_key)
        log.info(f"Gemini client initialized with model: {self.default_model}")

    @property
    def client(self) -> genai.Client:
        """Get the active client or fail."""
        if self._client is None:
            raise RuntimeError("Gemini client not connected. Call connect() first.")
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
        contents = []
        if images:
            for img_dict in images:
                contents.append(
                    types.Part.from_bytes(data=img_dict["bytes"], mime_type=img_dict["mime"])
                )
        contents.append(user_input)

        started = time.perf_counter()
        error_text: str | None = None
        response: Any | None = None
        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )

            raw_text = response.text
            if not raw_text:
                raise ValueError("Empty response from Gemini")

            try:
                parsed = schema.model_validate_json(raw_text)
            except Exception:
                log.warning("Gemini returned invalid JSON. Attempting repair...")
                retry_contents = contents.copy()
                retry_contents.pop()
                retry_contents.append(
                    f"{user_input}\n\n"
                    "[SYSTEM: Your previous response was not valid JSON. "
                    "Please respond ONLY with valid JSON matching the schema.]"
                )
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=retry_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=max(0.1, temperature - 0.2),
                    ),
                )
                parsed = schema.model_validate_json(response.text or "{}")
            return parsed
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            if self._usage_tracker:
                usage = getattr(response, "usage_metadata", None) if response else None
                await self._usage_tracker.log_llm_call(
                    provider=self.provider_name,
                    model=model,
                    request_kind=request_kind,
                    input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                    output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
                    estimated_cost_usd=None,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    success=error_text is None,
                    error=error_text,
                )

    @retry_on_transient(max_retries=3, base_delay=1.5)
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
        model = model_override or self.default_model
        contents = []
        if images:
            for img_dict in images:
                contents.append(
                    types.Part.from_bytes(data=img_dict["bytes"], mime_type=img_dict["mime"])
                )
        contents.append(user_input)

        started = time.perf_counter()
        error_text: str | None = None
        response: Any | None = None
        
        parser = StreamingJSONParser(text_callback)
        
        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )

            raw_chunks = []
            async for chunk in response_stream:
                chunk_text = chunk.text
                if chunk_text:
                    raw_chunks.append(chunk_text)
                    for char in chunk_text:
                        parser.feed(char)
                if getattr(chunk, "usage_metadata", None):
                    response = chunk

            raw_text = "".join(raw_chunks)
            if not raw_text:
                raise ValueError("Empty response from Gemini stream")

            try:
                parsed = schema.model_validate_json(raw_text)
            except Exception:
                log.warning("Gemini stream returned invalid JSON. Attempting repair...")
                retry_contents = contents.copy()
                retry_contents.pop()
                retry_contents.append(
                    f"{user_input}\n\n"
                    "[SYSTEM: Your previous response was not valid JSON. "
                    "Please respond ONLY with valid JSON matching the schema.]"
                )
                
                repair_response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=retry_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=max(0.1, temperature - 0.2),
                    ),
                )
                response = repair_response
                parsed = schema.model_validate_json(repair_response.text or "{}")
                if hasattr(parsed, "response"):
                    text_callback(parsed.response)
                elif isinstance(parsed, dict) and "response" in parsed:
                    text_callback(parsed["response"])
            return parsed
        except Exception as exc:
            error_text = str(exc)
            raise
        finally:
            if self._usage_tracker:
                usage = getattr(response, "usage_metadata", None) if response else None
                await self._usage_tracker.log_llm_call(
                    provider=self.provider_name,
                    model=model,
                    request_kind=request_kind,
                    input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                    output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
                    estimated_cost_usd=None,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    success=error_text is None,
                    error=error_text,
                )
