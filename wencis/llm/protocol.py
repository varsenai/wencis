# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
LLMClient Protocol — what Wencis requires from any LLM adapter.

Users implement this protocol for their own provider (OpenAI, Anthropic,
Gemini, Ollama, etc.) or use the included MockLLMClient for testing.
"""
from __future__ import annotations
from typing import Protocol, TypeVar, Type, runtime_checkable
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """
    Minimal LLM interface required by Wencis components.

    The single required method is complete_json — it sends a system+user
    prompt to an LLM and parses the response into a validated Pydantic model.

    Implementors may use any provider. The only contract is:
    - The LLM is instructed to return valid JSON matching `schema`
    - The response is parsed and validated via Pydantic
    - On parse failure, raise an exception (don't return garbage)
    """

    async def complete_json(
        self,
        *,
        schema: Type[SchemaT],
        system_prompt: str,
        user_input: str,
        temperature: float = 0.2,
    ) -> SchemaT:
        """
        Send a structured prompt and return a validated Pydantic model.

        Parameters
        ----------
        schema:
            The Pydantic model class to parse the response into.
            The LLM must be prompted to return JSON matching this schema.
        system_prompt:
            The system/instruction prompt sent to the LLM.
        user_input:
            The user-turn content (data to analyze).
        temperature:
            Sampling temperature. Wencis always uses low values (0.1-0.2)
            for deterministic analysis tasks.

        Returns
        -------
        An instance of `schema` populated from the LLM's JSON response.

        Raises
        ------
        Exception if the LLM is unavailable or returns unparseable output.
        """
        ...
