# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
MockLLMClient — a zero-dependency LLM client for tests and examples.

Returns configurable canned responses for each schema type. Use in
pytest fixtures and documentation examples.
"""
from __future__ import annotations
from typing import Any, Type, TypeVar
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class MockLLMClient:
    """
    Returns a minimal valid instance of whatever schema is requested.
    Pass `responses` dict to override specific schema types.

    Example:
        mock = MockLLMClient(responses={
            TrajectoryRevisionResult: TrajectoryRevisionResult(
                critique="Test failure in step 3",
                step_revisions=[]
            )
        })
    """

    def __init__(self, responses: dict[Type[BaseModel], BaseModel] | None = None):
        self._responses = responses or {}

    async def complete_json(
        self,
        *,
        schema: Type[SchemaT],
        system_prompt: str,
        user_input: str,
        temperature: float = 0.2,
    ) -> SchemaT:
        if schema in self._responses:
            return self._responses[schema]
        # Build a minimal valid instance by providing defaults for all fields
        # Pydantic will raise if required fields have no default — callers
        # must provide responses for schemas with required fields.
        try:
            return schema()
        except Exception:
            raise RuntimeError(
                f"MockLLMClient: no canned response for {schema.__name__}. "
                f"Pass responses={{'{schema.__name__}': <instance>}} to MockLLMClient."
            )
