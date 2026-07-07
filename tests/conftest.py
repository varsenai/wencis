# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import pytest_asyncio
from wencis.storage.sqlite_backend import SQLiteBackend


@pytest_asyncio.fixture
async def backend():
    """In-memory SQLite backend, initialized and torn down per test."""
    async with SQLiteBackend(":memory:") as db:
        yield db


@pytest_asyncio.fixture
def mock_llm():
    from examples.mock_llm import MockLLMClient
    return MockLLMClient()
