# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
StorageBackend — abstract interface for all persistence in Wencis.

Any backend must implement these five coroutines. The SQLite backend
is the reference implementation. A dict-based in-memory backend is
provided for testing.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol, Sequence, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal async persistence interface required by all Wencis components."""

    async def execute(
        self, sql: str, params: tuple = ()
    ) -> Any:
        """
        Execute a single DML statement (INSERT, UPDATE, DELETE).
        Returns a cursor-like object. The caller may call .fetchall()
        or .fetchone() on it if needed (e.g. for RETURNING clauses).
        Must be safe to call concurrently with reads.
        """
        ...

    async def fetch_one(
        self, sql: str, params: tuple = ()
    ) -> dict | None:
        """
        Execute a SELECT and return the first row as a plain dict.
        Returns None if no rows matched. Column names are keys.
        """
        ...

    async def fetch_all(
        self, sql: str, params: tuple = ()
    ) -> list[dict]:
        """
        Execute a SELECT and return all rows as plain dicts.
        Returns an empty list if no rows matched. Column names are keys.
        """
        ...

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """
        Async context manager that wraps a block in an atomic transaction.
        On success: COMMIT. On exception: ROLLBACK.

        Must support nesting (inner transactions are no-ops that yield
        to the outer transaction's scope).

        Usage:
            async with backend.transaction():
                await backend.execute(...)
                await backend.execute(...)
        """
        ...
        yield

    async def initialize_schema(self) -> None:
        """
        Create all required tables and indexes if they do not exist.
        Safe to call multiple times (idempotent). Should run SCHEMA_SQL.
        """
        ...
