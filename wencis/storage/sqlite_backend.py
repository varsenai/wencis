# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
SQLiteBackend — production-ready async SQLite implementation of StorageBackend.

Design decisions:
- WAL journal mode: allows concurrent reads while a write is in progress
- foreign_keys=ON: enforces referential integrity (CASCADE deletes work)
- busy_timeout=15000: wait up to 15s before raising SQLITE_BUSY
- Schema is applied on first connect; migrations run on each connect
- Supports reentrant nested transactions via Task-ownership tracking and Lock
  to prevent dirty reads and transaction contamination in concurrent tasks.
"""
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

SCHEMA_SQL = """
-- ── Epistemic Graph ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS epistemic_nodes (
    node_id       TEXT PRIMARY KEY,
    run_id        TEXT,
    session_id    TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    type          TEXT NOT NULL CHECK(type IN ('decision', 'hypothesis', 'fact', 'dead_end')),
    content       TEXT NOT NULL,
    provenance    TEXT NOT NULL,
    integrity_hash TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_epistemic_nodes_session
    ON epistemic_nodes(session_id, status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_epistemic_nodes_type
    ON epistemic_nodes(type, session_id);

CREATE TABLE IF NOT EXISTS epistemic_edges (
    edge_id         TEXT PRIMARY KEY,
    source_node_id  TEXT NOT NULL REFERENCES epistemic_nodes(node_id) ON DELETE CASCADE,
    target_node_id  TEXT NOT NULL REFERENCES epistemic_nodes(node_id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL CHECK(relation_type IN (
        'triggered_by', 'contradicts', 'prevented', 'caused_failure_in'
    )),
    weight          REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_epistemic_edges_source ON epistemic_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_epistemic_edges_target ON epistemic_edges(target_node_id);

-- ── Trajectory Tables ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id       TEXT PRIMARY KEY,
    task_description    TEXT NOT NULL,
    is_success          INTEGER NOT NULL CHECK(is_success IN (0, 1)),
    cumulative_latency  REAL NOT NULL,
    total_tokens        INTEGER NOT NULL,
    timestamp           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    step_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id       TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
    step_order          INTEGER NOT NULL,
    action_name         TEXT NOT NULL,
    tool_input          TEXT NOT NULL,
    execution_output    TEXT NOT NULL,
    epistemic_category  TEXT NOT NULL CHECK(epistemic_category IN (
        'decision', 'hypothesis', 'fact', 'dead_end'
    )),
    latency_ms          REAL NOT NULL,
    token_usage         INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trajectory_steps_order
    ON trajectory_steps(trajectory_id, step_order);

-- ── Improvement Tables ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS improvement_logs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    original_response   TEXT NOT NULL,
    feedback        TEXT NOT NULL,
    accuracy_score  REAL NOT NULL,
    depth_score     REAL NOT NULL,
    honesty_score   REAL NOT NULL,
    improved_response   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_improvement_logs_created ON improvement_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS improvement_proposals (
    id              TEXT PRIMARY KEY,
    target_system   TEXT NOT NULL,
    description     TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    success_metric  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK(status IN (
        'pending', 'approved', 'rejected', 'implemented'
    )),
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);

-- ── Agent Turn Confidence ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    confidence  REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at DESC);

-- ── Action Logs (tool failures) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS action_logs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_number     INTEGER NOT NULL DEFAULT 0,
    tool_name       TEXT NOT NULL,
    actual_outcome  TEXT NOT NULL,
    success         INTEGER NOT NULL CHECK(success IN (0, 1)),
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_action_logs_created ON action_logs(created_at DESC);

-- ── Uncertainty Tracking ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS uncertainties (
    id              TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    why_uncertain   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'resolved')),
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_uncertainties_created ON uncertainties(created_at DESC);
"""


class SQLiteBackend:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._transaction_depth = 0
        self._tx_lock = asyncio.Lock()
        self._tx_owner: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=15000")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self.initialize_schema()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def initialize_schema(self) -> None:
        if self._conn is None:
            raise RuntimeError("Database not connected.")
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()

    async def execute(self, sql: str, params: tuple = ()) -> Any:
        if self._conn is None:
            raise RuntimeError("Database not connected.")
        current_task = asyncio.current_task()
        own_tx = (self._tx_owner == current_task)
        if not own_tx:
            await self._tx_lock.acquire()
        try:
            cursor = await self._conn.execute(sql, params)
            if not own_tx:
                await self._conn.commit()
            return cursor
        finally:
            if not own_tx:
                self._tx_lock.release()

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        if self._conn is None:
            raise RuntimeError("Database not connected.")
        current_task = asyncio.current_task()
        own_tx = (self._tx_owner == current_task)
        if not own_tx:
            await self._tx_lock.acquire()
        try:
            cursor = await self._conn.execute(sql, params)
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            if not own_tx:
                self._tx_lock.release()

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        if self._conn is None:
            raise RuntimeError("Database not connected.")
        current_task = asyncio.current_task()
        own_tx = (self._tx_owner == current_task)
        if not own_tx:
            await self._tx_lock.acquire()
        try:
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            if not own_tx:
                self._tx_lock.release()

    @asynccontextmanager
    async def transaction(self):
        if self._conn is None:
            raise RuntimeError("Database not connected.")
        current_task = asyncio.current_task()
        own_tx = (self._tx_owner == current_task)
        if not own_tx:
            await self._tx_lock.acquire()
            self._tx_owner = current_task
            self._transaction_depth = 0
            try:
                await self._conn.execute("BEGIN IMMEDIATE")
            except Exception:
                self._tx_owner = None
                self._tx_lock.release()
                raise

        self._transaction_depth += 1
        try:
            yield
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                await self._conn.commit()
                self._tx_owner = None
                self._tx_lock.release()
        except Exception:
            if self._transaction_depth > 0:
                await self._conn.rollback()
                self._transaction_depth = 0
                self._tx_owner = None
                self._tx_lock.release()
            raise

    async def __aenter__(self) -> "SQLiteBackend":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
