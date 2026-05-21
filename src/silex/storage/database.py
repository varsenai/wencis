"""
SQLite database layer for ARIA.

Handles connection management, schema creation, and migrations.
All operations are async via aiosqlite.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

import aiosqlite

from silex.utils.config import VYN_DB
from silex.utils.logger import setup_logger

log = setup_logger("silex.storage")

transaction_depth_var: ContextVar[int] = ContextVar("transaction_depth_var", default=0)

# ---------------------------------------------------------------------------
# Schema — this IS the database definition
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT '[]',
    level INTEGER NOT NULL DEFAULT 1,
    child_memory_ids TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    related_memories TEXT NOT NULL DEFAULT '[]',
    superseded_by_id TEXT,
    archived_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC);

-- FTS5 virtual table for fast full-text search on memories (Phase 19)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    tags,
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with the memories table
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (NEW.rowid, NEW.content, NEW.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content, tags ON memories BEGIN
    UPDATE memories_fts SET content = NEW.content, tags = NEW.tags WHERE rowid = NEW.rowid;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = OLD.rowid;
END;

-- Goals table
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    priority TEXT NOT NULL DEFAULT 'medium',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    sub_goals TEXT NOT NULL DEFAULT '[]',
    completion_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    turn_count INTEGER NOT NULL DEFAULT 0,
    memories_created INTEGER NOT NULL DEFAULT 0,
    goals_modified INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL NOT NULL DEFAULT 0.0,
    topics TEXT NOT NULL DEFAULT '[]'
);

-- Turns table (conversation history)
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    user_input TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    response TEXT NOT NULL,
    self_reflection TEXT NOT NULL,
    confidence REAL NOT NULL,
    scratchpad TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_number);

-- =====================================================================
-- Phase 2 — World Model Tables
-- =====================================================================

-- Knowledge nodes (the graph's vertices)
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'fact',
    confidence REAL NOT NULL DEFAULT 0.5,
    source TEXT NOT NULL DEFAULT 'inference',
    created_at TEXT NOT NULL,
    last_validated TEXT NOT NULL,
    validation_count INTEGER NOT NULL DEFAULT 0,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    metadata TEXT NOT NULL DEFAULT '{}',
    valid_at TEXT,
    invalid_at TEXT,
    invalidated_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON knowledge_nodes(confidence DESC);

-- Causal edges (the graph's typed relationships)
CREATE TABLE IF NOT EXISTS causal_edges (
    id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    evidence TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_node) REFERENCES knowledge_nodes(id),
    FOREIGN KEY (target_node) REFERENCES knowledge_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON causal_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_edges_target ON causal_edges(target_node);
CREATE INDEX IF NOT EXISTS idx_edges_type ON causal_edges(edge_type);

-- Hypotheses (predictions from the world model)
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);

-- Contradictions (conflicts between knowledge nodes)
CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    node_a TEXT NOT NULL,
    node_b TEXT NOT NULL,
    analysis TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unresolved',
    resolution TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (node_a) REFERENCES knowledge_nodes(id),
    FOREIGN KEY (node_b) REFERENCES knowledge_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions(status);

-- =====================================================================
-- Phase 7 — Semantic Disambiguation
-- =====================================================================

-- Semantic profiles (learned subjective-to-objective mappings)
CREATE TABLE IF NOT EXISTS semantic_profiles (
    term TEXT PRIMARY KEY,
    objective_proxies TEXT NOT NULL, -- JSON list
    context_tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    updated_at TEXT NOT NULL
);

-- =====================================================================
-- Phase 3 — Self-Improvement Tables
-- =====================================================================

CREATE TABLE IF NOT EXISTS improvement_logs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    original_response TEXT NOT NULL,
    feedback TEXT NOT NULL,
    accuracy_score REAL NOT NULL,
    depth_score REAL NOT NULL,
    honesty_score REAL NOT NULL,
    improved_response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- =====================================================================
-- Phase 4 — Multi-Agent Debate Tables
-- =====================================================================

CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    resolution_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS uncertainties (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    why_uncertain TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);

-- =====================================================================
-- Phase 5 — Tool Use & Action Logs
-- =====================================================================

CREATE TABLE IF NOT EXISTS action_logs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    expected_outcome TEXT NOT NULL,
    actual_outcome TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'read_only',
    model_update TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    expected_outcome TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    execution_result_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_approvals_status ON tool_approvals(status, created_at);

CREATE TABLE IF NOT EXISTS ethical_decisions (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    turn_number INTEGER NOT NULL DEFAULT 0,
    tool_name TEXT NOT NULL,
    principle TEXT NOT NULL,
    action TEXT NOT NULL,
    rationale TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'read_only',
    requires_consent BOOLEAN NOT NULL DEFAULT 0,
    uncertainty REAL NOT NULL DEFAULT 0.0,
    context TEXT NOT NULL DEFAULT 'interactive',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_ethical_decisions_session ON ethical_decisions(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ethical_decisions_action ON ethical_decisions(action, created_at);

CREATE TABLE IF NOT EXISTS recent_failures (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    failure_type TEXT NOT NULL, -- 'critic_rejection', 'tool_error', 'consistency_mismatch'
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_recent_failures_session ON recent_failures(session_id, created_at);

-- =====================================================================
-- Phase 6 — Transfer + Generalization
-- =====================================================================

CREATE TABLE IF NOT EXISTS principles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    statement TEXT NOT NULL,
    original_domain TEXT NOT NULL,
    applicable_domains_json TEXT NOT NULL,
    source_observations_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- =====================================================================
-- Phase 7 — Recursive Self-Improvement
-- =====================================================================

CREATE TABLE IF NOT EXISTS improvement_proposals (
    id TEXT PRIMARY KEY,
    target_system TEXT NOT NULL,
    description TEXT NOT NULL,
    rationale TEXT NOT NULL,
    success_metric TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_history (
    id TEXT PRIMARY KEY,
    total_score REAL NOT NULL,
    accuracy_avg REAL NOT NULL,
    depth_avg REAL NOT NULL,
    honesty_avg REAL NOT NULL,
    domains_tested_json TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_kind TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    success BOOLEAN NOT NULL DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_model ON llm_usage(provider, model, created_at DESC);

-- =====================================================================
-- Durable Planning
-- =====================================================================

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    title TEXT NOT NULL,
    user_input TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    success_criteria TEXT NOT NULL DEFAULT '',
    tool_budget INTEGER NOT NULL DEFAULT 8,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_plans_session ON plans(session_id, status);

CREATE TABLE IF NOT EXISTS plan_steps (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    required_tools_json TEXT NOT NULL DEFAULT '[]',
    result TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES plans(id)
);

CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id, step_number);

CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'info',
    delivered INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_checkpoints (
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    draft_reasoning TEXT NOT NULL,
    draft_plan TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'executing_tools',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_number)
);

CREATE TABLE IF NOT EXISTS response_cache (
    query_hash TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Durable agent messages (durable Silex message bus)
CREATE TABLE IF NOT EXISTS agent_messages (
    id TEXT PRIMARY KEY,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    read_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_routing ON agent_messages(recipient, read_at);
"""

MIGRATIONS_SQL = [
    "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'semantic'",
    "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5",
    "ALTER TABLE memories ADD COLUMN level INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE memories ADD COLUMN child_memory_ids TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE memories ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE memories ADD COLUMN archived_at TEXT",
    "ALTER TABLE action_logs ADD COLUMN risk_level TEXT NOT NULL DEFAULT 'read_only'",
    "ALTER TABLE turns ADD COLUMN scratchpad TEXT",
    "ALTER TABLE knowledge_nodes ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'unverified'",
    "CREATE TABLE IF NOT EXISTS ethical_decisions (id TEXT PRIMARY KEY, session_id TEXT, turn_number INTEGER NOT NULL DEFAULT 0, tool_name TEXT NOT NULL, principle TEXT NOT NULL, action TEXT NOT NULL, rationale TEXT NOT NULL, risk_level TEXT NOT NULL DEFAULT 'read_only', requires_consent BOOLEAN NOT NULL DEFAULT 0, uncertainty REAL NOT NULL DEFAULT 0.0, context TEXT NOT NULL DEFAULT 'interactive', created_at TEXT NOT NULL, FOREIGN KEY (session_id) REFERENCES sessions(id))",
    # Indexes on columns added by migrations must run after ALTERs (older DBs skip CREATE TABLE).
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived_at)",
    "CREATE INDEX IF NOT EXISTS idx_ethical_decisions_session ON ethical_decisions(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ethical_decisions_action ON ethical_decisions(action, created_at)",
    "ALTER TABLE tool_approvals ADD COLUMN expected_outcome TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE tool_approvals ADD COLUMN execution_result_json TEXT",
    "CREATE TABLE IF NOT EXISTS llm_usage (id TEXT PRIMARY KEY, session_id TEXT, provider TEXT NOT NULL, model TEXT NOT NULL, request_kind TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, estimated_cost_usd REAL, duration_ms INTEGER NOT NULL DEFAULT 0, success BOOLEAN NOT NULL DEFAULT 1, error TEXT, created_at TEXT NOT NULL, FOREIGN KEY (session_id) REFERENCES sessions(id))",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_model ON llm_usage(provider, model, created_at DESC)",
    "CREATE TABLE IF NOT EXISTS notifications (id TEXT PRIMARY KEY, message TEXT NOT NULL, level TEXT NOT NULL DEFAULT 'info', delivered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS turn_checkpoints (session_id TEXT NOT NULL, turn_number INTEGER NOT NULL, draft_reasoning TEXT NOT NULL, draft_plan TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'executing_tools', updated_at TEXT NOT NULL, PRIMARY KEY (session_id, turn_number))",
    "CREATE TABLE IF NOT EXISTS response_cache (query_hash TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL)",
    "ALTER TABLE knowledge_nodes ADD COLUMN valid_at TEXT",
    "ALTER TABLE knowledge_nodes ADD COLUMN invalid_at TEXT",
    "ALTER TABLE knowledge_nodes ADD COLUMN invalidated_by TEXT",
    "CREATE TABLE IF NOT EXISTS agent_messages (id TEXT PRIMARY KEY, sender TEXT NOT NULL, recipient TEXT NOT NULL, message_type TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, read_at TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_agent_messages_routing ON agent_messages(recipient, read_at)",
    "ALTER TABLE memories ADD COLUMN superseded_by_id TEXT",
    # Phase 19 — FTS5 backfill: populate the FTS index from existing memories
    # The INSERT OR IGNORE ensures idempotency on re-run
    "INSERT OR IGNORE INTO memories_fts(rowid, content, tags) SELECT rowid, content, tags FROM memories",
]


# ---------------------------------------------------------------------------
# Database connection management
# ---------------------------------------------------------------------------

class Database:
    """Async SQLite database wrapper for ARIA."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(VYN_DB)
        self._conn: aiosqlite.Connection | None = None
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the database connection and ensure schema exists."""
        async with self._lifecycle_lock:
            if self._conn is not None:
                return  # Already connected
            log.info(f"Connecting to database: {self.db_path}")
            self._conn = await aiosqlite.connect(self.db_path, timeout=15.0)
            self._conn.row_factory = aiosqlite.Row

            # Enable WAL mode + settings for robust multi-process concurrency
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute("PRAGMA busy_timeout=15000")
            await self._conn.execute("PRAGMA foreign_keys=ON")

            # Create tables if they don't exist
            await self._conn.executescript(SCHEMA_SQL)
            await self._run_migrations()
            await self._conn.commit()
        log.info("Database schema initialized")

    async def _run_migrations(self) -> None:
        """Apply additive migrations for existing local SQLite brains."""
        for i, sql in enumerate(MIGRATIONS_SQL):
            try:
                await self._conn.execute(sql)
            except aiosqlite.OperationalError as e:
                msg = str(e).lower()
                if "duplicate column name" in msg or "already exists" in msg:
                    continue  # Expected for idempotent migrations
                log.error(f"Migration {i} failed: {sql[:80]}... → {e}")
                raise

    async def close(self) -> None:
        """Close the database connection."""
        async with self._lifecycle_lock:
            if self._conn:
                await self._conn.close()
                self._conn = None
                log.info("Database connection closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        """Get the active connection or fail."""
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a single SQL statement. Auto-commits unless inside a transaction()."""
        cursor = await self.conn.execute(sql, params)
        if transaction_depth_var.get() == 0:
            await self.conn.commit()
        return cursor

    @asynccontextmanager
    async def transaction(self):
        """
        Atomic transaction context manager.
        Supports nested transactions via SAVEPOINTs.
        Outermost transaction uses BEGIN IMMEDIATE for write-lock acquisition.
        Auto-commits when the outermost transaction completes successfully.
        Auto-rollbacks if any exception bubbles up.
        """
        depth = transaction_depth_var.get()
        if depth == 0:
            await self.conn.execute("BEGIN IMMEDIATE")
        else:
            await self.conn.execute(f"SAVEPOINT sp_{depth}")
        transaction_depth_var.set(depth + 1)
        try:
            yield
            if transaction_depth_var.get() == 1:
                await self.conn.commit()
            else:
                await self.conn.execute(f"RELEASE SAVEPOINT sp_{depth}")
        except BaseException:
            if transaction_depth_var.get() == 1:
                await self.conn.rollback()
            else:
                try:
                    await self.conn.execute(f"ROLLBACK TO SAVEPOINT sp_{depth}")
                except Exception:
                    pass  # Savepoint may already be released
            raise
        finally:
            transaction_depth_var.set(transaction_depth_var.get() - 1)

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Fetch a single row as a dict."""
        cursor = await self.conn.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as dicts."""
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
