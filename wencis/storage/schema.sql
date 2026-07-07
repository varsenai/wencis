-- ── Epistemic Graph ──────────────────────────────────────────────────────────
-- Tracks the agent's OWN reasoning trajectory as a directed graph.
-- node types: decision, hypothesis, fact, dead_end
-- edge types: triggered_by, contradicts, prevented, caused_failure_in

CREATE TABLE IF NOT EXISTS epistemic_nodes (
    node_id       TEXT PRIMARY KEY,
    run_id        TEXT,                    -- Optional: groups nodes from one agent run
    session_id    TEXT NOT NULL,           -- Logical conversation/task session
    timestamp     REAL NOT NULL,           -- Unix epoch float
    type          TEXT NOT NULL CHECK(type IN ('decision', 'hypothesis', 'fact', 'dead_end')),
    content       TEXT NOT NULL,           -- Human-readable description of the node
    provenance    TEXT NOT NULL,           -- What produced this node (tool name, module, etc.)
    integrity_hash TEXT NOT NULL,          -- SHA-256 of type|content|provenance|timestamp
    metadata      TEXT NOT NULL DEFAULT '{}',  -- JSON blob for arbitrary extra data
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
-- Used by TrajectoryOptimizer. A trajectory = one complete agent run.
-- Steps = the individual tool calls / actions within that run.

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id       TEXT PRIMARY KEY,
    task_description    TEXT NOT NULL,
    is_success          INTEGER NOT NULL CHECK(is_success IN (0, 1)),
    cumulative_latency  REAL NOT NULL,     -- Total wall-clock seconds
    total_tokens        INTEGER NOT NULL,  -- Aggregated token usage
    timestamp           REAL NOT NULL      -- Unix epoch float
);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    step_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id       TEXT NOT NULL REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
    step_order          INTEGER NOT NULL,
    action_name         TEXT NOT NULL,     -- Tool or action name (e.g. "web_search")
    tool_input          TEXT NOT NULL,     -- JSON-serialized input to the tool
    execution_output    TEXT NOT NULL,     -- Raw output from tool execution
    epistemic_category  TEXT NOT NULL CHECK(epistemic_category IN (
        'decision', 'hypothesis', 'fact', 'dead_end'
    )),
    latency_ms          REAL NOT NULL,     -- Time this step took in milliseconds
    token_usage         INTEGER NOT NULL   -- Tokens consumed by this step
);

CREATE INDEX IF NOT EXISTS idx_trajectory_steps_order
    ON trajectory_steps(trajectory_id, step_order);

-- ── Improvement Tables ────────────────────────────────────────────────────────
-- Used by MetaReasoningEngine. Tracks critic rejections and proposals.

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
    target_system   TEXT NOT NULL,     -- system_prompt | tool_registry | cognitive_loop | memory_store | other
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
-- Minimal table used by MetaReasoningEngine for confidence drift analysis.
-- If integrating into an existing agent DB, this table may already exist.

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    confidence  REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at DESC);

-- ── Action Logs (tool failures) ───────────────────────────────────────────────
-- Minimal table for MetaReasoningEngine tool failure analysis.

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
