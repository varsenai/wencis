# Changelog

All notable changes to the Wencis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-07-07

### Added
- **Causal Knowledge Graph**: Implemented an async epistemic graph tracker (`CausalKnowledgeGraph`) supporting `decision`, `hypothesis`, `fact`, and `dead_end` node types.
- **Recursive Cycle-Guarded CTE**: Implemented SQLite recursive Common Table Expressions (CTE) with JSON-based visited-path cycle checking to trace back failure paths.
- **Response Critic**: Implemented pre-response critiques checking for Accuracy, Depth, and Honesty, using programmatic geometric mean checks to block unacceptable drafts.
- **Trajectory Optimizer**: Added operators to revise step categories, compress verbose execution outputs (>500 characters) via LLM summaries, and splice/recombine successful trajectories.
- **Meta-Reasoning Engine**: Implemented performance drift analysis fitting turn confidence scores to an Ordinary Least Squares (OLS) linear regression model.
- **Compliance Markers**: Added empty `py.typed` to support PEP 561 downstream static type checking.
- **Developer Assets**: Created standard open-source documentation including `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, and a transparent vector SVG schematic logo.

### Fixed (Post-Audit Security & Concurrency Refinement)
- **Reentrant Transaction Contamination**: Resolved a critical concurrency bug where multiple concurrent tasks sharing a database backend could corrupt each other's transactions. Implemented a task-ownership lock with `asyncio.Lock` serialization on `SQLiteBackend`.
- **Validation Bounds crash**: Replaced `revised_category: str` with `Literal["decision", "hypothesis", "fact", "dead_end"]` in Pydantic schemas to prevent database IntegrityError exceptions on invalid LLM output values.
- **Database Scale performance**: Created descending indexes on the `created_at` fields of `turns`, `improvement_logs`, `action_logs`, and `uncertainties` tables to prevent full sequential table scans.
- **Node Tamper-proofing**: Expanded the `integrity_hash` payload in `EpistemicNode` to cover the `node_id`, `session_id`, `run_id`, and sorted `metadata` parameters, preventing silent record modification.
- **Critic Outage Logging**: Added the `is_fallback: bool` flag to prevent LLM errors from polluting confidence drift telemetry with fake perfect scores.
