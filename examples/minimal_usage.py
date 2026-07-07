# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
Minimal example: record an agent's decision trace, then traceback a failure.
Uses MockLLMClient (no real API key needed).
"""
import asyncio
from wencis import CausalKnowledgeGraph, SQLiteBackend, EpistemicNode


async def main():
    async with SQLiteBackend(":memory:") as backend:
        graph = CausalKnowledgeGraph(backend)

        # Record a decision
        decision_id = await graph.register_observation(
            session_id="demo-session-001",
            type="decision",
            content="Executing shell command: pip install requests",
            provenance="tool:bash",
        )
        print(f"Recorded decision: {decision_id}")

        # Record a hypothesis triggered by the decision
        hypothesis_id = await graph.register_observation(
            session_id="demo-session-001",
            type="hypothesis",
            content="requests library will be installed in the active venv",
            provenance="reasoning:loop",
            parent_node_id=decision_id,
            edge_type="triggered_by",
        )
        print(f"Recorded hypothesis: {hypothesis_id}")

        # Record a failure
        dead_end_id = await graph.register_observation(
            session_id="demo-session-001",
            type="dead_end",
            content="ERROR: pip not found in PATH. Exit code 127.",
            provenance="tool:bash",
            parent_node_id=decision_id,
            edge_type="caused_failure_in",
        )
        print(f"Recorded failure: {dead_end_id}")

        # Traceback from the failure
        chain = await graph.query_traceback_tree(dead_end_id)
        print(f"\nCausal chain ({len(chain)} nodes):")
        for node in chain:
            print(f"  depth={node['depth']} type={node['node_type']} -> {node['content'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
