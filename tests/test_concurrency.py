# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pytest
from wencis.graph.causal_graph import CausalKnowledgeGraph
from wencis.graph.nodes import EpistemicNode


@pytest.mark.asyncio
async def test_reentrant_transaction_concurrency(backend):
    graph = CausalKnowledgeGraph(backend)
    events = []

    async def task_1():
        async with backend.transaction():
            events.append("task_1_start")
            # Create a node
            await graph.register_node(
                EpistemicNode.new(session_id="session-1", type="decision", content="T1", provenance="test")
            )
            # Yield control to let task_2 run
            await asyncio.sleep(0.1)
            events.append("task_1_end")

    async def task_2():
        # Task 2 should wait for Task 1's transaction lock to be released
        async with backend.transaction():
            events.append("task_2_start")
            await graph.register_node(
                EpistemicNode.new(session_id="session-1", type="decision", content="T2", provenance="test")
            )
            events.append("task_2_end")

    # Run concurrently
    await asyncio.gather(task_1(), task_2())

    # Assert sequence is serialized: task_1 must fully complete before task_2 starts
    assert events == ["task_1_start", "task_1_end", "task_2_start", "task_2_end"]
