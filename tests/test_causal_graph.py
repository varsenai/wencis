# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import asyncio
from wencis.graph.nodes import EpistemicNode
from wencis.graph.edges import CausalEdge
from wencis.graph.causal_graph import CausalKnowledgeGraph


@pytest.mark.asyncio
async def test_register_node_idempotent(backend):
    graph = CausalKnowledgeGraph(backend)
    node = EpistemicNode.new(
        session_id="session-1",
        type="decision",
        content="Testing",
        provenance="test",
    )
    # First registration
    ok1 = await graph.register_node(node)
    assert ok1 is True

    # Second registration (idempotent)
    ok2 = await graph.register_node(node)
    assert ok2 is True


@pytest.mark.asyncio
async def test_register_observation_creates_node_and_edge(backend):
    graph = CausalKnowledgeGraph(backend)
    # 1. Create a parent node
    parent_id = await graph.register_observation(
        session_id="session-1",
        type="decision",
        content="Parent decision",
        provenance="test",
    )
    assert parent_id is not None

    # 2. Create child node referencing parent
    child_id = await graph.register_observation(
        session_id="session-1",
        type="hypothesis",
        content="Child hypothesis",
        provenance="test",
        parent_node_id=parent_id,
        edge_type="triggered_by",
    )
    assert child_id is not None

    # Verify both exist in DB
    node1 = await graph.get_node(parent_id)
    node2 = await graph.get_node(child_id)
    assert node1["content"] == "Parent decision"
    assert node2["content"] == "Child hypothesis"

    # Verify edge exists
    edges = await backend.fetch_all(
        "SELECT * FROM epistemic_edges WHERE source_node_id = ? AND target_node_id = ?",
        (parent_id, child_id),
    )
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "triggered_by"


@pytest.mark.asyncio
async def test_register_observation_validates_parent_edge_pair(backend):
    graph = CausalKnowledgeGraph(backend)
    with pytest.raises(ValueError):
        await graph.register_observation(
            session_id="session-1",
            type="hypothesis",
            content="Invalid combination",
            provenance="test",
            parent_node_id="some-parent",
            edge_type=None,
        )


@pytest.mark.asyncio
async def test_traceback_walks_to_root(backend):
    graph = CausalKnowledgeGraph(backend)
    # Chain: decision -> hypothesis -> dead_end
    node1_id = await graph.register_observation(
        session_id="session-1", type="decision", content="Step 1 decision", provenance="test"
    )
    node2_id = await graph.register_observation(
        session_id="session-1",
        type="hypothesis",
        content="Step 2 hypothesis",
        provenance="test",
        parent_node_id=node1_id,
        edge_type="triggered_by",
    )
    node3_id = await graph.register_observation(
        session_id="session-1",
        type="dead_end",
        content="Step 3 dead end",
        provenance="test",
        parent_node_id=node2_id,
        edge_type="caused_failure_in",
    )

    chain = await graph.query_traceback_tree(node3_id)
    assert len(chain) == 3
    # depth=0 should be terminal node3
    assert chain[0]["node_id"] == node3_id
    assert chain[0]["depth"] == 0
    # depth=1 should be node2
    assert chain[1]["node_id"] == node2_id
    assert chain[1]["depth"] == 1
    # depth=2 should be node1
    assert chain[2]["node_id"] == node1_id
    assert chain[2]["depth"] == 2


@pytest.mark.asyncio
async def test_traceback_detects_cycles(backend):
    graph = CausalKnowledgeGraph(backend)
    # Create nodes
    nodeA_id = await graph.register_observation(
        session_id="session-1", type="decision", content="A", provenance="test"
    )
    nodeB_id = await graph.register_observation(
        session_id="session-1", type="decision", content="B", provenance="test"
    )

    # Manually insert cycle: A -> B and B -> A
    await graph.register_edge(CausalEdge.new(nodeA_id, nodeB_id, "triggered_by"))
    await graph.register_edge(CausalEdge.new(nodeB_id, nodeA_id, "triggered_by"))

    # Running query_traceback_tree should return without hanging (max depth checks)
    chain = await graph.query_traceback_tree(nodeA_id, max_depth=5)
    assert len(chain) >= 2


@pytest.mark.asyncio
async def test_archive_old_nodes(backend):
    graph = CausalKnowledgeGraph(backend)

    # 1. Insert an old unlinked node
    old_unlinked = EpistemicNode.new(
        session_id="session-1", type="fact", content="Old unlinked", provenance="test"
    )
    old_unlinked.timestamp = time.time() - (40 * 86400)  # 40 days ago
    await graph.register_node(old_unlinked)

    # 2. Insert an old linked node (part of a chain)
    old_linked_parent = EpistemicNode.new(
        session_id="session-1", type="decision", content="Old linked parent", provenance="test"
    )
    old_linked_parent.timestamp = time.time() - (40 * 86400)
    await graph.register_node(old_linked_parent)

    old_linked_child = EpistemicNode.new(
        session_id="session-1", type="fact", content="Old linked child", provenance="test"
    )
    old_linked_child.timestamp = time.time() - (40 * 86400)
    await graph.register_node(old_linked_child)

    edge = CausalEdge.new(old_linked_parent.node_id, old_linked_child.node_id, "triggered_by")
    await graph.register_edge(edge)

    # Run archive
    archived_count = await graph.archive_old_nodes(days=30)
    assert archived_count == 1  # Only old_unlinked is archived

    # Check status
    node1 = await graph.get_node(old_unlinked.node_id)
    node2 = await graph.get_node(old_linked_parent.node_id)
    node3 = await graph.get_node(old_linked_child.node_id)

    assert node1["status"] == "archived"
    assert node2["status"] == "active"
    assert node3["status"] == "active"


def test_integrity_hash_verification():
    node = EpistemicNode.new(
        session_id="session-1",
        type="decision",
        content="Clean content",
        provenance="test",
        metadata={"foo": "bar"},
    )
    assert node.verify_integrity() is True

    # Alter metadata: should fail verification
    node.metadata = {"foo": "changed"}
    assert node.verify_integrity() is False

    # Restore metadata: should pass again
    node.metadata = {"foo": "bar"}
    assert node.verify_integrity() is True

    # Alter session_id: should fail verification
    node.session_id = "session-2"
    assert node.verify_integrity() is False

