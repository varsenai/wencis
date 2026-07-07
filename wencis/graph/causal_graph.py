# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
CausalKnowledgeGraph — manages registration, querying, and archiving
of epistemic nodes and causal edges.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from wencis.storage.protocol import StorageBackend
from wencis.graph.nodes import EpistemicNode, NodeType
from wencis.graph.edges import CausalEdge, EdgeType

log = logging.getLogger("wencis.graph.causal_graph")


class CausalKnowledgeGraph:
    def __init__(self, backend: StorageBackend):
        self.backend = backend

    async def register_node(self, node: EpistemicNode) -> bool:
        """
        Inserts a node into epistemic_nodes.
        Returns True on success.
        If UNIQUE constraint violation, returns True (idempotent).
        On any other exception, returns False.
        """
        sql = """
        INSERT INTO epistemic_nodes
            (node_id, run_id, session_id, timestamp, type, content,
             provenance, integrity_hash, metadata, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """
        params = (
            node.node_id,
            node.run_id,
            node.session_id,
            node.timestamp,
            node.type,
            node.content,
            node.provenance,
            node.integrity_hash,
            json.dumps(node.metadata),
        )
        try:
            await self.backend.execute(sql, params)
            return True
        except Exception as e:
            # Check if it is a UNIQUE constraint / integrity error
            err_str = str(e).lower()
            if "unique" in err_str or "integrity" in err_str:
                log.warning("Idempotent registration of duplicate node: %s", node.node_id)
                return True
            log.error("Failed to register node %s: %s", node.node_id, e)
            return False

    async def register_edge(self, edge: CausalEdge) -> bool:
        """
        Inserts an edge into epistemic_edges.
        Returns True on success.
        If UNIQUE constraint violation, returns True (idempotent).
        On any other exception, returns False.
        """
        sql = """
        INSERT INTO epistemic_edges
            (edge_id, source_node_id, target_node_id, relation_type, weight)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (
            edge.edge_id,
            edge.source_node_id,
            edge.target_node_id,
            edge.relation_type,
            edge.weight,
        )
        try:
            await self.backend.execute(sql, params)
            return True
        except Exception as e:
            err_str = str(e).lower()
            if "unique" in err_str or "integrity" in err_str:
                log.warning("Idempotent registration of duplicate edge: %s", edge.edge_id)
                return True
            log.error("Failed to register edge %s: %s", edge.edge_id, e)
            return False

    async def register_observation(
        self,
        session_id: str,
        type: NodeType,
        content: str,
        provenance: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_node_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
        edge_weight: float = 1.0,
    ) -> Optional[str]:
        """
        Registers an observation. If parent_node_id is provided, creates a causal edge.
        Both parent_node_id and edge_type must be set or both must be None.
        Returns the new node_id on success, or None on failure.
        """
        if (parent_node_id is None) != (edge_type is None):
            raise ValueError(
                "Both parent_node_id and edge_type must be specified, or both must be None."
            )

        node = EpistemicNode.new(
            session_id=session_id,
            type=type,
            content=content,
            provenance=provenance,
            run_id=run_id,
            metadata=metadata,
        )

        async with self.backend.transaction():
            node_ok = await self.register_node(node)
            if not node_ok:
                return None

            if parent_node_id is not None and edge_type is not None:
                edge = CausalEdge.new(
                    source_node_id=parent_node_id,
                    target_node_id=node.node_id,
                    relation_type=edge_type,
                    weight=edge_weight,
                )
                edge_ok = await self.register_edge(edge)
                if not edge_ok:
                    return None

        return node.node_id

    async def query_traceback_tree(
        self, terminal_node_id: str, max_depth: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Uses a recursive CTE to traverse backwards along causal paths
        starting from the terminal node.
        Returns a list of dicts representing the causal path nodes in order of depth.
        """
        sql = """
        WITH RECURSIVE CausalTrace(
            node_id, parent_id, link_type, depth, visited_path
        ) AS (
            -- Anchor: start at the terminal (failed) node
            SELECT
                ?,
                NULL,
                'terminal_sink',
                0,
                json_array(?)

            UNION ALL

            -- Recursive step: walk backwards along ALL incoming edges
            SELECT
                ee.source_node_id,
                ct.node_id,
                ee.relation_type,
                ct.depth + 1,
                json_insert(ct.visited_path, '$[#]', ee.source_node_id)
            FROM epistemic_edges ee
            INNER JOIN CausalTrace ct ON ee.target_node_id = ct.node_id
            WHERE
                ct.depth < ?
                AND ee.source_node_id NOT IN (
                    SELECT value FROM json_each(ct.visited_path)
                )
        )
        SELECT
            ct.node_id,
            ct.parent_id,
            ct.link_type,
            ct.depth,
            en.type       AS node_type,
            en.content,
            en.provenance,
            en.integrity_hash,
            en.metadata,
            en.timestamp,
            en.status
        FROM CausalTrace ct
        INNER JOIN epistemic_nodes en ON ct.node_id = en.node_id
        ORDER BY ct.depth ASC
        """
        params = (terminal_node_id, terminal_node_id, max_depth)
        try:
            rows = await self.backend.fetch_all(sql, params)
            for row in rows:
                if "metadata" in row and isinstance(row["metadata"], str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        pass
            return rows
        except Exception as e:
            log.error("Failed to query traceback tree for %s: %s", terminal_node_id, e)
            return []

    async def get_recent_trajectory(
        self, session_id: str, limit: int = 20, include_types: Optional[List[NodeType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches the most recent active nodes for a session.
        """
        if include_types:
            placeholders = ",".join("?" for _ in include_types)
            sql = f"""
            SELECT node_id, type, content, provenance, timestamp, metadata, status
            FROM epistemic_nodes
            WHERE session_id = ? AND status = 'active' AND type IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT ?
            """
            params = (session_id, *include_types, limit)
        else:
            sql = """
            SELECT node_id, type, content, provenance, timestamp, metadata, status
            FROM epistemic_nodes
            WHERE session_id = ? AND status = 'active'
            ORDER BY timestamp DESC
            LIMIT ?
            """
            params = (session_id, limit)

        try:
            rows = await self.backend.fetch_all(sql, params)
            for row in rows:
                if "metadata" in row and isinstance(row["metadata"], str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        pass
            return rows
        except Exception as e:
            log.error("Failed to fetch trajectory for session %s: %s", session_id, e)
            return []

    async def get_dead_ends_for_session(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Returns active dead_end nodes for a session.
        """
        sql = """
        SELECT node_id, content, provenance, timestamp, metadata
        FROM epistemic_nodes
        WHERE session_id = ? AND type = 'dead_end' AND status = 'active'
        ORDER BY timestamp DESC
        LIMIT ?
        """
        try:
            rows = await self.backend.fetch_all(sql, (session_id, limit))
            for row in rows:
                if "metadata" in row and isinstance(row["metadata"], str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        pass
            return rows
        except Exception as e:
            log.error("Failed to fetch dead ends for session %s: %s", session_id, e)
            return []

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a single node by node_id.
        """
        sql = "SELECT * FROM epistemic_nodes WHERE node_id = ?"
        try:
            row = await self.backend.fetch_one(sql, (node_id,))
            if row:
                if "metadata" in row and isinstance(row["metadata"], str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        pass
                return row
            return None
        except Exception as e:
            log.error("Failed to fetch node %s: %s", node_id, e)
            return None

    async def archive_old_nodes(self, days: int = 30) -> int:
        """
        Soft-deletes active nodes older than days that are NOT referenced
        by any active edge. Sets status = 'archived'.
        Returns count of archived nodes.
        """
        cutoff = time.time() - (days * 86400)
        sql = """
        UPDATE epistemic_nodes
        SET status = 'archived'
        WHERE status = 'active'
          AND timestamp < ?
          AND node_id NOT IN (
              SELECT source_node_id FROM epistemic_edges
              UNION
              SELECT target_node_id FROM epistemic_edges
          )
        """
        try:
            cursor = await self.backend.execute(sql, (cutoff,))
            # Row count is available via rowcount attribute on sqlite cursor
            return cursor.rowcount
        except Exception as e:
            log.error("Failed to archive old nodes: %s", e)
            return 0
