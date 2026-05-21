"""
Knowledge Graph Engine — ARIA's causal world model.

Uses NetworkX as an in-memory directed graph with SQLite persistence.
Every piece of knowledge is a node. Relationships are typed edges
(causes, enables, requires, contradicts, supports, part_of, similar_to, temporal).

The graph is loaded from SQLite on startup and saved on shutdown.
All mutations go through SQLite first (source of truth), then update the
in-memory graph.

Note: NetworkX is kept as the in-memory data structure for node/edge management.
Graph traversal algorithms (BFS path-finding, component counting) use
SQLite recursive CTEs directly — faster and without loading the full graph into RAM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import networkx as nx

from silex.models.schemas import (
    CausalEdge,
    EdgeType,
    KnowledgeNode,
    NodeType,
    VerificationStatus,
)
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.world.graph")


class KnowledgeGraph:
    """
    NetworkX-backed causal knowledge graph.

    Nodes are knowledge (facts, concepts, entities).
    Edges are typed causal relationships.
    """

    def __init__(self, db: Database):
        self.db = db
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # Inverted index: word → set of node_ids containing that word
        self._word_index: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load the graph from SQLite into memory."""
        # Load nodes
        node_rows = await self.db.fetch_all(
            "SELECT * FROM knowledge_nodes ORDER BY created_at"
        )
        for row in node_rows:
            self.graph.add_node(
                row["id"],
                content=row["content"],
                node_type=row["node_type"],
                confidence=row["confidence"],
                source=row["source"],
                created_at=row["created_at"],
                last_validated=row["last_validated"],
                validation_count=row["validation_count"],
                contradiction_count=row["contradiction_count"],
                verification_status=row.get("verification_status", "unverified"),
                metadata=json.loads(row["metadata"]),
                valid_at=row.get("valid_at"),
                invalid_at=row.get("invalid_at"),
                invalidated_by=row.get("invalidated_by"),
            )

        # Load edges
        edge_rows = await self.db.fetch_all(
            "SELECT * FROM causal_edges ORDER BY created_at"
        )
        for row in edge_rows:
            if row["source_node"] in self.graph and row["target_node"] in self.graph:
                self.graph.add_edge(
                    row["source_node"],
                    row["target_node"],
                    key=row["edge_type"],
                    id=row["id"],
                    edge_type=row["edge_type"],
                    strength=row["strength"],
                    evidence=row["evidence"],
                    created_at=row["created_at"],
                )

        log.info(
            f"Knowledge graph loaded: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        # Build the inverted index from loaded nodes
        self._rebuild_word_index()

    async def load_relevant(self, query: str | None, max_nodes: int = 200) -> None:
        """Phase A Bridge: Load only a relevant subgraph based on query to prevent 15s cold starts."""
        if not query:
            return await self.load()
            
        words = [w.lower() for w in query.split() if len(w) > 3]
        if not words:
            return await self.load()
            
        conditions = " OR ".join(["LOWER(content) LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        
        query_sql = f"""
            SELECT * FROM knowledge_nodes 
            WHERE {conditions}
            ORDER BY confidence DESC, validation_count DESC
            LIMIT ?
        """
        params.append(max_nodes)
        
        node_rows = await self.db.fetch_all(query_sql, tuple(params))
        
        # Ensure we have at least a baseline of high-confidence nodes if query was too narrow
        if len(node_rows) < 20:
            extra = await self.db.fetch_all(
                "SELECT * FROM knowledge_nodes ORDER BY confidence DESC, validation_count DESC LIMIT ?",
                (50,)
            )
            seen = {r["id"] for r in node_rows}
            for r in extra:
                if r["id"] not in seen:
                    node_rows.append(r)
        
        loaded_node_ids = set()
        for row in node_rows:
            self.graph.add_node(
                row["id"],
                content=row["content"],
                node_type=row["node_type"],
                confidence=row["confidence"],
                source=row["source"],
                created_at=row["created_at"],
                last_validated=row["last_validated"],
                validation_count=row["validation_count"],
                contradiction_count=row["contradiction_count"],
                verification_status=row.get("verification_status", "unverified"),
                metadata=json.loads(row["metadata"]),
                valid_at=row.get("valid_at"),
                invalid_at=row.get("invalid_at"),
                invalidated_by=row.get("invalidated_by"),
            )
            loaded_node_ids.add(row["id"])
            
        if loaded_node_ids:
            placeholders = ",".join(["?"] * len(loaded_node_ids))
            edge_query = f"""
                SELECT * FROM causal_edges 
                WHERE source_node IN ({placeholders}) AND target_node IN ({placeholders})
            """
            edge_params = tuple(list(loaded_node_ids) + list(loaded_node_ids))
            
            edge_rows = await self.db.fetch_all(edge_query, edge_params)
            for row in edge_rows:
                self.graph.add_edge(
                    row["source_node"],
                    row["target_node"],
                    key=row["edge_type"],
                    id=row["id"],
                    edge_type=row["edge_type"],
                    strength=row["strength"],
                    evidence=row["evidence"],
                    created_at=row["created_at"],
                )
                
        log.info(
            f"Knowledge subgraph loaded (Pragmatic Bridge): {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges."
        )
        
        self._rebuild_word_index()

    def _rebuild_word_index(self) -> None:
        """Build the inverted keyword index from all in-memory nodes."""
        self._word_index.clear()
        for node_id, data in self.graph.nodes(data=True):
            self._index_node(node_id, data["content"])

    def _index_node(self, node_id: str, content: str) -> None:
        """Add a single node's words to the inverted index."""
        for word in content.lower().strip().split():
            if len(word) > 1:  # Skip single-character words
                if word not in self._word_index:
                    self._word_index[word] = set()
                self._word_index[word].add(node_id)

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    async def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        """Add a knowledge node to the graph and database."""
        # Check for near-duplicate
        existing = await self.find_similar_node(node.content)
        if existing:
            # Reinforce existing node instead of creating duplicate
            await self._reinforce_node(existing)
            log.debug(f"Reinforced existing node: {existing[:40]}...")
            return self._get_node_model(existing)

        # Persist to SQLite
        node_type = node.node_type.value if isinstance(node.node_type, NodeType) else node.node_type
        await self.db.execute(
            """
            INSERT INTO knowledge_nodes (id, content, node_type, confidence, source,
                                         created_at, last_validated, validation_count,
                                         contradiction_count, verification_status, metadata,
                                         valid_at, invalid_at, invalidated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id, node.content, node_type, node.confidence,
                node.source, node.created_at, node.last_validated,
                node.validation_count, node.contradiction_count,
                node.verification_status.value if isinstance(node.verification_status, VerificationStatus) else node.verification_status,
                json.dumps(node.metadata),
                node.valid_at, node.invalid_at, node.invalidated_by,
            ),
        )

        # Add to in-memory graph
        self.graph.add_node(
            node.id,
            content=node.content,
            node_type=node_type,
            confidence=node.confidence,
            source=node.source,
            created_at=node.created_at,
            last_validated=node.last_validated,
            validation_count=node.validation_count,
            contradiction_count=node.contradiction_count,
            verification_status=node.verification_status.value if isinstance(node.verification_status, VerificationStatus) else node.verification_status,
            metadata=node.metadata,
            valid_at=node.valid_at,
            invalid_at=node.invalid_at,
            invalidated_by=node.invalidated_by,
        )

        # Update the inverted index
        self._index_node(node.id, node.content)

        log.debug(f"Added node: {node.content[:50]}...")
        return node

    async def find_similar_node(self, content: str, threshold: float = 0.8) -> str | None:
        """Find an existing node with very similar content using two-tier cache/DB strategy."""
        content_words = set(content.lower().strip().split())
        if not content_words:
            return None

        # TIER 1: In-Memory Check
        candidate_ids: set[str] = set()
        for word in content_words:
            if word in self._word_index:
                candidate_ids.update(self._word_index[word])

        for node_id in candidate_ids:
            if node_id not in self.graph:
                continue
            existing_words = set(self.graph.nodes[node_id]["content"].lower().strip().split())
            if not existing_words:
                continue
            overlap = content_words & existing_words
            smaller = min(len(content_words), len(existing_words))
            if smaller > 0 and len(overlap) / smaller >= threshold:
                return node_id

        # TIER 2: Database Fallback Check
        # Extract salient words (length > 4, max 5 words to keep query fast)
        salient_words = sorted([w for w in content_words if len(w) > 4], key=len, reverse=True)[:5]
        if not salient_words:
            salient_words = sorted([w for w in content_words if len(w) > 3], key=len, reverse=True)[:3]
            if not salient_words:
                return None

        conditions = " OR ".join(["content LIKE ?"] * len(salient_words))
        params = [f"%{w}%" for w in salient_words]
        
        query_sql = f"""
            SELECT id, content FROM knowledge_nodes
            WHERE {conditions}
            LIMIT 50
        """
        
        db_candidates = await self.db.fetch_all(query_sql, tuple(params))
        for row in db_candidates:
            if row["id"] in self.graph:
                continue  # Already checked in Tier 1
                
            existing_words = set(row["content"].lower().strip().split())
            if not existing_words:
                continue
                
            overlap = content_words & existing_words
            smaller = min(len(content_words), len(existing_words))
            
            if smaller > 0 and len(overlap) / smaller >= threshold:
                # Cache miss hit! Load this node into memory to repair fragmentation
                full_row = await self.db.fetch_one("SELECT * FROM knowledge_nodes WHERE id = ?", (row["id"],))
                if full_row:
                    self.graph.add_node(
                        full_row["id"],
                        content=full_row["content"],
                        node_type=full_row["node_type"],
                        confidence=full_row["confidence"],
                        source=full_row["source"],
                        created_at=full_row["created_at"],
                        last_validated=full_row["last_validated"],
                        validation_count=full_row["validation_count"],
                        contradiction_count=full_row["contradiction_count"],
                        verification_status=full_row.get("verification_status", "unverified"),
                        metadata=json.loads(full_row["metadata"]),
                    )
                    self._index_node(full_row["id"], full_row["content"])
                    log.debug(f"Tier 2 cache miss resolved for node {row['id']}")
                    
                    return row["id"]

        return None

    async def _reinforce_node(self, node_id: str) -> None:
        """Increase validation count and update timestamp for a reinforced node."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            UPDATE knowledge_nodes
            SET validation_count = validation_count + 1, last_validated = ?
            WHERE id = ?
            """,
            (now, node_id),
        )
        if node_id in self.graph:
            self.graph.nodes[node_id]["validation_count"] += 1
            self.graph.nodes[node_id]["last_validated"] = now

    async def update_confidence(self, node_id: str, new_confidence: float) -> None:
        """Update a node's confidence score."""
        await self.db.execute(
            "UPDATE knowledge_nodes SET confidence = ? WHERE id = ?",
            (new_confidence, node_id),
        )
        if node_id in self.graph:
            self.graph.nodes[node_id]["confidence"] = new_confidence

    async def increment_contradictions(self, node_id: str) -> None:
        """Increment contradiction count for a node."""
        await self.db.execute(
            """
            UPDATE knowledge_nodes
            SET contradiction_count = contradiction_count + 1
            WHERE id = ?
            """,
            (node_id,),
        )
        if node_id in self.graph:
            self.graph.nodes[node_id]["contradiction_count"] += 1

    def get_node(self, node_id: str) -> dict | None:
        """Get a node's data from the in-memory graph."""
        if node_id in self.graph:
            return {"id": node_id, **self.graph.nodes[node_id]}
        return None

    def _get_node_model(self, node_id: str) -> KnowledgeNode:
        """Convert an in-memory node to a KnowledgeNode model."""
        data = self.graph.nodes[node_id]
        return KnowledgeNode(
            id=node_id,
            content=data["content"],
            node_type=data["node_type"],
            confidence=data["confidence"],
            source=data["source"],
            created_at=data["created_at"],
            last_validated=data["last_validated"],
            validation_count=data["validation_count"],
            contradiction_count=data["contradiction_count"],
            verification_status=data.get("verification_status", "unverified"),
            metadata=data.get("metadata", {}),
            valid_at=data.get("valid_at"),
            invalid_at=data.get("invalid_at"),
            invalidated_by=data.get("invalidated_by"),
        )

    async def remove_nodes_by_source(self, source: str) -> int:
        """Remove all nodes matching a source URI from both SQLite and in-memory graph."""
        # Find node IDs to remove
        rows = await self.db.fetch_all(
            "SELECT id FROM knowledge_nodes WHERE source = ?", (source,)
        )
        node_ids = [r["id"] for r in rows]
        if not node_ids:
            return 0

        # Atomic removal from SQLite: nodes + all associated edges
        async with self.db.transaction():
            await self.db.execute(
                "DELETE FROM knowledge_nodes WHERE source = ?", (source,)
            )
            # Batch delete edges referencing any of these nodes
            placeholders = ",".join(["?"] * len(node_ids))
            await self.db.execute(
                f"DELETE FROM causal_edges WHERE source_node IN ({placeholders}) OR target_node IN ({placeholders})",
                tuple(node_ids + node_ids)
            )

        # Remove from in-memory graph + inverted index
        for nid in node_ids:
            if nid in self.graph:
                # Clean inverted index
                content = self.graph.nodes[nid].get("content", "")
                for word in content.lower().strip().split():
                    if word in self._word_index:
                        self._word_index[word].discard(nid)
                        if not self._word_index[word]:
                            del self._word_index[word]
                self.graph.remove_node(nid)

        log.debug(f"Removed {len(node_ids)} nodes with source={source}")
        return len(node_ids)

    # ------------------------------------------------------------------
    # Edge Operations
    # ------------------------------------------------------------------

    async def add_edge(self, edge: CausalEdge) -> CausalEdge:
        """Add a causal edge between two nodes."""
        # Ensure both nodes exist
        if edge.source_node not in self.graph or edge.target_node not in self.graph:
            log.warning(
                f"Cannot add edge: nodes not found "
                f"(src={edge.source_node[:8]}, tgt={edge.target_node[:8]})"
            )
            return edge

        edge_type = edge.edge_type.value if isinstance(edge.edge_type, EdgeType) else edge.edge_type

        # Same typed edge already exists — reinforce only that relationship.
        if self.graph.has_edge(edge.source_node, edge.target_node, key=edge_type):
            existing = self.graph.edges[edge.source_node, edge.target_node, edge_type]
            new_strength = min(1.0, existing.get("strength", 0.5) + 0.1)
            self.graph.edges[edge.source_node, edge.target_node, edge_type]["strength"] = new_strength
            await self.db.execute(
                """
                UPDATE causal_edges
                SET strength = ?
                WHERE source_node = ? AND target_node = ? AND edge_type = ?
                """,
                (new_strength, edge.source_node, edge.target_node, edge_type),
            )
            log.debug("Reinforced existing typed edge")
            return edge

        # Persist to SQLite
        await self.db.execute(
            """
            INSERT INTO causal_edges (id, source_node, target_node, edge_type,
                                      strength, evidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id, edge.source_node, edge.target_node,
                edge_type, edge.strength, edge.evidence, edge.created_at,
            ),
        )

        # Add to in-memory graph
        self.graph.add_edge(
            edge.source_node, edge.target_node,
            key=edge_type,
            id=edge.id,
            edge_type=edge_type,
            strength=edge.strength,
            evidence=edge.evidence,
            created_at=edge.created_at,
        )

        src_content = self.graph.nodes[edge.source_node]["content"][:30]
        tgt_content = self.graph.nodes[edge.target_node]["content"][:30]
        log.debug(f"Added edge: {src_content} --[{edge_type}]--> {tgt_content}")
        return edge

    # ------------------------------------------------------------------
    # Graph Queries
    # ------------------------------------------------------------------

    async def get_neighborhood(self, node_id: str, depth: int = 2) -> dict:
        """
        Get the causal neighborhood of a node.

        Returns all nodes and edges within `depth` hops, in both directions.
        """
        # Fetch center node content
        center_row = await self.db.fetch_one(
            "SELECT content FROM knowledge_nodes WHERE id = ?", (node_id,)
        )
        if not center_row:
            return {"center": None, "nodes": [], "edges": []}

        center_content = center_row["content"]

        # Recursive CTE to find all nearby node IDs without OR-join indexing bottleneck
        cte_sql = """
        WITH RECURSIVE neighborhood(node, depth) AS (
            SELECT ?, 0
            UNION
            SELECT e.target_node, n.depth + 1
            FROM causal_edges e
            JOIN neighborhood n ON e.source_node = n.node
            WHERE n.depth < ?
            UNION
            SELECT e.source_node, n.depth + 1
            FROM causal_edges e
            JOIN neighborhood n ON e.target_node = n.node
            WHERE n.depth < ?
        )
        SELECT DISTINCT node FROM neighborhood
        """
        rows = await self.db.fetch_all(cte_sql, (node_id, depth, depth))
        nearby_nodes = {r["node"] for r in rows}

        if not nearby_nodes:
            return {"center": center_content, "nodes": [], "edges": []}

        # Fetch node details
        placeholders = ",".join(["?"] * len(nearby_nodes))
        nodes_sql = f"""
        SELECT id, content, node_type, confidence 
        FROM knowledge_nodes 
        WHERE id IN ({placeholders})
        """
        node_rows = await self.db.fetch_all(nodes_sql, tuple(nearby_nodes))

        nodes = []
        node_content_map = {}
        for r in node_rows:
            nodes.append({
                "id": r["id"],
                "content": r["content"],
                "type": r["node_type"],
                "confidence": r["confidence"],
            })
            node_content_map[r["id"]] = r["content"]

        # Fetch edges between any of these nodes
        edges_sql = f"""
        SELECT source_node, target_node, edge_type, strength 
        FROM causal_edges 
        WHERE source_node IN ({placeholders}) AND target_node IN ({placeholders})
        """
        params = list(nearby_nodes) + list(nearby_nodes)
        edge_rows = await self.db.fetch_all(edges_sql, tuple(params))

        edges = []
        for r in edge_rows:
            edges.append({
                "from": node_content_map.get(r["source_node"], r["source_node"])[:40],
                "to": node_content_map.get(r["target_node"], r["target_node"])[:40],
                "type": r["edge_type"],
                "strength": r["strength"],
            })

        return {
            "center": center_content,
            "nodes": nodes,
            "edges": edges,
        }

    async def find_causal_chain(self, source_id: str, target_id: str) -> list[dict] | None:
        """
        Find the shortest causal path between two nodes using a SQLite recursive CTE.

        Replaces nx.shortest_path — runs directly in the DB without loading the full
        graph into memory. Returns a list of steps, or None if no path exists.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None

        # Recursive CTE BFS: finds shortest path in causal_edges table.
        # Returns all nodes on the shortest path from source to target.
        cte_sql = """
        WITH RECURSIVE path_search(node, path, depth) AS (
            -- Base case: start at source node
            SELECT ?, ?, 0
            UNION ALL
            -- Recursive case: follow outgoing edges
            SELECT e.target_node,
                   path_search.path || ',' || e.target_node,
                   path_search.depth + 1
            FROM causal_edges e
            JOIN path_search ON e.source_node = path_search.node
            WHERE path_search.depth < 8
              AND path_search.path NOT LIKE '%' || e.target_node || '%'
        )
        SELECT path FROM path_search
        WHERE node = ?
        ORDER BY depth ASC
        LIMIT 1
        """
        row = await self.db.fetch_one(cte_sql, (source_id, source_id, target_id))
        if not row:
            return None

        # Parse path string back into node id list
        path = row["path"].split(",")

        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_bundle = self.graph.get_edge_data(u, v) or {}
            edge_data = next(iter(edge_bundle.values())) if edge_bundle else {}
            u_content = self.graph.nodes[u]["content"] if u in self.graph else u
            v_content = self.graph.nodes[v]["content"] if v in self.graph else v
            steps.append({
                "from": u_content,
                "relationship": edge_data.get("edge_type", "→"),
                "to": v_content,
                "strength": edge_data.get("strength", 0.5),
            })

        return steps

    def find_node_by_content(self, query: str) -> str | None:
        """Find a node ID by partial content match."""
        query_lower = query.lower().strip()
        best_match = None
        best_overlap = 0

        for node_id, data in self.graph.nodes(data=True):
            content_lower = data["content"].lower()
            if query_lower in content_lower:
                # Exact substring match — return immediately
                return node_id
            # Word overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = node_id

        if best_overlap >= 2:
            return best_match
        return None

    def get_contradicting_nodes(self, node_id: str) -> list[dict]:
        """Find all nodes that contradict a given node."""
        results = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "contradicts":
                if u == node_id:
                    results.append({"id": v, **self.graph.nodes[v]})
                elif v == node_id:
                    results.append({"id": u, **self.graph.nodes[u]})
        return results

    # ------------------------------------------------------------------
    # Context Retrieval (replaces flat keyword search)
    # ------------------------------------------------------------------

    def _resolve_latest_valid_node(self, node_id: str) -> str:
        """Traverse the invalidated_by chain to find the latest valid superseding node ID."""
        curr_id = node_id
        visited = {curr_id}
        while curr_id in self.graph:
            data = self.graph.nodes[curr_id]
            invalidated_by = data.get("invalidated_by")
            if invalidated_by and invalidated_by in self.graph and invalidated_by not in visited:
                curr_id = invalidated_by
                visited.add(curr_id)
            else:
                break
        return curr_id

    async def retrieve_relevant_context(self, query: str, max_nodes: int = 15) -> list[dict]:
        """
        Graph-aware context retrieval using inverted index.

        Given a query, find relevant nodes via the keyword index
        and their causal neighborhoods.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        query_words = set(query.lower().split())
        if not query_words:
            return []

        # Use inverted index to find candidate nodes
        candidate_ids: set[str] = set()
        for word in query_words:
            if word in self._word_index:
                candidate_ids.update(self._word_index[word])

        # Score only candidates
        scored_nodes: list[tuple[str, float]] = []
        for node_id in candidate_ids:
            if node_id not in self.graph:
                continue
            
            # Temporal Redirection
            actual_node_id = self._resolve_latest_valid_node(node_id)
            if actual_node_id not in self.graph:
                continue

            data = self.graph.nodes[actual_node_id]
            content_words = set(data["content"].lower().split())
            if not content_words:
                continue

            overlap = len(query_words & content_words) / max(len(query_words), 1)
            confidence_bonus = data.get("confidence", 0.5) * 0.2
            degree_bonus = min(self.graph.degree(actual_node_id) * 0.05, 0.3)

            score = overlap + confidence_bonus + degree_bonus
            if score > 0.1:
                scored_nodes.append((actual_node_id, score))

        # Sort by score and take top matches
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored_nodes[:max_nodes]

        # Build rich context with relationships
        context = []
        seen_ids = set()

        for node_id, score in top_nodes:
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            data = self.graph.nodes[node_id]
            node_context = {
                "content": data["content"],
                "type": data["node_type"],
                "confidence": data["confidence"],
                "causes": [],
                "caused_by": [],
                "contradicts": [],
                "related": [],
            }

            # Dead-End Blocking Check
            is_failed = False
            failure_details = ""
            for _, target, edata in self.graph.out_edges(node_id, data=True):
                if edata.get("edge_type") == "contradicts" and target in self.graph:
                    target_data = self.graph.nodes[target]
                    if target_data.get("node_type") == "dead_end":
                        is_failed = True
                        failure_details = target_data.get("content", "")
                        break
            if not is_failed:
                for source, _, edata in self.graph.in_edges(node_id, data=True):
                    if edata.get("edge_type") == "contradicts" and source in self.graph:
                        source_data = self.graph.nodes[source]
                        if source_data.get("node_type") == "dead_end":
                            is_failed = True
                            failure_details = source_data.get("content", "")
                            break

            if is_failed:
                node_context["proven_failed"] = True
                node_context["failure_details"] = failure_details

            # Add relationship context
            for _, target, edata in self.graph.out_edges(node_id, data=True):
                edge_type = edata.get("edge_type", "related")
                target_node_id = self._resolve_latest_valid_node(target)
                if target_node_id not in self.graph:
                    continue
                target_content = self.graph.nodes[target_node_id]["content"][:60]
                if edge_type == "causes":
                    node_context["causes"].append(target_content)
                elif edge_type == "contradicts":
                    node_context["contradicts"].append(target_content)
                else:
                    node_context["related"].append(f"--[{edge_type}]--> {target_content}")

            for source, _, edata in self.graph.in_edges(node_id, data=True):
                edge_type = edata.get("edge_type", "related")
                source_node_id = self._resolve_latest_valid_node(source)
                if source_node_id not in self.graph:
                    continue
                source_content = self.graph.nodes[source_node_id]["content"][:60]
                if edge_type == "causes":
                    node_context["caused_by"].append(source_content)
                elif edge_type == "contradicts":
                    node_context["contradicts"].append(source_content)

            context.append(node_context)

        return context

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        """Get graph statistics."""
        # 1. Total nodes count and node types distribution
        node_rows = await self.db.fetch_all(
            "SELECT node_type, COUNT(*) as cnt FROM knowledge_nodes GROUP BY node_type"
        )
        node_types = {}
        total_nodes = 0
        for r in node_rows:
            nt = r["node_type"]
            cnt = r["cnt"]
            node_types[nt] = cnt
            total_nodes += cnt

        # 2. Total edges count and edge types distribution
        edge_rows = await self.db.fetch_all(
            "SELECT edge_type, COUNT(*) as cnt FROM causal_edges GROUP BY edge_type"
        )
        edge_types = {}
        total_edges = 0
        for r in edge_rows:
            et = r["edge_type"]
            cnt = r["cnt"]
            edge_types[et] = cnt
            total_edges += cnt

        # 3. Isolated nodes: nodes that have no incoming or outgoing edges
        isolated_row = await self.db.fetch_one(
            """
            SELECT COUNT(*) as cnt FROM knowledge_nodes 
            WHERE id NOT IN (
                SELECT source_node FROM causal_edges 
                UNION 
                SELECT target_node FROM causal_edges
            )
            """
        )
        isolated = isolated_row["cnt"] if isolated_row else 0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": node_types,
            "edge_types": edge_types,
            "isolated_nodes": isolated,
        }
