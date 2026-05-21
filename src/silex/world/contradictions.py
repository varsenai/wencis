"""
Contradiction Detector — finds and resolves conflicts in ARIA's knowledge.

When new information conflicts with existing beliefs:
  1. Detect — compare against related nodes
  2. Flag — create a 'contradicts' edge
  3. Reason — Gemini evaluates which is more likely
  4. Resolve — update confidence scores, the losing belief gets downgraded

Contradictions are never deleted — ARIA should know what it used to believe
and why it changed its mind.
"""

from __future__ import annotations

from datetime import datetime, timezone

from silex.models.schemas import (
    CausalEdge,
    Contradiction,
    EdgeType,
    KnowledgeNode,
    NodeType,
    StoredContradiction,
    VerificationStatus,
)
from silex.storage.database import Database
from silex.world.graph import KnowledgeGraph
from silex.utils.logger import setup_logger

log = setup_logger("silex.world.contradictions")


class ContradictionDetector:
    """Detects and manages contradictions in ARIA's world model."""

    def __init__(self, db: Database, graph: KnowledgeGraph):
        self.db = db
        self.graph = graph

    async def process_contradiction(self, contradiction: Contradiction) -> StoredContradiction | None:
        """
        Process a contradiction detected by Gemini.

        1. Find or create nodes for both claims
        2. Create a 'contradicts' edge
        3. Store the analysis
        4. Update confidence scores
        """
        # Find existing nodes for both claims
        node_a_id = self.graph.find_node_by_content(contradiction.existing_claim)
        node_b_id = self.graph.find_node_by_content(contradiction.new_claim)

        async with self.db.transaction():
            if not node_a_id:
                node_a = await self.graph.add_node(KnowledgeNode(
                    content=contradiction.existing_claim,
                    node_type=NodeType.HYPOTHESIS,
                    confidence=0.4,
                    source="contradiction",
                    verification_status=VerificationStatus.CONTRADICTED,
                    metadata={"provenance": "contradiction_detector"},
                ))
                node_a_id = node_a.id

            if not node_b_id:
                node_b = await self.graph.add_node(KnowledgeNode(
                    content=contradiction.new_claim,
                    node_type=NodeType.HYPOTHESIS,
                    confidence=0.4,
                    source="contradiction",
                    verification_status=VerificationStatus.CONTRADICTED,
                    metadata={"provenance": "contradiction_detector"},
                ))
                node_b_id = node_b.id

            # Create the contradiction record
            stored = StoredContradiction(
                node_a=node_a_id,
                node_b=node_b_id,
                analysis=contradiction.analysis,
                status="unresolved",
            )

            await self.db.execute(
                """
                INSERT INTO contradictions (id, node_a, node_b, analysis, status,
                                            resolution, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored.id, stored.node_a, stored.node_b,
                    stored.analysis, stored.status, stored.resolution,
                    stored.created_at, stored.resolved_at,
                ),
            )

            # Add contradicts edge if both nodes exist in the graph
            if node_a_id and node_b_id:
                edge = CausalEdge(
                    source_node=node_a_id,
                    target_node=node_b_id,
                    edge_type=EdgeType.CONTRADICTS,
                    strength=contradiction.confidence,
                    evidence=contradiction.analysis,
                )
                await self.graph.add_edge(edge)

                # Update contradiction counts
                await self.graph.increment_contradictions(node_a_id)
                await self.graph.increment_contradictions(node_b_id)

                # Adjust confidence — the resolution determines who "wins"
                if contradiction.confidence > 0.6:
                    # New claim wins — downgrade existing
                    if node_a_id:
                        old_conf = self.graph.graph.nodes[node_a_id].get("confidence", 0.5)
                        await self.graph.update_confidence(
                            node_a_id, max(0.1, old_conf - 0.2)
                        )
                else:
                    # Existing wins — downgrade new
                    if node_b_id:
                        old_conf = self.graph.graph.nodes[node_b_id].get("confidence", 0.5)
                        await self.graph.update_confidence(
                            node_b_id, max(0.1, old_conf - 0.2)
                        )

        log.info(
            f"Contradiction recorded: '{contradiction.existing_claim[:30]}' "
            f"vs '{contradiction.new_claim[:30]}'"
        )
        return stored

    async def get_unresolved(self) -> list[StoredContradiction]:
        """Get all unresolved contradictions."""
        rows = await self.db.fetch_all(
            "SELECT * FROM contradictions WHERE status = 'unresolved' ORDER BY created_at DESC"
        )
        return [self._row_to_contradiction(r) for r in rows]

    async def get_all(self) -> list[StoredContradiction]:
        """Get all contradictions."""
        rows = await self.db.fetch_all(
            "SELECT * FROM contradictions ORDER BY created_at DESC"
        )
        return [self._row_to_contradiction(r) for r in rows]

    async def resolve(self, contradiction_id: str, resolution: str) -> None:
        """Mark a contradiction as resolved."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            UPDATE contradictions
            SET status = 'resolved', resolution = ?, resolved_at = ?
            WHERE id = ?
            """,
            (resolution, now, contradiction_id),
        )

    async def count_unresolved(self) -> int:
        """Count unresolved contradictions."""
        row = await self.db.fetch_one(
            "SELECT COUNT(*) as cnt FROM contradictions WHERE status = 'unresolved'"
        )
        return row["cnt"] if row else 0

    @staticmethod
    def _row_to_contradiction(row: dict) -> StoredContradiction:
        return StoredContradiction(
            id=row["id"],
            node_a=row["node_a"],
            node_b=row["node_b"],
            analysis=row["analysis"],
            status=row["status"],
            resolution=row["resolution"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )
