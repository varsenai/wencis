# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
CausalEdge — a directed typed relationship between two epistemic nodes.

Edge types define the semantic meaning of the link:
  triggered_by      — this node was caused by a parent decision
  contradicts       — a verified fact invalidates a hypothesis
  prevented         — an action successfully bypassed a dead_end
  caused_failure_in — a decision is the root cause of a dead_end
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

EdgeType = Literal["triggered_by", "contradicts", "prevented", "caused_failure_in"]


@dataclass
class CausalEdge:
    """A directed typed relationship between two epistemic nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: EdgeType
    weight: float = 1.0

    @staticmethod
    def new(
        source_node_id: str,
        target_node_id: str,
        relation_type: EdgeType,
        weight: float = 1.0,
    ) -> "CausalEdge":
        """Factory: create a new edge with auto-generated UUID."""
        return CausalEdge(
            edge_id=str(uuid.uuid4()),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            weight=weight,
        )
