# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
Wencis — Epistemic Reasoning Engine for AI Agents.

Public API surface. Everything not exported here is internal.
"""

from wencis.graph.nodes import EpistemicNode, NodeType
from wencis.graph.edges import CausalEdge, EdgeType
from wencis.graph.causal_graph import CausalKnowledgeGraph

from wencis.optimizer.schemas import (
    TrajectoryRevisionResult,
    TrajectoryRefinementResult,
    StepRevision,
    StepRefinement,
)
from wencis.optimizer.trajectory_optimizer import TrajectoryOptimizer

from wencis.critic.schemas import CritiqueScore, CritiqueResponse
from wencis.critic.response_critic import ResponseCritic

from wencis.meta.schemas import (
    FailureClusterReport,
    MetaAnalysisResponse,
    SelfImprovementProposal,
)
from wencis.meta.drift import LocalAlignmentVerifier
from wencis.meta.meta_reasoning import MetaReasoningEngine

from wencis.storage.protocol import StorageBackend
from wencis.storage.sqlite_backend import SQLiteBackend
from wencis.llm.protocol import LLMClient

__version__ = "0.1.0"
__all__ = [
    # Graph
    "EpistemicNode",
    "NodeType",
    "CausalEdge",
    "EdgeType",
    "CausalKnowledgeGraph",
    # Optimizer
    "TrajectoryOptimizer",
    "TrajectoryRevisionResult",
    "TrajectoryRefinementResult",
    # Critic
    "ResponseCritic",
    "CritiqueScore",
    "CritiqueResponse",
    # Meta
    "MetaReasoningEngine",
    "FailureClusterReport",
    "SelfImprovementProposal",
    "LocalAlignmentVerifier",
    # Infrastructure
    "StorageBackend",
    "SQLiteBackend",
    "LLMClient",
]
