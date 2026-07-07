# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
EpistemicNode — a single typed vertex in the causal reasoning graph.

Node types map to logical validity of information:
  decision   — an active tool execution, command, or file edit
  hypothesis — an untested proposed change or predicted outcome
  fact       — a verified result (test pass, confirmed state, tool output)
  dead_end   — a subprocess crash, compiler error, or unhandled exception

Each node contains a SHA-256 integrity hash over its type+content+provenance+timestamp.
This makes post-hoc tampering detectable.
"""
from __future__ import annotations

import hashlib
import time
import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

NodeType = Literal["decision", "hypothesis", "fact", "dead_end"]


@dataclass
class EpistemicNode:
    """A single typed vertex in the epistemic graph."""

    node_id: str
    session_id: str
    type: NodeType
    content: str
    provenance: str
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 of all node properties including metadata."""
        meta_str = json.dumps(self.metadata, sort_keys=True)
        run_id_val = self.run_id or ""
        payload = f"{self.node_id}|{self.session_id}|{run_id_val}|{self.type}|{self.content}|{self.provenance}|{self.timestamp}|{meta_str}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """True if stored hash matches current content (tamper detection)."""
        return self.integrity_hash == self._compute_hash()

    @staticmethod
    def new(
        session_id: str,
        type: NodeType,
        content: str,
        provenance: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EpistemicNode":
        """Factory: create a new node with auto-generated UUID."""
        return EpistemicNode(
            node_id=str(uuid.uuid4()),
            session_id=session_id,
            type=type,
            content=content,
            provenance=provenance,
            run_id=run_id,
            metadata=metadata or {},
        )
