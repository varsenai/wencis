"""
Memory Store — ARIA's persistent knowledge.

Handles storing, retrieving, searching, and managing memories in SQLite.
The retrieval strategy uses three pools: recency, importance, and relevance.

Polish additions:
  - Duplicate detection before storing
  - Memory deletion (forget)
  - Memory search command
  - Manual memory injection
  - Importance decay over time
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from math import exp

from silex.models.schemas import Memory, MemorySource, MemoryType
from silex.storage.database import Database
from silex.utils.config import (
    MAX_IMPORTANT_MEMORIES,
    MAX_RECENT_MEMORIES,
    MAX_RELEVANT_MEMORIES,
    MEMORY_HALFLIFE_DAYS,
)
from silex.memory.vector_store import VectorStore
from silex.utils.logger import setup_logger

log = setup_logger("silex.memory")


class MemoryStore:
    """SQLite-backed persistent memory for ARIA."""

    def __init__(self, db: Database):
        self.db = db
        self.vs = VectorStore(collection_name="aria_memories")
        self._unconsolidated_count = 0
        self._needs_consolidation = False
        self._consolidation_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def add(self, memory: Memory) -> Memory | None:
        """Store a new memory (with duplicate detection)."""
        # Check for duplicates — skip if a very similar memory exists
        if await self._is_duplicate(memory.content):
            log.debug(f"Skipped duplicate memory: {memory.content[:40]}...")
            return None

        await self._persist_memory(memory)
        log.debug(f"Stored memory: {memory.content[:60]}...")
        return memory

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        if row is None:
            return None
        return self._row_to_memory(row)

    async def get_by_index(self, index: int) -> Memory | None:
        """Get a memory by its display index (1-based, sorted by importance)."""
        rows = await self.db.fetch_all(
            "SELECT * FROM memories WHERE archived_at IS NULL AND superseded_by_id IS NULL ORDER BY importance DESC"
        )
        if 1 <= index <= len(rows):
            return self._row_to_memory(rows[index - 1])
        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        row = await self.db.fetch_one(
            "SELECT content FROM memories WHERE id = ?", (memory_id,)
        )
        if row is None:
            return False
        await self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        
        if self.vs.is_active:
            await asyncio.to_thread(self.vs.delete_by_ids, [memory_id])
            
        log.info(f"Deleted memory: {row['content'][:40]}...")
        return True

    async def delete_by_index(self, index: int) -> bool:
        """Delete a memory by its display index (1-based)."""
        memory = await self.get_by_index(index)
        if memory:
            return await self.delete(memory.id)
        return False

    async def update_access(self, memory_id: str) -> None:
        """Mark a memory as accessed (updates timestamp and counter)."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            UPDATE memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            (now, memory_id),
        )

    async def count(self) -> int:
        """Get total memory count."""
        row = await self.db.fetch_one("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"] if row else 0

    @property
    def unconsolidated_count(self) -> int:
        return self._unconsolidated_count

    async def check_consolidation_trigger(self) -> bool:
        """Check if consolidation is required, resetting the flag if it was set."""
        async with self._consolidation_lock:
            if self._needs_consolidation:
                self._needs_consolidation = False
                return True
            return False

    async def all_memories(self) -> list[Memory]:
        """Retrieve all memories (use sparingly)."""
        rows = await self.db.fetch_all(
            "SELECT * FROM memories ORDER BY importance DESC"
        )
        return [self._row_to_memory(r) for r in rows]

    async def search(self, query: str) -> list[Memory]:
        """Search memories by keyword (for the :search command)."""
        return await self._search_relevant(query, limit=50)

    async def add_manual(
        self, content: str, importance: float = 0.7, level: int = 1, child_memory_ids: list[str] | None = None
    ) -> Memory:
        """Add a memory manually from user command."""
        memory = Memory(
            content=content,
            source=MemorySource.USER,
            importance=importance,
            tags=["manual"],
            level=level,
            child_memory_ids=child_memory_ids or [],
        )
        # Bypass duplicate check for manual memories — user explicitly wants it
        await self._persist_memory(memory)
        log.info(f"Manual memory stored: {content[:40]}...")
        return memory

    async def _persist_memory(self, memory: Memory) -> None:
        """Shared persistence logic for add() and add_manual() — single source of truth."""
        await self.db.execute(
            """
            INSERT INTO memories (id, content, source, memory_type, importance,
                                  confidence, created_at, last_accessed,
                                  access_count, tags, level, child_memory_ids, provenance_json,
                                  related_memories, superseded_by_id, archived_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.source.value if isinstance(memory.source, MemorySource) else memory.source,
                memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
                memory.importance,
                memory.confidence,
                memory.created_at,
                memory.last_accessed,
                memory.access_count,
                json.dumps(memory.tags),
                memory.level,
                json.dumps(memory.child_memory_ids),
                json.dumps(memory.provenance),
                json.dumps(memory.related_memories),
                memory.superseded_by_id,
                memory.archived_at,
            ),
        )
        if self.vs.is_active:
            type_val = memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type
            await asyncio.to_thread(self.vs.add_chunks, [memory.content], [{"type": type_val, "timestamp": datetime.now(timezone.utc).timestamp()}], ids=[memory.id])

        if memory.level == 1:
            async with self._consolidation_lock:
                self._unconsolidated_count += 1
                if self._unconsolidated_count >= 50:
                    self._unconsolidated_count = 0
                    self._needs_consolidation = True

    # ------------------------------------------------------------------
    # Retrieval Strategy
    # ------------------------------------------------------------------

    async def retrieve_context(self, query: str = "") -> list[Memory]:
        """
        Retrieve memories for context injection using the three-pool strategy:
          1. Recent — last N accessed memories (short-term recall)
          2. Important — top N by importance (core knowledge)
          3. Relevant — keyword match against query (situational)

        Results are deduplicated and sorted by a fused trust/relevance score.
        """
        candidates: dict[str, Memory] = {}

        # Pool 1: Recent
        recent = await self._get_recent(MAX_RECENT_MEMORIES)
        for m in recent:
            candidates[m.id] = m

        # Pool 2: Important
        important = await self._get_important(MAX_IMPORTANT_MEMORIES)
        for m in important:
            candidates[m.id] = m

        # Pool 3: Relevant (keyword search)
        if query.strip():
            relevant = await self._search_relevant(query, MAX_RELEVANT_MEMORIES)
            for m in relevant:
                candidates[m.id] = m

        # Pool 4: Semantic (vector search)
        if query.strip() and self.vs.is_active:
            semantic_results = await asyncio.to_thread(self.vs.search, query, MAX_RELEVANT_MEMORIES)
            semantic_ids = [res["id"] for res in semantic_results if res.get("id")]
            if semantic_ids:
                # Build the orig_id → resolved_id mapping ONCE (avoids O(n×m) re-resolution)
                id_resolution_map: dict[str, str] = {}
                resolved_ids = []
                for s_id in semantic_ids:
                    latest_id = await self._resolve_latest_memory_id(s_id)
                    id_resolution_map[s_id] = latest_id
                    resolved_ids.append(latest_id)
                unique_resolved_ids = list(dict.fromkeys(resolved_ids))
                placeholders = ",".join("?" * len(unique_resolved_ids))
                rows = await self.db.fetch_all(
                    f"SELECT * FROM memories WHERE id IN ({placeholders}) AND archived_at IS NULL AND superseded_by_id IS NULL",
                    tuple(unique_resolved_ids)
                )
                now_ts = datetime.now(timezone.utc).timestamp()
                for row in rows:
                    # Find the matching semantic result using the pre-built map
                    res = next((r for r in semantic_results if r["id"] == row["id"]), None)
                    if not res:
                        # Check via resolution map (O(n) lookup instead of O(n×m) DB queries)
                        for orig_id, resolved_id in id_resolution_map.items():
                            if resolved_id == row["id"]:
                                res = next((r for r in semantic_results if r["id"] == orig_id), None)
                                if res:
                                    break
                    if res:
                        created_at = datetime.fromisoformat(row["created_at"])
                        age_days = (now_ts - created_at.timestamp()) / 86400.0
                        adjusted_score = (1.0 - res.get("distance", 1.0)) * exp(-age_days / MEMORY_HALFLIFE_DAYS)
                        
                        # Only keep memories that meet the decayed relevance threshold
                        if adjusted_score > 0.1:
                            m = self._row_to_memory(row)
                            candidates[m.id] = m

        result = sorted(
            candidates.values(),
            key=lambda m: self._retrieval_score(m, query),
            reverse=True,
        )

        # Batch update access timestamps in a single transaction
        if result:
            async with self.db.transaction():
                for m in result:
                    await self.update_access(m.id)

        log.debug(f"Retrieved {len(result)} memories for context")
        return result

    async def _get_recent(self, limit: int) -> list[Memory]:
        """Get most recently accessed memories."""
        rows = await self.db.fetch_all(
            "SELECT * FROM memories WHERE archived_at IS NULL AND superseded_by_id IS NULL ORDER BY last_accessed DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_memory(r) for r in rows]

    async def _get_important(self, limit: int) -> list[Memory]:
        """Get highest importance memories."""
        rows = await self.db.fetch_all(
            "SELECT * FROM memories WHERE archived_at IS NULL AND superseded_by_id IS NULL ORDER BY importance DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_memory(r) for r in rows]

    async def _search_relevant(self, query: str, limit: int) -> list[Memory]:
        """
        Full-text keyword relevance search using FTS5 (Phase 19 upgrade).

        Uses SQLite FTS5 MATCH with BM25 ranking for O(1) keyword lookup
        instead of the previous O(n) LIKE scan. Falls back to LIKE if
        FTS5 is unavailable (pre-migration databases).
        """
        keywords = [kw.strip().lower() for kw in query.split() if len(kw.strip()) > 2]
        if not keywords:
            return []

        # ── FTS5 path (fast, O(log n)) ─────────────────────────────
        try:
            # Build FTS5 MATCH expression: each keyword joined with OR
            fts_query = " OR ".join(f'"{kw}"' for kw in keywords)

            rows = await self.db.fetch_all(
                """
                SELECT m.*
                FROM memories m
                JOIN memories_fts fts ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.archived_at IS NULL
                  AND m.superseded_by_id IS NULL
                ORDER BY bm25(memories_fts) ASC
                LIMIT ?
                """,
                (fts_query, limit),
            )
            if rows:
                return [self._row_to_memory(r) for r in rows]
        except Exception as e:
            log.debug(f"FTS5 search failed, falling back to LIKE: {e}")

        # ── LIKE fallback (slow, O(n)) ─────────────────────────────
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in keywords])
        params = tuple(f"%{kw}%" for kw in keywords)

        rows = await self.db.fetch_all(
            f"SELECT * FROM memories WHERE archived_at IS NULL AND superseded_by_id IS NULL AND ({conditions}) ORDER BY importance DESC LIMIT ?",
            (*params, limit),
        )
        return [self._row_to_memory(r) for r in rows]

    @staticmethod
    def _retrieval_score(memory: Memory, query: str) -> float:
        """Fuse importance, reliability, recency, relevance, and source trust."""
        query_words = {w.lower() for w in query.split() if len(w) > 2}
        content_words = {w.lower() for w in memory.content.split() if len(w) > 2}
        relevance = 0.0
        if query_words:
            relevance = len(query_words & content_words) / max(len(query_words), 1)

        try:
            last_accessed = datetime.fromisoformat(memory.last_accessed)
            age_days = max((datetime.now(timezone.utc) - last_accessed).days, 0)
            recency = exp(-age_days / MEMORY_HALFLIFE_DAYS)
        except Exception:
            recency = 0.5

        source_trust = {
            MemorySource.USER: 0.9,
            MemorySource.SYSTEM: 0.85,
            MemorySource.REFLECTION: 0.65,
            MemorySource.INFERENCE: 0.55,
        }.get(memory.source, 0.5)

        type_bonus = {
            MemoryType.PREFERENCE: 0.08,
            MemoryType.PROCEDURAL: 0.06,
            MemoryType.PROJECT: 0.06,
            MemoryType.NORMATIVE: 0.10,
            MemoryType.CHARACTER: 0.09,
        }.get(memory.memory_type, 0.0)

        return (
            memory.importance * 0.35
            + memory.confidence * 0.20
            + relevance * 0.25
            + recency * 0.10
            + source_trust * 0.10
            + type_bonus
        )

    # ------------------------------------------------------------------
    # Duplicate Detection
    # ------------------------------------------------------------------

    async def _is_duplicate(self, content: str) -> bool:
        """
        Check if a very similar memory already exists.
        
        Uses vector semantic similarity to catch rephrased facts.
        Falls back to word overlap if VectorStore is offline.
        """
        if self.vs.is_active:
            results = await asyncio.to_thread(self.vs.search, content, 1)
            # Distance < 0.2 typically indicates semantic equivalence with MiniLM
            if results and results[0].get("distance", 1.0) < 0.2:
                return True

        content_lower = content.lower().strip()
        content_words = set(content_lower.split())

        if not content_words:
            return False

        # Fallback: Check against recent memories
        recent = await self._get_recent(50)
        for mem in recent:
            existing_words = set(mem.content.lower().strip().split())
            if not existing_words:
                continue

            # Calculate word overlap
            overlap = content_words & existing_words
            smaller = min(len(content_words), len(existing_words))

            if smaller > 0 and len(overlap) / smaller >= 0.8:
                return True

        return False

    async def archive(self, memory_id: str) -> bool:
        """Soft-archive a memory without destroying provenance."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE memories SET archived_at = ? WHERE id = ?",
            (now, memory_id),
        )
        return True

    async def update_confidence(self, memory_id: str, confidence: float) -> bool:
        """Adjust memory confidence for correction workflows."""
        confidence = max(0.0, min(1.0, confidence))
        await self.db.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (confidence, memory_id),
        )
        return True

    async def _resolve_latest_memory_id(self, memory_id: str) -> str:
        """Follow the superseded_by_id chain to resolve to the latest active memory ID."""
        curr_id = memory_id
        visited = {curr_id}
        while True:
            row = await self.db.fetch_one("SELECT superseded_by_id FROM memories WHERE id = ?", (curr_id,))
            if row and row["superseded_by_id"]:
                next_id = row["superseded_by_id"]
                if next_id in visited:
                    break
                visited.add(next_id)
                curr_id = next_id
            else:
                break
        return curr_id

    async def _resolve_latest_memory(self, memory_id: str) -> Memory | None:
        """Resolve latest memory object by following superseded chains."""
        latest_id = await self._resolve_latest_memory_id(memory_id)
        return await self.get(latest_id)

    async def merge(self, keep_id: str, merge_id: str) -> bool:
        """Merge two memories by archiving the duplicate and linking superseded relation."""
        keep = await self.get(keep_id)
        duplicate = await self.get(merge_id)
        if not keep or not duplicate:
            return False
        related = set(keep.related_memories)
        related.add(merge_id)
        provenance = dict(keep.provenance)
        provenance.setdefault("merged_memory_ids", [])
        provenance["merged_memory_ids"].append(merge_id)
        async with self.db.transaction():
            await self.db.execute(
                "UPDATE memories SET related_memories = ?, provenance_json = ? WHERE id = ?",
                (json.dumps(sorted(related)), json.dumps(provenance), keep_id),
            )
            now = datetime.now(timezone.utc).isoformat()
            await self.db.execute(
                "UPDATE memories SET archived_at = ?, superseded_by_id = ? WHERE id = ?",
                (now, keep_id, merge_id),
            )
        return True

    async def decay_importance(self, days: int = 7, decay_factor: float = 0.95):
        """Multiplies importance by decay_factor for memories not accessed in the last `days`."""
        await self.db.execute(
            """
            UPDATE memories
            SET importance = importance * ?
            WHERE (julianday('now') - julianday(last_accessed)) > ?
              AND archived_at IS NULL
            """,
            (decay_factor, days)
        )
        log.info(f"Decayed importance of memories untouched in {days} days by factor {decay_factor}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        """Convert a database row to a Memory model."""
        return Memory(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            memory_type=row.get("memory_type", "semantic"),
            importance=row["importance"],
            confidence=row.get("confidence", 0.5),
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            tags=json.loads(row["tags"]),
            level=row.get("level", 1),
            child_memory_ids=json.loads(row.get("child_memory_ids", "[]")),
            provenance=json.loads(row.get("provenance_json", "{}")),
            related_memories=json.loads(row["related_memories"]),
            archived_at=row.get("archived_at"),
            superseded_by_id=row.get("superseded_by_id"),
        )
    # ------------------------------------------------------------------
    # Semantic Profiles (Phase 7)
    # ------------------------------------------------------------------

    async def get_semantic_profile(self, term: str) -> dict | None:
        """Retrieve the objective mapping for a subjective term."""
        row = await self.db.fetch_one(
            "SELECT * FROM semantic_profiles WHERE term = ?", (term.lower(),)
        )
        if row:
            return {
                "term": row["term"],
                "objective_proxies": json.loads(row["objective_proxies"]),
                "context_tags": json.loads(row["context_tags"]),
                "confidence": row["confidence"],
                "updated_at": row["updated_at"]
            }
        return None

    async def save_semantic_profile(self, term: str, objective_proxies: list[str], confidence: float = 0.5):
        """Save or update a semantic profile."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO semantic_profiles (term, objective_proxies, confidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                objective_proxies = excluded.objective_proxies,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (term.lower(), json.dumps(objective_proxies), confidence, now)
        )

    async def get_all_semantic_profiles(self) -> dict[str, list[str]]:
        """Retrieve all learned semantic mappings."""
        rows = await self.db.fetch_all("SELECT term, objective_proxies FROM semantic_profiles")
        return {row["term"]: json.loads(row["objective_proxies"]) for row in rows}
