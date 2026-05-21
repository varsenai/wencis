"""
Session Manager — ARIA's continuity tracker.

Manages conversation sessions, turn history, and aggregate stats.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import datetime, timezone

from silex.models.schemas import Session, Turn
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.session")

current_session_var: ContextVar[Session | None] = ContextVar("current_session_var", default=None)


class SessionManager:
    """Tracks conversation sessions and turn history."""

    def __init__(self, db: Database):
        self.db = db

    @property
    def current(self) -> Session | None:
        return current_session_var.get()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(self) -> Session:
        """Begin a new conversation session."""
        session = Session()
        await self.db.execute(
            """
            INSERT INTO sessions (id, started_at, turn_count, memories_created,
                                  goals_modified, avg_confidence, topics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.started_at,
                session.turn_count,
                session.memories_created,
                session.goals_modified,
                session.avg_confidence,
                json.dumps(session.topics),
            ),
        )
        current_session_var.set(session)
        log.info(f"Started session {session.id[:8]}...")
        return session

    async def resume_or_start(self) -> Session:
        """
        Attempt to resume the most recent unclosed session.
        Falls back to creating a new session if none exists.
        This prevents amnesia on server restart.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
        )
        if row:
            session = self._row_to_session(row)
            current_session_var.set(session)
            log.info(f"Resumed session {session.id[:8]}... ({session.turn_count} prior turns)")
            return session
        return await self.start_session()

    async def resume_specific(self, session_id: str) -> Session | None:
        """
        Resume a specific session by ID (used when the web client sends its last known session).
        Returns None if the session_id doesn't exist in the DB.
        """
        row = await self.db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        if row:
            session = self._row_to_session(row)
            # Clear the ended_at so the session is considered active again
            await self.db.execute(
                "UPDATE sessions SET ended_at = NULL WHERE id = ?",
                (session_id,)
            )
            current_session_var.set(session)
            log.info(f"Reconnected to session {session.id[:8]}... ({session.turn_count} prior turns)")
            return session
        return None

    async def end_session(self) -> None:
        """End the current session."""
        session = self.current
        if session is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (now, session.id),
        )
        log.info(f"Ended session {session.id[:8]}...")
        current_session_var.set(None)

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    async def record_turn(
        self,
        user_input: str,
        reasoning: str,
        response: str,
        self_reflection: str,
        confidence: float,
        memories_added: int = 0,
        goals_changed: int = 0,
        scratchpad: str | None = None,
    ) -> Turn:
        """Record a conversation turn and update session stats."""
        session = self.current
        if session is None:
            raise RuntimeError("No active session. Call start_session() or ensure current_session_var is set first.")

        # Compute new values before mutating in-memory state
        new_turn_count = session.turn_count + 1
        new_memories_created = session.memories_created + memories_added
        new_goals_modified = session.goals_modified + goals_changed
        new_avg_confidence = session.avg_confidence + (confidence - session.avg_confidence) / new_turn_count

        turn = Turn(
            session_id=session.id,
            turn_number=new_turn_count,
            user_input=user_input,
            reasoning=reasoning,
            response=response,
            self_reflection=self_reflection,
            confidence=confidence,
            scratchpad=scratchpad,
        )

        # Atomic: store turn + update session stats
        async with self.db.transaction():
            await self.db.execute(
                """
                INSERT INTO turns (id, session_id, turn_number, user_input,
                                   reasoning, response, self_reflection, confidence, scratchpad, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.id,
                    turn.session_id,
                    turn.turn_number,
                    turn.user_input,
                    turn.reasoning,
                    turn.response,
                    turn.self_reflection,
                    turn.confidence,
                    turn.scratchpad,
                    turn.created_at,
                ),
            )

            await self.db.execute(
                """
                UPDATE sessions
                SET turn_count = ?, memories_created = ?, goals_modified = ?, avg_confidence = ?
                WHERE id = ?
                """,
                (
                    new_turn_count,
                    new_memories_created,
                    new_goals_modified,
                    new_avg_confidence,
                    session.id,
                ),
            )

        # Only mutate in-memory state after DB success
        session.turn_count = new_turn_count
        session.memories_created = new_memories_created
        session.goals_modified = new_goals_modified
        session.avg_confidence = new_avg_confidence

        return turn

    async def get_recent_turns(self, limit: int = 10) -> list[Turn]:
        """Get the most recent turns from the current session."""
        session = self.current
        if session is None:
            return []

        rows = await self.db.fetch_all(
            """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
            """,
            (session.id, limit),
        )

        turns = [self._row_to_turn(r) for r in rows]
        turns.reverse()  # Chronological order
        return turns

    async def compress_turns(self, session_id: str, old_turn_ids: list[str], new_virtual_turn: Turn) -> None:
        """Replace old raw turns with a compressed virtual turn."""
        async with self.db.transaction():
            for turn_id in old_turn_ids:
                await self.db.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
            
            await self.db.execute(
                """
                INSERT INTO turns (id, session_id, turn_number, user_input,
                                   reasoning, response, self_reflection, confidence, scratchpad, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_virtual_turn.id,
                    new_virtual_turn.session_id,
                    new_virtual_turn.turn_number,
                    new_virtual_turn.user_input,
                    new_virtual_turn.reasoning,
                    new_virtual_turn.response,
                    new_virtual_turn.self_reflection,
                    new_virtual_turn.confidence,
                    new_virtual_turn.scratchpad,
                    new_virtual_turn.created_at,
                ),
            )

    async def get_last_reflection(self) -> str | None:
        """Get the self_reflection from the most recent turn in the current session."""
        session = self.current
        if session is None:
            return None
        row = await self.db.fetch_one(
            """
            SELECT self_reflection FROM turns
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT 1
            """,
            (session.id,),
        )
        if row and row["self_reflection"]:
            return row["self_reflection"]
        return None

    async def get_recent_failures(self, limit: int = 3) -> list[dict]:
        """Fetch the most recent failures from the current session."""
        session = self.current
        if session is None:
            return []
        rows = await self.db.fetch_all(
            """
            SELECT failure_type, description, created_at FROM recent_failures
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session.id, limit),
        )
        return [dict(r) for r in rows]

    async def get_all_sessions(self) -> list[Session]:
        """Get all past sessions."""
        rows = await self.db.fetch_all(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        )
        return [self._row_to_session(r) for r in rows]

    async def get_session(self, session_id: str) -> Session | None:
        """Get a specific session by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        return self._row_to_session(row) if row else None

    async def get_turns_for_session(self, session_id: str) -> list[Turn]:
        """Get all turns for a specific session."""
        rows = await self.db.fetch_all(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number ASC",
            (session_id,)
        )
        return [self._row_to_turn(r) for r in rows]

    async def get_total_turns(self) -> int:
        """Get total turns across all sessions."""
        row = await self.db.fetch_one("SELECT COUNT(*) as cnt FROM turns")
        return row["cnt"] if row else 0

    async def get_time_since_last_user_message(self) -> float:
        """Calculate and return the elapsed time in seconds since the last real user message."""
        from datetime import datetime, timezone
        row = await self.db.fetch_one(
            """
            SELECT created_at FROM turns
            WHERE user_input NOT LIKE '[SYSTEM:%'
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        if not row:
            session = self.current
            if session:
                try:
                    started_at = datetime.fromisoformat(session.started_at.replace("Z", "+00:00"))
                except ValueError:
                    started_at = datetime.now(timezone.utc)
                return (datetime.now(timezone.utc) - started_at).total_seconds()
            return 0.0

        try:
            last_time = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        return (datetime.now(timezone.utc) - last_time).total_seconds()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    async def get_cached_response(self, user_input: str, ttl_seconds: int = 900) -> str | None:
        """Retrieve a cached response if it exists and hasn't expired."""
        import hashlib
        from datetime import datetime, timezone

        query_hash = hashlib.sha256(user_input.strip().lower().encode("utf-8")).hexdigest()
        row = await self.db.fetch_one(
            "SELECT response, created_at FROM response_cache WHERE query_hash = ?",
            (query_hash,)
        )
        if not row:
            return None

        try:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - created_at).total_seconds()
            if elapsed < ttl_seconds:
                log.info(f"Response cache HIT for hash: {query_hash[:8]}... (Age: {elapsed:.1f}s)")
                return row["response"]
            else:
                log.debug(f"Response cache hit but expired for hash: {query_hash[:8]}... (Age: {elapsed:.1f}s)")
        except ValueError as e:
            log.warning(f"Error parsing cache created_at timestamp: {e}")

        return None

    async def store_cached_response(self, user_input: str, response: str) -> None:
        """Store a response in the cache for future queries."""
        import hashlib
        from datetime import datetime, timezone

        query_hash = hashlib.sha256(user_input.strip().lower().encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT OR REPLACE INTO response_cache (query_hash, response, created_at)
            VALUES (?, ?, ?)
            """,
            (query_hash, response, now)
        )
        log.info(f"Stored response in cache for hash: {query_hash[:8]}...")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_turn(row: dict) -> Turn:
        return Turn(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            user_input=row["user_input"],
            reasoning=row["reasoning"],
            response=row["response"],
            self_reflection=row["self_reflection"],
            confidence=row["confidence"],
            scratchpad=row.get("scratchpad"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_session(row: dict) -> Session:
        return Session(
            id=row["id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            turn_count=row["turn_count"],
            memories_created=row["memories_created"],
            goals_modified=row["goals_modified"],
            avg_confidence=row["avg_confidence"],
            topics=json.loads(row["topics"]),
        )
