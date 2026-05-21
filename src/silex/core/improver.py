"""
Improver — Tracks self-corrections and calibration (Phase 3).

Logs when ARIA rejects her own response, the critique feedback,
and the final improved response to SQLite.
"""

from __future__ import annotations

from silex.models.schemas import CognitiveResponse, CritiqueResponse, ImprovementLogEntry
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.improver")

class ImprovementLogger:
    """Logs self-improvement cycles to the database."""

    def __init__(self, db: Database):
        self.db = db

    async def log_improvement(
        self,
        session_id: str,
        turn_number: int,
        draft: CognitiveResponse,
        critique: CritiqueResponse,
        final: CognitiveResponse,
    ) -> None:
        """Record a successful self-correction cycle."""
        entry = ImprovementLogEntry(
            session_id=session_id,
            turn_number=turn_number,
            original_response=draft.response,
            feedback=critique.feedback,
            accuracy_score=critique.scores.accuracy,
            depth_score=critique.scores.depth,
            honesty_score=critique.scores.honesty,
            improved_response=final.response,
        )

        await self.db.execute(
            """
            INSERT INTO improvement_logs (
                id, session_id, turn_number, original_response, feedback,
                accuracy_score, depth_score, honesty_score,
                improved_response, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id, entry.session_id, entry.turn_number,
                entry.original_response, entry.feedback,
                entry.accuracy_score, entry.depth_score, entry.honesty_score,
                entry.improved_response, entry.created_at
            )
        )
        
        log.info(f"Improvement log recorded for turn {turn_number}")

    async def get_recent_improvements(self, limit: int = 10) -> list[ImprovementLogEntry]:
        """Fetch recent self-improvement logs."""
        rows = await self.db.fetch_all(
            "SELECT * FROM improvement_logs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        return [
            ImprovementLogEntry(
                id=r["id"],
                session_id=r["session_id"],
                turn_number=r["turn_number"],
                original_response=r["original_response"],
                feedback=r["feedback"],
                accuracy_score=r["accuracy_score"],
                depth_score=r["depth_score"],
                honesty_score=r["honesty_score"],
                improved_response=r["improved_response"],
                created_at=r["created_at"],
            ) for r in rows
        ]
