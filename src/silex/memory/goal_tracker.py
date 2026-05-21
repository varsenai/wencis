"""
Goal Tracker — ARIA's objective management.

Handles creating, updating, completing, and abandoning goals.
Goals persist across sessions and drive ARIA's sense of purpose.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from silex.models.schemas import Goal, GoalPriority, GoalStatus
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.goals")


class GoalTracker:
    """SQLite-backed goal lifecycle manager for ARIA."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(self, description: str, priority: str = "medium") -> Goal:
        """Create a new active goal."""
        goal = Goal(
            description=description,
            priority=GoalPriority(priority),
            status=GoalStatus.ACTIVE,
        )
        await self.db.execute(
            """
            INSERT INTO goals (id, description, status, priority,
                               created_at, updated_at, sub_goals, completion_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goal.id,
                goal.description,
                goal.status.value,
                goal.priority.value,
                goal.created_at,
                goal.updated_at,
                json.dumps(goal.sub_goals),
                goal.completion_notes,
            ),
        )
        log.info(f"Created goal: {description}")
        return goal

    async def get(self, goal_id: str) -> Goal | None:
        """Retrieve a single goal by ID."""
        row = await self.db.fetch_one("SELECT * FROM goals WHERE id = ?", (goal_id,))
        if row is None:
            return None
        return self._row_to_goal(row)

    async def get_active(self) -> list[Goal]:
        """Get all active goals, ordered by priority."""
        priority_order = "CASE priority " \
                         "WHEN 'critical' THEN 1 " \
                         "WHEN 'high' THEN 2 " \
                         "WHEN 'medium' THEN 3 " \
                         "WHEN 'low' THEN 4 END"
        rows = await self.db.fetch_all(
            f"SELECT * FROM goals WHERE status = 'active' ORDER BY {priority_order}"
        )
        return [self._row_to_goal(r) for r in rows]

    async def get_all(self) -> list[Goal]:
        """Get all goals regardless of status."""
        rows = await self.db.fetch_all("SELECT * FROM goals ORDER BY created_at DESC")
        return [self._row_to_goal(r) for r in rows]

    async def complete(self, goal_id: str, notes: str | None = None) -> Goal | None:
        """Mark a goal as completed."""
        return await self._update_status(goal_id, GoalStatus.COMPLETED, notes)

    async def abandon(self, goal_id: str, notes: str | None = None) -> Goal | None:
        """Mark a goal as abandoned."""
        return await self._update_status(goal_id, GoalStatus.ABANDONED, notes)

    async def find_by_description(self, description: str) -> Goal | None:
        """Find a goal by partial description match (case-insensitive)."""
        row = await self.db.fetch_one(
            "SELECT * FROM goals WHERE LOWER(description) LIKE ? AND status = 'active'",
            (f"%{description.lower()}%",),
        )
        if row is None:
            return None
        return self._row_to_goal(row)

    async def count_active(self) -> int:
        """Count active goals."""
        row = await self.db.fetch_one(
            "SELECT COUNT(*) as cnt FROM goals WHERE status = 'active'"
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _update_status(
        self, goal_id: str, status: GoalStatus, notes: str | None
    ) -> Goal | None:
        """Update a goal's status."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            UPDATE goals SET status = ?, updated_at = ?, completion_notes = ?
            WHERE id = ?
            """,
            (status.value, now, notes, goal_id),
        )
        goal = await self.get(goal_id)
        if goal:
            log.info(f"Goal {status.value}: {goal.description}")
        return goal

    @staticmethod
    def _row_to_goal(row: dict) -> Goal:
        """Convert a database row to a Goal model."""
        return Goal(
            id=row["id"],
            description=row["description"],
            status=GoalStatus(row["status"]),
            priority=GoalPriority(row["priority"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            sub_goals=json.loads(row["sub_goals"]),
            completion_notes=row["completion_notes"],
        )
