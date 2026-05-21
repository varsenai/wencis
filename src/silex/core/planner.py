"""
Durable planner for complex ARIA tasks.

This is intentionally conservative: it records a resumable plan artifact and
reconciles tool progress, without asking the LLM to produce a new schema yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from silex.models.schemas import Plan, PlanStatus, PlanStep, ToolResult
from silex.storage.database import Database
from silex.utils.logger import setup_logger

log = setup_logger("silex.planner")


class Planner:
    """SQLite-backed plan and step tracker."""

    COMPLEX_TRIGGERS = {
        "build", "implement", "fix", "refactor", "audit", "review",
        "research", "deploy", "debug", "improve", "create", "add",
    }

    def __init__(self, db: Database):
        self.db = db

    async def get_active_plan(self, session_id: str) -> dict | None:
        """Query for an active plan and its associated steps."""
        plan_row = await self.db.fetch_one(
            "SELECT * FROM plans WHERE session_id = ? AND status = 'active' LIMIT 1",
            (session_id,)
        )
        if not plan_row:
            return None
        
        step_rows = await self.db.fetch_all(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_number ASC",
            (plan_row["id"],)
        )
        
        return {
            "plan": dict(plan_row),
            "steps": [dict(r) for r in step_rows]
        }

    def should_plan(self, user_input: str, tool_count: int = 0) -> bool:
        words = {w.strip(".,!?;:").lower() for w in user_input.split()}
        return tool_count > 0 or len(user_input) > 220 or bool(words & self.COMPLEX_TRIGGERS)

    async def create_plan(self, user_input: str, session_id: str | None, tool_names: list[str]) -> Plan:
        title = self._make_title(user_input)
        plan = Plan(
            session_id=session_id,
            title=title,
            user_input=user_input,
            success_criteria="Produce a correct, safe response and reconcile any tool results.",
            tool_budget=max(4, len(tool_names) + 4),
        )
        await self.db.execute(
            """
            INSERT INTO plans (id, session_id, title, user_input, status,
                               success_criteria, tool_budget, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.id, plan.session_id, plan.title, plan.user_input,
                plan.status.value, plan.success_criteria, plan.tool_budget,
                plan.created_at, plan.updated_at,
            ),
        )

        step_descriptions = ["Understand the request and gather relevant context."]
        if tool_names:
            step_descriptions.append(f"Execute tools safely: {', '.join(tool_names)}.")
            step_descriptions.append("Integrate tool results and update state.")
        step_descriptions.append("Return a concise final answer with uncertainty called out.")

        for idx, description in enumerate(step_descriptions, 1):
            await self.add_step(plan.id, idx, description, tool_names if idx == 2 else [])

        log.debug(f"Created plan {plan.id[:8]} for: {plan.title}")
        return plan

    async def add_step(self, plan_id: str, step_number: int, description: str, required_tools: list[str]) -> PlanStep:
        step = PlanStep(
            plan_id=plan_id,
            step_number=step_number,
            description=description,
            required_tools=required_tools,
        )
        await self.db.execute(
            """
            INSERT INTO plan_steps (id, plan_id, step_number, description, status,
                                    required_tools_json, result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.id, step.plan_id, step.step_number, step.description,
                step.status.value, json.dumps(step.required_tools),
                step.result, step.created_at, step.updated_at,
            ),
        )
        return step

    async def reconcile_tools(self, plan_id: str | None, results: list[ToolResult]) -> None:
        if not plan_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        failures = [r for r in results if not r.success]
        status = PlanStatus.BLOCKED if failures else PlanStatus.COMPLETED
        summary = (
            f"{len(results)} tool calls executed; {len(failures)} failed."
            if results else "No tools executed."
        )
        await self.db.execute(
            """
            UPDATE plan_steps
            SET status = ?, result = ?, updated_at = ?
            WHERE plan_id = ? AND step_number = 2
            """,
            (status.value, summary, now, plan_id),
        )

    async def complete_plan(self, plan_id: str | None, blocked: bool = False) -> None:
        if not plan_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        status = PlanStatus.BLOCKED if blocked else PlanStatus.COMPLETED
        await self.db.execute(
            "UPDATE plans SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, now, plan_id),
        )
        if not blocked:
            await self.db.execute(
                """
                UPDATE plan_steps
                SET status = ?, updated_at = ?
                WHERE plan_id = ? AND status = 'active'
                """,
                (PlanStatus.COMPLETED.value, now, plan_id),
            )

    @staticmethod
    def _make_title(user_input: str) -> str:
        collapsed = " ".join(user_input.split())
        return collapsed[:80] or "Untitled ARIA task"
