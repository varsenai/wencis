from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from silex.storage.database import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageTracker:
    """Persist usage, cost, and operator telemetry."""

    def __init__(self, db: Database):
        self.db = db

    async def log_llm_call(
        self,
        *,
        provider: str,
        model: str,
        request_kind: str,
        input_tokens: int | None,
        output_tokens: int | None,
        estimated_cost_usd: float | None,
        duration_ms: int,
        success: bool,
        error: str | None = None,
        session_id: str | None = None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO llm_usage (
                id, session_id, provider, model, request_kind, input_tokens,
                output_tokens, estimated_cost_usd, duration_ms, success, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                session_id,
                provider,
                model,
                request_kind,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                duration_ms,
                int(success),
                error,
                utc_now(),
            ),
        )

    async def summary(self) -> dict[str, Any]:
        rows = await self.db.fetch_all(
            """
            SELECT provider, model, COUNT(*) AS requests,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                   COALESCE(AVG(duration_ms), 0) AS avg_duration_ms
            FROM llm_usage
            GROUP BY provider, model
            ORDER BY estimated_cost_usd DESC, requests DESC
            """
        )
        totals = await self.db.fetch_one(
            """
            SELECT COUNT(*) AS requests,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
            FROM llm_usage
            """
        ) or {}
        approvals = await self.db.fetch_all(
            """
            SELECT status, COUNT(*) AS count
            FROM tool_approvals
            GROUP BY status
            """
        )
        tool_logs = await self.db.fetch_all(
            """
            SELECT tool_name, COUNT(*) AS calls, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes
            FROM action_logs
            GROUP BY tool_name
            ORDER BY calls DESC
            LIMIT 10
            """
        )
        return {
            "totals": totals,
            "models": rows,
            "approvals": approvals,
            "tools": tool_logs,
        }
