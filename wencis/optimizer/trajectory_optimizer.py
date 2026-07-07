# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
TrajectoryOptimizer — optimizes historical run trajectories using revision,
recombination, and refinement operators.
"""
import json
import logging
import uuid
import time
from typing import Optional, List, Dict, Any

from wencis.storage.protocol import StorageBackend
from wencis.llm.protocol import LLMClient
from wencis.optimizer.schemas import (
    TrajectoryRevisionResult,
    TrajectoryRefinementResult,
    StepRevision,
    StepRefinement,
)

log = logging.getLogger("wencis.optimizer.trajectory_optimizer")

REVISION_SYSTEM_PROMPT = """You are a Metacognitive Revision Engine for an AI agent.
Your task is to review a series of execution steps from a task, locate where errors or
dead-ends occurred, write a critique summarizing the failure mode, and propose revised
epistemic categories for the steps. Step categories must be exactly one of:
'decision', 'hypothesis', 'fact', 'dead_end'.
If a step was a mistake or led to a dead-end, its revised category should be 'dead_end'."""

REFINEMENT_SYSTEM_PROMPT = """You are a Token Footprint Refinement Engine for an AI agent.
Review the provided verbose execution outputs and produce a compressed/summarized version
for each step, keeping all critical errors, paths, and metadata, but stripping redundant logs."""


class TrajectoryOptimizer:
    def __init__(self, backend: StorageBackend, llm: LLMClient):
        self.backend = backend
        self.llm = llm

    async def run_revision(self, trajectory_id: str) -> dict:
        """
        Runs LLM-guided critique and category revision on a trajectory's steps.
        Updates revised step categories in the database.
        """
        traj = await self.backend.fetch_one(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        )
        if not traj:
            raise ValueError(f"Trajectory {trajectory_id} not found.")

        steps = await self.backend.fetch_all(
            "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY step_order ASC",
            (trajectory_id,),
        )

        steps_summary = []
        for s in steps:
            exec_out = s.get("execution_output", "")
            if len(exec_out) > 300:
                exec_out = exec_out[:300] + "..."
            steps_summary.append(
                {
                    "step_order": s["step_order"],
                    "action_name": s["action_name"],
                    "tool_input": s["tool_input"],
                    "execution_output": exec_out,
                    "current_category": s["epistemic_category"],
                }
            )

        user_input = json.dumps(
            {
                "task_description": traj["task_description"],
                "is_success": traj["is_success"],
                "steps": steps_summary,
            },
            indent=2,
        )

        result = await self.llm.complete_json(
            schema=TrajectoryRevisionResult,
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_input=user_input,
            temperature=0.2,
        )

        async with self.backend.transaction():
            for rev in result.step_revisions:
                await self.backend.execute(
                    """
                    UPDATE trajectory_steps
                    SET epistemic_category = ?
                    WHERE trajectory_id = ? AND step_order = ?
                    """,
                    (rev.revised_category, trajectory_id, rev.step_order),
                )

        return {
            "trajectory_id": trajectory_id,
            "critique": result.critique,
            "step_revisions": [
                {
                    "step_order": r.step_order,
                    "revised_category": r.revised_category,
                    "reasoning": r.reasoning,
                }
                for r in result.step_revisions
            ],
        }

    async def run_recombination(self, trajectory_id_1: str, trajectory_id_2: str) -> Optional[str]:
        """
        Splicing two successful trajectories at a crossover point.
        Returns the new trajectory ID or None.
        """
        traj1 = await self.backend.fetch_one(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id_1,)
        )
        traj2 = await self.backend.fetch_one(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id_2,)
        )

        if not traj1 or not traj2:
            log.warning("One or both trajectories not found for recombination.")
            return None

        if traj1["is_success"] == 0 or traj2["is_success"] == 0:
            log.warning("Recombination requires both trajectories to be successful runs.")
            return None

        steps1 = await self.backend.fetch_all(
            "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY step_order ASC",
            (trajectory_id_1,),
        )
        steps2 = await self.backend.fetch_all(
            "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY step_order ASC",
            (trajectory_id_2,),
        )

        # Find crossover points (matching tool actions, not "initialize")
        crossovers = []
        for i, s1 in enumerate(steps1):
            for j, s2 in enumerate(steps2):
                if s1["action_name"] == s2["action_name"] and s1["action_name"] != "initialize":
                    crossovers.append((i, j, s1["action_name"]))

        if not crossovers:
            log.info("No crossover points found for recombination.")
            return None

        # Take the first crossover point
        idx1, idx2, _ = crossovers[0]

        combined_steps = []
        # steps1 up to and including idx1
        combined_steps.extend(steps1[: idx1 + 1])
        # steps2 starting after idx2
        combined_steps.extend(steps2[idx2 + 1 :])

        # Renumber step_order sequentially from 1
        new_steps = []
        total_latency_ms = 0.0
        total_tokens = 0
        for i, step in enumerate(combined_steps):
            new_step = dict(step)
            new_step["step_order"] = i + 1
            total_latency_ms += float(new_step.get("latency_ms", 0.0))
            total_tokens += int(new_step.get("token_usage", 0))
            new_steps.append(new_step)

        new_trajectory_id = f"recomb_{uuid.uuid4().hex[:8]}"

        async with self.backend.transaction():
            # Insert trajectory
            await self.backend.execute(
                """
                INSERT INTO trajectories
                    (trajectory_id, task_description, is_success, cumulative_latency, total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    new_trajectory_id,
                    f"Spliced: {traj1['task_description']} + {traj2['task_description']}",
                    1,
                    total_latency_ms / 1000.0,
                    total_tokens,
                    time.time(),
                ),
            )

            # Insert steps
            for s in new_steps:
                await self.backend.execute(
                    """
                    INSERT INTO trajectory_steps
                        (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_trajectory_id,
                        s["step_order"],
                        s["action_name"],
                        s["tool_input"],
                        s["execution_output"],
                        s["epistemic_category"],
                        s["latency_ms"],
                        s["token_usage"],
                    ),
                )

        return new_trajectory_id

    async def run_refinement(self, trajectory_id: str) -> str:
        """
        Removes redundant consecutive tool calls and compresses verbose output.
        """
        traj = await self.backend.fetch_one(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        )
        if not traj:
            raise ValueError(f"Trajectory {trajectory_id} not found.")

        steps = await self.backend.fetch_all(
            "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY step_order ASC",
            (trajectory_id,),
        )

        # 1. Heuristic prune: remove consecutive duplicate tool calls
        pruned = []
        for step in steps:
            if (
                pruned
                and pruned[-1]["action_name"] == step["action_name"]
                and pruned[-1]["tool_input"] == step["tool_input"]
            ):
                continue
            pruned.append(dict(step))

        # 2. LLM compress: execution outputs > 500 chars
        steps_to_compress = [
            {
                "step_order": s["step_order"],
                "action_name": s["action_name"],
                "execution_output": s["execution_output"],
            }
            for s in pruned
            if len(s["execution_output"]) > 500
        ]

        compression_map = {}
        llm_success = False

        if steps_to_compress:
            try:
                result = await self.llm.complete_json(
                    schema=TrajectoryRefinementResult,
                    system_prompt=REFINEMENT_SYSTEM_PROMPT,
                    user_input=json.dumps(steps_to_compress, indent=2),
                    temperature=0.1,
                )
                for ref in result.step_refinements:
                    compression_map[ref.step_order] = (
                        ref.compressed_output,
                        ref.token_saved_estimate,
                    )
                llm_success = True
            except Exception as e:
                log.error("LLM refinement failed, running fallback truncation: %s", e)

        # Apply compression and fallback
        final_steps = []
        total_latency_ms = 0.0
        total_tokens = 0

        for i, s in enumerate(pruned):
            step_order = s["step_order"]
            s["step_order"] = i + 1  # Re-sequence

            if step_order in compression_map:
                compressed_out, saved_tokens = compression_map[step_order]
                s["execution_output"] = compressed_out
                s["token_usage"] = max(50, int(s["token_usage"]) - int(saved_tokens))
            elif not llm_success and len(s["execution_output"]) > 1000:
                # Fallback: truncate and multiply tokens by 0.6
                s["execution_output"] = (
                    s["execution_output"][:800] + "\n[Truncated by Refinement]"
                )
                s["token_usage"] = max(50, int(int(s["token_usage"]) * 0.6))

            total_latency_ms += float(s.get("latency_ms", 0.0))
            total_tokens += int(s.get("token_usage", 0))
            final_steps.append(s)

        async with self.backend.transaction():
            # Delete old steps
            await self.backend.execute(
                "DELETE FROM trajectory_steps WHERE trajectory_id = ?", (trajectory_id,)
            )

            # Re-insert steps
            for s in final_steps:
                await self.backend.execute(
                    """
                    INSERT INTO trajectory_steps
                        (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory_id,
                        s["step_order"],
                        s["action_name"],
                        s["tool_input"],
                        s["execution_output"],
                        s["epistemic_category"],
                        s["latency_ms"],
                        s["token_usage"],
                    ),
                )

            # Update trajectory statistics
            await self.backend.execute(
                """
                UPDATE trajectories
                SET total_tokens = ?, cumulative_latency = ?
                WHERE trajectory_id = ?
                """,
                (total_tokens, total_latency_ms / 1000.0, trajectory_id),
            )

        return trajectory_id
