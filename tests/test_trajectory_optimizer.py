# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from wencis.optimizer.trajectory_optimizer import TrajectoryOptimizer
from wencis.optimizer.schemas import (
    TrajectoryRevisionResult,
    StepRevision,
    TrajectoryRefinementResult,
    StepRefinement,
)
from examples.mock_llm import MockLLMClient


@pytest.mark.asyncio
async def test_run_revision(backend, mock_llm):
    # Setup database with trajectory and steps
    await backend.execute(
        """
        INSERT INTO trajectories
            (trajectory_id, task_description, is_success, cumulative_latency, total_tokens, timestamp)
        VALUES ('traj-1', 'Test task', 0, 5.0, 100, ?)
        """,
        (time.time(),),
    )
    await backend.execute(
        """
        INSERT INTO trajectory_steps
            (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
        VALUES ('traj-1', 1, 'test_action', '{}', 'Test output', 'decision', 500.0, 50)
        """
    )

    # Setup LLM response
    mock_llm._responses[TrajectoryRevisionResult] = TrajectoryRevisionResult(
        critique="Step 1 was wrong",
        step_revisions=[
            StepRevision(
                step_order=1,
                revised_category="dead_end",
                reasoning="It failed",
            )
        ],
    )

    optimizer = TrajectoryOptimizer(backend, mock_llm)
    res = await optimizer.run_revision("traj-1")

    assert res["trajectory_id"] == "traj-1"
    assert res["critique"] == "Step 1 was wrong"
    assert res["step_revisions"][0]["revised_category"] == "dead_end"

    # Verify database was updated
    step = await backend.fetch_one(
        "SELECT * FROM trajectory_steps WHERE trajectory_id = 'traj-1' AND step_order = 1"
    )
    assert step["epistemic_category"] == "dead_end"


@pytest.mark.asyncio
async def test_run_recombination(backend):
    # Trajectory 1: Successful
    await backend.execute(
        """
        INSERT INTO trajectories
            (trajectory_id, task_description, is_success, cumulative_latency, total_tokens, timestamp)
        VALUES ('traj-1', 'Task A', 1, 5.0, 100, ?)
        """,
        (time.time(),),
    )
    # Steps: initialize -> action_x -> action_y
    await backend.execute(
        """
        INSERT INTO trajectory_steps
            (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
        VALUES 
            ('traj-1', 1, 'initialize', '{}', 'Init', 'decision', 100.0, 10),
            ('traj-1', 2, 'action_x', '{}', 'X Output', 'fact', 200.0, 20),
            ('traj-1', 3, 'action_y', '{}', 'Y Output', 'fact', 300.0, 30)
        """
    )

    # Trajectory 2: Successful
    await backend.execute(
        """
        INSERT INTO trajectories
            (trajectory_id, task_description, is_success, cumulative_latency, total_tokens, timestamp)
        VALUES ('traj-2', 'Task B', 1, 6.0, 120, ?)
        """,
        (time.time(),),
    )
    # Steps: initialize -> action_x -> action_z
    await backend.execute(
        """
        INSERT INTO trajectory_steps
            (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
        VALUES 
            ('traj-2', 1, 'initialize', '{}', 'Init', 'decision', 100.0, 10),
            ('traj-2', 2, 'action_x', '{}', 'X Output Newer', 'fact', 250.0, 25),
            ('traj-2', 3, 'action_z', '{}', 'Z Output', 'fact', 400.0, 40)
        """
    )

    optimizer = TrajectoryOptimizer(backend, MockLLMClient())
    recomb_id = await optimizer.run_recombination("traj-1", "traj-2")

    assert recomb_id is not None
    assert recomb_id.startswith("recomb_")

    # Combined steps should be:
    # From traj-1: initialize (1), action_x (2) [crossover at action_x]
    # From traj-2: action_z (3) [anything after action_x in traj-2]
    steps = await backend.fetch_all(
        "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY step_order ASC",
        (recomb_id,),
    )
    assert len(steps) == 3
    assert steps[0]["action_name"] == "initialize"
    assert steps[1]["action_name"] == "action_x"
    assert steps[1]["execution_output"] == "X Output"
    assert steps[2]["action_name"] == "action_z"
    assert steps[2]["execution_output"] == "Z Output"


@pytest.mark.asyncio
async def test_run_refinement(backend, mock_llm):
    await backend.execute(
        """
        INSERT INTO trajectories
            (trajectory_id, task_description, is_success, cumulative_latency, total_tokens, timestamp)
        VALUES ('traj-1', 'Refining task', 1, 5.0, 200, ?)
        """,
        (time.time(),),
    )
    # Steps with consecutive duplicates and one verbose output
    await backend.execute(
        """
        INSERT INTO trajectory_steps
            (trajectory_id, step_order, action_name, tool_input, execution_output, epistemic_category, latency_ms, token_usage)
        VALUES 
            ('traj-1', 1, 'duplicate_tool', '{"arg": 1}', 'Output 1', 'decision', 100.0, 10),
            ('traj-1', 2, 'duplicate_tool', '{"arg": 1}', 'Output 1', 'decision', 100.0, 10),
            ('traj-1', 3, 'verbose_tool', '{"verbose": true}', ?, 'fact', 500.0, 150)
        """,
        ('x' * 600,)
    )

    # Setup refinement mock
    mock_llm._responses[TrajectoryRefinementResult] = TrajectoryRefinementResult(
        rationale="Compressed verbose logs",
        step_refinements=[
            StepRefinement(
                step_order=3,
                compressed_output="Compressed x",
                token_saved_estimate=80,
            )
        ],
    )

    optimizer = TrajectoryOptimizer(backend, mock_llm)
    await optimizer.run_refinement("traj-1")

    # Verify duplicate is pruned and verbose output is compressed
    steps = await backend.fetch_all(
        "SELECT * FROM trajectory_steps WHERE trajectory_id = 'traj-1' ORDER BY step_order ASC",
    )
    assert len(steps) == 2  # The duplicate was removed
    assert steps[0]["action_name"] == "duplicate_tool"
    assert steps[1]["action_name"] == "verbose_tool"
    # Wait, the step order in pruned was 1: duplicate_tool, 3: verbose_tool. But in final_steps:
    # step_order should be mapped back and compressed. Let's make sure the compressed output matches
    assert steps[1]["execution_output"] == "Compressed x"
    assert steps[1]["token_usage"] == 70  # 150 - 80


def test_step_revision_invalid_category():
    from pydantic import ValidationError
    # revised_category must be decision, hypothesis, fact, or dead_end
    with pytest.raises(ValidationError):
        StepRevision(
            step_order=1,
            revised_category="invalid_category",
            reasoning="Test",
        )

