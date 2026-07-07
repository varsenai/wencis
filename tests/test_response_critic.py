# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from wencis.critic.response_critic import ResponseCritic
from wencis.critic.schemas import CritiqueScore, CritiqueResponse
from examples.mock_llm import MockLLMClient


def test_geometric_score_zero_collapse():
    assert ResponseCritic.geometric_score(0.9, 0.9, 0.0) == 0.0
    assert ResponseCritic.geometric_score(0.0, 1.0, 1.0) == 0.0


def test_geometric_score_all_high():
    assert ResponseCritic.geometric_score(1.0, 1.0, 1.0) == 1.0
    assert abs(ResponseCritic.geometric_score(0.9, 0.9, 0.9) - 0.9) < 1e-9


@pytest.mark.asyncio
async def test_critique_fails_open_on_llm_error():
    # Make a mock LLM client that throws an exception
    class FailingLLMClient:
        async def complete_json(self, **kwargs):
            raise RuntimeError("LLM Failure")

    critic = ResponseCritic(FailingLLMClient())
    res = await critic.critique(
        user_input="test",
        system_context="test",
        draft_response="test",
        draft_reasoning="test",
    )
    assert res.is_acceptable is True
    assert res.scores.accuracy == 1.0
    assert res.scores.depth == 1.0
    assert res.scores.honesty == 1.0
    assert "Critic unavailable" in res.feedback
    assert res.is_fallback is True


@pytest.mark.asyncio
async def test_critique_zero_axis_forces_unacceptable(mock_llm):
    # Setup canned response with zero score
    canned_res = CritiqueResponse(
        scores=CritiqueScore(accuracy=0.0, depth=1.0, honesty=1.0),
        feedback="Failed accuracy check",
        is_acceptable=True,  # Mock suggests True initially
        is_fallback=False,
    )
    mock_llm._responses[CritiqueResponse] = canned_res

    critic = ResponseCritic(mock_llm)
    res = await critic.critique(
        user_input="test",
        system_context="test",
        draft_response="test",
        draft_reasoning="test",
    )

    # Programmatic override should force is_acceptable to False
    assert res.is_acceptable is False
    assert res.is_fallback is False

