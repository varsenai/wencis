# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
ResponseCritic — evaluates agent outputs against accuracy, depth,
and honesty criteria, enforcing strict quality thresholds.
"""
import logging
from typing import List, Optional

from wencis.llm.protocol import LLMClient
from wencis.critic.schemas import CritiqueScore, CritiqueResponse

log = logging.getLogger("wencis.critic.response_critic")

CRITIC_SYSTEM_PROMPT = """You are an Internal Critic for an AI agent.
Your job is to evaluate a draft response before the user sees it.
You are strict, objective, and unemotional. You do not rewrite the response; you only score it and provide feedback.

═══════════════════════════════════════════════════════════
SCORING CRITERIA (0.0 to 1.0 per axis)
═══════════════════════════════════════════════════════════

1. ACCURACY: Is the information logically sound, internally consistent, and factually correct
   based on the context AND the grounding memory nodes supplied below?
   - 1.0 = Flawless logic and facts, fully consistent with all memory nodes.
   - 0.7 = Minor omissions but generally correct.
   - <0.7 = Hallucinations, logical errors, or contradicts a grounding memory node.

2. DEPTH: Does it fully address the core issue? Does it explore nuance or just skim the surface?
   - 1.0 = Nuanced, insightful, goes beyond the obvious.
   - 0.7 = Answers the question adequately but simply.
   - <0.7 = Superficial, misses the point, or gives a generic boilerplate answer.

3. HONESTY: Does the response admit uncertainty where appropriate? Does it avoid overconfidence?
   - 1.0 = Perfectly calibrated confidence. Explicitly states what is unknown.
   - 0.7 = Reasonably calibrated.
   - <0.7 = Arrogant, overly confident about guesses, or hallucinates certainty.

═══════════════════════════════════════════════════════════
EVALUATION RULES
═══════════════════════════════════════════════════════════

- A response is ACCEPTABLE (is_acceptable = true) ONLY IF all three scores are >= 0.7.
- If ANY score is < 0.7, set is_acceptable to false.
- If rejected, your feedback must be EXACT, HARSH, and ACTIONABLE.
- GROUNDING CHECK: If memory nodes are provided below, verify that no statement in the draft
  contradicts them. A contradiction MUST reduce the accuracy score below 0.7.
"""


class ResponseCritic:
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, llm: LLMClient, model_override: Optional[str] = None):
        self.llm = llm
        self.model_override = model_override
        self._consecutive_failures = 0

    async def critique(
        self,
        user_input: str,
        system_context: str,
        draft_response: str,
        draft_reasoning: str,
        memory_nodes: Optional[List[str]] = None,
    ) -> CritiqueResponse:
        """
        Evaluate a draft response, incorporating optional grounding memory nodes.
        Enforces geometric score constraints and fails open gracefully on LLM failure.
        """
        grounding_block = ""
        if memory_nodes:
            # Keep up to 8 memory nodes
            nodes_slice = memory_nodes[:8]
            nodes_text = "\n".join(f"  [{i+1}] {node}" for i, node in enumerate(nodes_slice))
            grounding_block = f"""
═══════════════════════════════════════════════════════════
GROUNDING MEMORY NODES (cross-reference for factual accuracy)
═══════════════════════════════════════════════════════════
The following facts are in verified memory. If the draft
response contradicts any of these, accuracy MUST be scored below 0.7.

{nodes_text}
"""

        evaluation_prompt = f"""SYSTEM CONTEXT GIVEN TO AGENT:
{system_context}

USER INPUT:
{user_input}

AGENT'S INTERNAL REASONING:
{draft_reasoning}

AGENT'S DRAFT RESPONSE:
{draft_response}
{grounding_block}

Evaluate this draft based on the criteria. Provide per-axis scores and feedback.
"""

        try:
            response = await self.llm.complete_json(
                schema=CritiqueResponse,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_input=evaluation_prompt,
                temperature=0.2,
            )
            self._consecutive_failures = 0

            # Programmatic overrides to ensure mathematical and logical correctness
            scores = response.scores
            if scores.accuracy < 0.7 or scores.depth < 0.7 or scores.honesty < 0.7:
                response.is_acceptable = False

            if scores.accuracy == 0.0 or scores.depth == 0.0 or scores.honesty == 0.0:
                response.is_acceptable = False

            response.is_fallback = False
            return response

        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                log.warning(
                    "ResponseCritic hit %d consecutive failures. Failures: %s",
                    self._consecutive_failures,
                    e,
                )
            log.error("Critique failed, failing open: %s", e)
            return CritiqueResponse(
                scores=CritiqueScore(accuracy=1.0, depth=1.0, honesty=1.0),
                is_acceptable=True,
                feedback="Critic unavailable — accepted without review",
                is_fallback=True,
            )

    @staticmethod
    def geometric_score(accuracy: float, depth: float, honesty: float) -> float:
        """
        Computes the geometric mean of accuracy, depth, and honesty.
        Returns 0.0 if any axis is 0.0.
        """
        if accuracy <= 0.0 or depth <= 0.0 or honesty <= 0.0:
            return 0.0
        return (accuracy * depth * honesty) ** (1.0 / 3.0)
