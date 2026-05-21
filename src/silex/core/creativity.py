"""
Bounded creativity helpers for Explorer/Critic/Taste workflows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CreativeVariant:
    role: str
    prompt: str
    rubric: str


class CreativityStack:
    """Generate role prompts for divergent/convergent idea search."""

    def variants_for(self, task: str) -> list[CreativeVariant]:
        return [
            CreativeVariant(
                role="Explorer",
                prompt=f"Generate unconventional approaches for: {task}",
                rubric="Novelty, leverage, and cross-domain insight.",
            ),
            CreativeVariant(
                role="Critic",
                prompt=f"Attack weaknesses and risks in the proposed approaches for: {task}",
                rubric="Correctness, feasibility, safety, and missing constraints.",
            ),
            CreativeVariant(
                role="Taste",
                prompt=f"Evaluate elegance and premium product quality for: {task}",
                rubric="Simplicity, polish, coherence, restraint, and user delight.",
            ),
        ]

    def format_for_prompt(self, task: str) -> str:
        lines = [
            "CREATIVITY STACK",
            "Use this only for high-stakes design, architecture, or research tasks.",
        ]
        for variant in self.variants_for(task):
            lines.append(f"- {variant.role}: {variant.prompt} Rubric: {variant.rubric}")
        return "\n".join(lines)
