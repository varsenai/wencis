"""
Schemas for the Skill system (Phase 19 + Phase 21).

Provides structured output schemas for LLM-driven skill synthesis
and background review, plus models for parsing YAML frontmatter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillSynthesisResponse(BaseModel):
    """LLM output when synthesizing a skill from session history."""

    skill_name: str = Field(
        description="Short slug name using hyphens, e.g. 'deploy-docker-container'"
    )
    description: str = Field(
        description="One-line 'Use when...' trigger description"
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Required tools, libraries, or prerequisite knowledge",
    )
    steps: str = Field(
        description="The full skill body in Markdown — step-by-step workflow"
    )
    is_generalizable: bool = Field(
        description=(
            "True if this workflow applies beyond this specific session context. "
            "False if it was a one-off fix or too project-specific to reuse."
        )
    )


class SkillFrontmatter(BaseModel):
    """YAML frontmatter parsed from a skill .md file."""

    name: str = ""
    description: str = ""
    version: int = 1
    uses: int = 0
    created_at: str = ""
    last_used: str = ""
    refined: bool = False
    # Phase 21 — Lifecycle fields
    status: str = "active"       # active | stale | archived
    pinned: bool = False          # Pinned skills exempt from lifecycle transitions
    archived_at: str = ""         # ISO timestamp of archival
    absorbed_into: str = ""       # Name of umbrella skill if consolidated


class BackgroundReviewResponse(BaseModel):
    """Structured output from the Background Review Daemon (Phase 21)."""

    user_corrections: list[str] = Field(
        default_factory=list,
        description="Things the user explicitly corrected or clarified",
    )
    failed_approaches: list[str] = Field(
        default_factory=list,
        description="Approaches that were tried and failed",
    )
    new_memories: list[str] = Field(
        default_factory=list,
        description="Facts or preferences to remember for future sessions",
    )
    skill_candidates: list[dict] = Field(
        default_factory=list,
        description=(
            "Workflows worth saving as skills. "
            "Each dict: {name: str, description: str, steps: str}"
        ),
    )

