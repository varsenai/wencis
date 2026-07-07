# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer schemas for Pydantic validation.
"""
from typing import Literal
from pydantic import BaseModel, Field


class StepRevision(BaseModel):
    step_order: int = Field(..., description="Step index (1-based) to revise")
    revised_category: Literal["decision", "hypothesis", "fact", "dead_end"] = Field(
        ..., description="Must be: 'decision', 'hypothesis', 'fact', or 'dead_end'"
    )
    reasoning: str = Field(..., description="Why this category was revised")


class TrajectoryRevisionResult(BaseModel):
    critique: str = Field(..., description="Critique of the failure mode")
    step_revisions: list[StepRevision] = Field(..., description="Step category corrections")


class StepRefinement(BaseModel):
    step_order: int = Field(..., description="Step index (1-based) to compress")
    compressed_output: str = Field(..., description="Summarized output keeping critical info")
    token_saved_estimate: int = Field(..., description="Estimated tokens saved")


class TrajectoryRefinementResult(BaseModel):
    rationale: str = Field(..., description="Why certain steps were compressed")
    step_refinements: list[StepRefinement] = Field(..., description="Compressions to apply")
