# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
Critique schemas for Pydantic validation.
"""
from pydantic import BaseModel, Field


class CritiqueScore(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0, description="Factual correctness [0-1]")
    depth: float = Field(ge=0.0, le=1.0, description="Thoroughness of reasoning [0-1]")
    honesty: float = Field(ge=0.0, le=1.0, description="Calibrated confidence [0-1]")


class CritiqueResponse(BaseModel):
    scores: CritiqueScore
    feedback: str = Field(description="Specific, actionable critique")
    is_acceptable: bool = Field(description="True only if ALL scores >= 0.7")
    is_fallback: bool = Field(default=False, description="True if the response is a system fallback")
