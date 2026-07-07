# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
Meta schemas and helper dataclasses for meta-reasoning statistics.
"""
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


@dataclass
class FailureClusterReport:
    """Pre-computed statistical summary of critic failures."""

    total_rejections: int
    avg_accuracy: float
    avg_depth: float
    avg_honesty: float
    bottleneck_axis: str  # 'accuracy', 'depth', 'honesty', or 'none'
    bottleneck_avg: float  # Average score of the weakest axis
    sample_feedbacks: List[str] = dc_field(default_factory=list)

    def to_prompt_block(self) -> str:
        feedbacks_str = "\n".join(f"  - {f[:120]}" for f in self.sample_feedbacks[:5])
        return (
            "═══════════════════════════════════════════════════════════\n"
            "FAILURE CLUSTER REPORT (pre-computed)\n"
            "═══════════════════════════════════════════════════════════\n"
            f"  Total self-correction events:  {self.total_rejections}\n"
            f"  Avg accuracy across events:    {self.avg_accuracy:.3f}\n"
            f"  Avg depth across events:       {self.avg_depth:.3f}\n"
            f"  Avg honesty across events:     {self.avg_honesty:.3f}\n"
            f"  BOTTLENECK AXIS IDENTIFIED:    {self.bottleneck_axis.upper()} "
            f"(mean={self.bottleneck_avg:.3f})\n"
            f"\nSample rejection feedback messages:\n{feedbacks_str}\n"
        )


class MetaAnalysisResponse(BaseModel):
    has_proposal: bool = Field(description="Whether a valid improvement proposal was found")
    target_system: str = Field(default="", description="Which system to modify")
    description: str = Field(default="", description="What exactly to change")
    rationale: str = Field(default="", description="Why, with evidence from failures")
    success_metric: str = Field(default="", description="How to measure if it worked")


class SelfImprovementProposal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_system: str
    description: str
    rationale: str
    success_metric: str
    status: str = Field(default="pending")  # pending | approved | rejected | implemented
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
