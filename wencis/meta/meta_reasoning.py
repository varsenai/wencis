# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
MetaReasoningEngine — analyzes historical turns, critic rejections, and failures,
generating self-improvement proposals based on structural patterns.
"""
from datetime import datetime, timezone
import json
import logging
from typing import Optional, List, Dict, Any, Callable

from wencis.storage.protocol import StorageBackend
from wencis.llm.protocol import LLMClient
from wencis.meta.schemas import (
    FailureClusterReport,
    MetaAnalysisResponse,
    SelfImprovementProposal,
)

log = logging.getLogger("wencis.meta.meta_reasoning")

META_ANALYSIS_PROMPT = """You are a Meta-Reasoning Analyst for an AI agent system.
You will receive pre-computed performance analytics. The data has already been
statistically processed — your job is to interpret the structured analysis
and propose ONE specific, actionable improvement.

The data includes:
- A FailureClusterReport: per-axis average scores, failure counts, bottleneck axis
- A confidence drift slope: negative = declining over time
- Raw tool failure descriptions
- Open uncertainty topics

Rules:
1. Only propose changes for REAL, REPEATED failures — not one-off mistakes.
2. Every proposal must include a measurable success metric.
3. Be conservative. A bad change is worse than no change.
4. You must specify exactly which system to modify.
5. If the data shows no clear pattern, set has_proposal to false.
6. PRIORITIZE the bottleneck_axis from the cluster report — this is the
   mathematically weakest dimension.

Valid target systems:
- system_prompt: Changes to the agent's base instructions
- tool_registry: Adding new tools or modifying existing ones
- cognitive_loop: Changes to the reasoning pipeline
- memory_store: Changes to memory storage or retrieval
- other: Anything else"""


class MetaReasoningEngine:
    def __init__(self, llm: LLMClient, backend: StorageBackend):
        self.llm = llm
        self.backend = backend

    async def analyze_and_propose(
        self, status_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[SelfImprovementProposal]:
        """
        Runs the 3-phase meta-reasoning analysis to identify performance
        bottlenecks and generate self-improvement proposals.
        """
        if status_callback:
            status_callback("Phase 1: Fetching raw analytics data from storage...")

        rejection_rows = await self.backend.fetch_all(
            "SELECT * FROM improvement_logs ORDER BY created_at DESC LIMIT 20"
        )
        tool_failures = await self.backend.fetch_all(
            "SELECT * FROM action_logs WHERE success = 0 ORDER BY created_at DESC LIMIT 10"
        )
        uncertainties = await self.backend.fetch_all(
            "SELECT * FROM uncertainties WHERE status = 'open' ORDER BY created_at DESC LIMIT 5"
        )
        recent_turns = await self.backend.fetch_all(
            "SELECT confidence FROM turns ORDER BY created_at DESC LIMIT 30"
        )

        if not rejection_rows and not tool_failures:
            log.info("No failure or correction data available for meta-reasoning.")
            if status_callback:
                status_callback("Aborted: Insufficient data to analyze.")
            return None

        if status_callback:
            status_callback("Phase 2: Running statistical preprocessing algorithms...")

        cluster_report = self._compute_failure_clusters(rejection_rows)
        confidences = [float(t["confidence"]) for t in recent_turns]
        confidence_slope = self._compute_confidence_drift(confidences)

        # Build performance_data prompt input
        sections = []

        if cluster_report:
            sections.append(cluster_report.to_prompt_block())

        trend_label = "STABLE"
        if confidence_slope < -0.005:
            trend_label = "DECLINING"
        elif confidence_slope > 0.005:
            trend_label = "IMPROVING"

        drift_block = f"""═══════════════════════════════════════════════════════════
CONFIDENCE DRIFT ANALYSIS
═══════════════════════════════════════════════════════════
  Confidence slope (beta): {confidence_slope:.6f}
  Trend identified: {trend_label}
"""
        sections.append(drift_block)

        tool_failures_text = "  None"
        if tool_failures:
            failures_list = []
            for f in tool_failures:
                outcome = f.get("actual_outcome", "")[:100]
                failures_list.append(f"  - Tool: {f['tool_name']}, Outcome: {outcome}...")
            tool_failures_text = "\n".join(failures_list)

        failures_block = f"""═══════════════════════════════════════════════════════════
RECENT TOOL FAILURES
═══════════════════════════════════════════════════════════
{tool_failures_text}
"""
        sections.append(failures_block)

        uncertainties_text = "  None"
        if uncertainties:
            unc_list = []
            for u in uncertainties:
                unc_list.append(f"  - Topic: {u['topic']}, Why uncertain: {u['why_uncertain']}")
            uncertainties_text = "\n".join(unc_list)

        uncertainties_block = f"""═══════════════════════════════════════════════════════════
UNRESOLVED UNCERTAINTIES
═══════════════════════════════════════════════════════════
{uncertainties_text}
"""
        sections.append(uncertainties_block)

        performance_data = "\n".join(sections)

        if status_callback:
            status_callback("Phase 3: Dispatching to LLM for meta-proposal formulation...")

        try:
            analysis = await self.llm.complete_json(
                schema=MetaAnalysisResponse,
                system_prompt=META_ANALYSIS_PROMPT,
                user_input=performance_data,
                temperature=0.2,
            )

            if analysis.has_proposal:
                proposal = SelfImprovementProposal(
                    target_system=analysis.target_system,
                    description=analysis.description,
                    rationale=analysis.rationale,
                    success_metric=analysis.success_metric,
                )

                # Persist proposal
                sql = """
                INSERT INTO improvement_proposals
                    (id, target_system, description, rationale, success_metric, status, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    proposal.id,
                    proposal.target_system,
                    proposal.description,
                    proposal.rationale,
                    proposal.success_metric,
                    proposal.status,
                    proposal.created_at,
                    proposal.resolved_at,
                )
                await self.backend.execute(sql, params)

                if status_callback:
                    status_callback(f"Successfully generated proposal: {proposal.description}")
                return proposal

            if status_callback:
                status_callback("Meta-analysis completed: No proposal suggested.")
            return None

        except Exception as e:
            log.error("Meta-reasoning analysis LLM call failed: %s", e)
            if status_callback:
                status_callback(f"Error during LLM formulation: {e}")
            return None

    def _compute_failure_clusters(self, rows: List[Dict[str, Any]]) -> FailureClusterReport:
        """
        Computes average per-axis scores, identifies the bottleneck axis,
        and aggregates sample rejection feedback comments.
        """
        n = len(rows)
        if n == 0:
            return FailureClusterReport(0, 1.0, 1.0, 1.0, "none", 1.0)

        avg_acc = sum(float(r.get("accuracy_score", 1.0)) for r in rows) / n
        avg_dep = sum(float(r.get("depth_score", 1.0)) for r in rows) / n
        avg_hon = sum(float(r.get("honesty_score", 1.0)) for r in rows) / n

        axis_scores = {"accuracy": avg_acc, "depth": avg_dep, "honesty": avg_hon}
        bottleneck = min(axis_scores, key=axis_scores.__getitem__)

        feedbacks = [str(r.get("feedback", ""))[:120] for r in rows[:5] if r.get("feedback")]

        return FailureClusterReport(
            total_rejections=n,
            avg_accuracy=avg_acc,
            avg_depth=avg_dep,
            avg_honesty=avg_hon,
            bottleneck_axis=bottleneck,
            bottleneck_avg=axis_scores[bottleneck],
            sample_feedbacks=feedbacks,
        )

    def _compute_confidence_drift(self, confidences: List[float]) -> float:
        """
        Fits confidence drift to a chronologically ordered line to measure
        systematic drift rate.
        """
        n = len(confidences)
        if n < 2:
            return 0.0

        # Reverse so index 0 = oldest (chronological order)
        vals = list(reversed(confidences))
        xs = list(range(n))

        mean_x = sum(xs) / n
        mean_y = sum(vals) / n

        numerator = sum((xs[i] - mean_x) * (vals[i] - mean_y) for i in range(n))
        denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0
