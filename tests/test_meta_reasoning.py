# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from wencis.meta.meta_reasoning import MetaReasoningEngine
from wencis.meta.schemas import FailureClusterReport
from examples.mock_llm import MockLLMClient


def test_compute_failure_clusters_identifies_bottleneck():
    engine = MetaReasoningEngine(MockLLMClient(), None)
    rows = [
        {"accuracy_score": 0.9, "depth_score": 0.5, "honesty_score": 0.8},
        {"accuracy_score": 0.9, "depth_score": 0.5, "honesty_score": 0.8},
    ]
    report = engine._compute_failure_clusters(rows)
    assert report.bottleneck_axis == "depth"
    assert report.bottleneck_avg == 0.5


def test_compute_confidence_drift_negative_slope():
    engine = MetaReasoningEngine(MockLLMClient(), None)
    # Chronological: 0.9 (oldest), 0.8, 0.7, 0.6, 0.5 (newest)
    # The recent_turns array is ordered newest first, which matches:
    # [0.5, 0.6, 0.7, 0.8, 0.9]
    confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
    slope = engine._compute_confidence_drift(confidences)
    assert slope < 0.0


def test_compute_confidence_drift_stable():
    engine = MetaReasoningEngine(MockLLMClient(), None)
    confidences = [0.7, 0.7, 0.7, 0.7, 0.7]
    slope = engine._compute_confidence_drift(confidences)
    assert abs(slope) < 1e-9


@pytest.mark.asyncio
async def test_analyze_and_propose_returns_none_without_data(backend):
    engine = MetaReasoningEngine(MockLLMClient(), backend)
    proposal = await engine.analyze_and_propose()
    assert proposal is None
