# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from wencis.meta.drift import LocalAlignmentVerifier


def test_drift_variance_budget_halt():
    verifier = LocalAlignmentVerifier(
        baseline_prompt="System prompt",
        stability_threshold=-0.015,
        variance_budget=0.15,
    )
    verifier.cosine_drifts = [0.05, 0.08, 0.20]  # Exceeds budget
    verifier.composite_scores = [0.9, 0.9, 0.9]
    assert verifier.analyze_drift_trend() is False


def test_drift_systematic_decay():
    verifier = LocalAlignmentVerifier(
        baseline_prompt="System prompt",
        stability_threshold=-0.015,
        variance_budget=0.15,
    )
    verifier.cosine_drifts = [0.01, 0.02, 0.02, 0.03, 0.03]
    # Steep decline: slope will be -0.1 per turn, way below stability threshold (-0.015)
    verifier.composite_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert verifier.analyze_drift_trend() is False


def test_drift_stable_trend():
    verifier = LocalAlignmentVerifier(
        baseline_prompt="System prompt",
        stability_threshold=-0.015,
        variance_budget=0.15,
    )
    verifier.cosine_drifts = [0.01, 0.02, 0.01, 0.02, 0.01]
    verifier.composite_scores = [0.85, 0.86, 0.84, 0.85, 0.85]
    assert verifier.analyze_drift_trend() is True
