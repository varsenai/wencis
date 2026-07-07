# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

"""
LocalAlignmentVerifier — OLS linear regression over agent confidence scores.

Detects systematic degradation by computing the slope (beta) of a linear
fit over historical composite confidence values. A negative slope beyond
the stability threshold indicates the agent is getting worse over time.

Does NOT require numpy — implemented in pure Python to minimize dependencies.
"""
import logging
from typing import List

log = logging.getLogger("wencis.meta.drift")


class LocalAlignmentVerifier:
    """
    Detects confidence drift using OLS regression.

    Parameters
    ----------
    baseline_prompt: str
        The original system prompt (kept for reference).
    stability_threshold: float
        Minimum allowable OLS slope. Default: -0.015 (15 units per 1000 turns).
    variance_budget: float
        Maximum single-turn cosine drift. Default: 0.15.

    Attributes
    ----------
    composite_scores: List[float]
        Chronologically ordered (oldest to newest) confidence metrics.
    cosine_drifts: List[float]
        Chronologically ordered (oldest to newest) alignment drift measurements.
    """

    def __init__(
        self,
        baseline_prompt: str,
        stability_threshold: float = -0.015,
        variance_budget: float = 0.15,
    ):
        self.baseline_prompt = baseline_prompt
        self.stability_threshold = stability_threshold
        self.variance_budget = variance_budget
        self.composite_scores: List[float] = []
        self.cosine_drifts: List[float] = []

    def analyze_drift_trend(self) -> bool:
        """
        Run OLS regression over composite_scores.

        Returns True (stable) or False (decaying).
        Caller should set self.composite_scores and self.cosine_drifts before calling.
        Note: The input arrays must be sorted in chronological order (oldest first)
        for the regression slope direction to be correct.
        """
        # Instant halt: if latest drift exceeds variance budget
        if self.cosine_drifts and self.cosine_drifts[-1] > self.variance_budget:
            log.critical(
                "Instant alignment halt: drift %.4f exceeds budget %.4f",
                self.cosine_drifts[-1],
                self.variance_budget,
            )
            return False

        n = len(self.composite_scores)
        if n < 5:
            log.info("Insufficient history (%d samples) for drift regression.", n)
            return True  # Not enough data → assume stable

        # OLS: Y = alpha + beta * X
        xs = list(range(n))
        ys = self.composite_scores

        # Y = alpha + beta * X
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        numerator = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))

        beta = numerator / denominator if denominator != 0 else 0.0
        log.info("OLS drift analysis: beta=%.6f (threshold=%.6f)", beta, self.stability_threshold)

        if beta < self.stability_threshold:
            log.critical(
                "Systematic degradation detected: slope %.6f < threshold %.6f",
                beta,
                self.stability_threshold,
            )
            return False

        return True
