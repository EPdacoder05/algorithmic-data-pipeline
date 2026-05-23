"""
Universal live-total regime classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    """Deterministic game-state regime labels."""

    RUNWAY = "RUNWAY"
    NEUTRAL = "NEUTRAL"
    COMPRESSION = "COMPRESSION"


@dataclass(frozen=True)
class RegimeFeatures:
    """State snapshot used to classify live-total regime."""

    quarter: int
    time_remaining_seconds: float
    possessions_remaining: float
    score_margin: float
    quarter_scoring_rate: float
    baseline_scoring_rate: float
    foul_risk: float = 0.0
    trading_threes_risk: float = 0.0
    timeout_cluster_risk: float = 0.0

    @property
    def chaos_proxy(self) -> float:
        """Composite late-game volatility proxy in [0, 1]."""
        raw = (self.foul_risk + self.trading_threes_risk + self.timeout_cluster_risk) / 3.0
        return max(0.0, min(1.0, raw))


def classify_regime(features: RegimeFeatures) -> Regime:
    """Classify game-state regime for all quarters using deterministic rules."""
    safe_baseline = max(features.baseline_scoring_rate, 0.1)
    rate_ratio = features.quarter_scoring_rate / safe_baseline
    margin = abs(features.score_margin)

    runway_score = 0.0
    if features.possessions_remaining >= 10:
        runway_score += 1.1
    if features.time_remaining_seconds >= 360:
        runway_score += 1.0
    if margin <= 8:
        runway_score += 0.45
    if rate_ratio >= 0.95:
        runway_score += 0.35
    if features.chaos_proxy >= 0.55:
        runway_score += 0.3

    compression_score = 0.0
    if features.possessions_remaining <= 7:
        compression_score += 1.0
    if features.time_remaining_seconds <= 240:
        compression_score += 1.0
    if margin >= 10:
        compression_score += 0.45
    if rate_ratio <= 0.9:
        compression_score += 0.35
    if features.quarter >= 4 and features.time_remaining_seconds <= 300:
        compression_score += 0.35

    delta = runway_score - compression_score
    if delta >= 0.7:
        return Regime.RUNWAY
    if delta <= -0.7:
        return Regime.COMPRESSION
    return Regime.NEUTRAL
