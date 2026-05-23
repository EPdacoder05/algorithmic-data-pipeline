"""
Universal all-quarter live totals classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from analysis.confidence import ConfidenceScorer, UnderConfidenceContext
from engine.regime import Regime, RegimeFeatures, classify_regime


class TotalsAction(str, Enum):
    """Allowed normalized actions."""

    BET_OVER = "BET_OVER"
    BET_UNDER = "BET_UNDER"
    PASS = "PASS"


@dataclass(frozen=True)
class GuardrailConfig:
    """Guardrails for all-quarter live totals decisions."""

    high_possessions_threshold: float = 10.0
    reachable_over_rate_band: Tuple[float, float] = (4.8, 5.8)
    burst_window_seconds: int = 90
    burst_rate_multiplier: float = 1.2
    under_strong_threshold: float = 0.72
    under_medium_threshold: float = 0.64
    under_lean_threshold: float = 0.58


@dataclass(frozen=True)
class LiveTotalsState:
    """Input state for a live-total decision."""

    quarter: int
    time_remaining_seconds: float
    possessions_remaining: float
    score_margin: float
    quarter_scoring_rate: float
    baseline_scoring_rate: float
    current_total: float
    posted_total_line: float
    base_edge: float = 0.0
    foul_risk: float = 0.0
    trading_threes_risk: float = 0.0
    timeout_cluster_risk: float = 0.0
    leader_clock_pressure: float = 0.0
    recent_window_seconds: Optional[int] = None
    recent_scoring_rate: Optional[float] = None
    burst_persistence_active: bool = False

    @property
    def over_points_needed(self) -> float:
        return max(0.0, self.posted_total_line - self.current_total)

    @property
    def over_required_rate(self) -> float:
        minutes = max(self.time_remaining_seconds, 1.0) / 60.0
        return self.over_points_needed / minutes


@dataclass(frozen=True)
class DecisionMeta:
    """Explainable score decomposition."""

    regime: str
    score: float
    compression_score: float
    chaos_score: float
    runway_over_pressure: float
    over_required_rate: float
    tier: str
    guardrails: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiveTotalsDecision:
    """Final action + separated confidence metadata."""

    action: str
    confidence: float
    meta: DecisionMeta


class LiveTotalsClassifier:
    """State-aware all-quarter classifier for totals decisions."""

    def __init__(
        self,
        *,
        guardrails: Optional[GuardrailConfig] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
    ):
        self.guardrails = guardrails or GuardrailConfig()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def classify(self, state: LiveTotalsState) -> LiveTotalsDecision:
        """Return BET_OVER/BET_UNDER/PASS with confidence metadata."""
        features = RegimeFeatures(
            quarter=state.quarter,
            time_remaining_seconds=state.time_remaining_seconds,
            possessions_remaining=state.possessions_remaining,
            score_margin=state.score_margin,
            quarter_scoring_rate=state.quarter_scoring_rate,
            baseline_scoring_rate=state.baseline_scoring_rate,
            foul_risk=state.foul_risk,
            trading_threes_risk=state.trading_threes_risk,
            timeout_cluster_risk=state.timeout_cluster_risk,
        )
        regime = classify_regime(features)

        compression_score = self._compression_score(state, regime)
        chaos_score = self._chaos_score(state)
        runway_over_pressure = self._runway_over_pressure(state, regime)
        total_decision_score = state.base_edge + compression_score - chaos_score + runway_over_pressure

        action = TotalsAction.PASS
        if total_decision_score >= 0.20:
            action = TotalsAction.BET_UNDER
        elif total_decision_score <= -0.20:
            action = TotalsAction.BET_OVER

        confidence = max(0.0, min(0.95, 0.5 + abs(total_decision_score) * 0.4))
        tier = self._tier_from_confidence(confidence, action)
        guardrail_reasons: List[str] = []

        if action == TotalsAction.BET_UNDER:
            context = UnderConfidenceContext(
                regime=regime.value,
                possessions_remaining=state.possessions_remaining,
                over_required_rate=state.over_required_rate,
                burst_persistence_active=self._burst_persistence_active(state),
                reachable_over_rate_band=self.guardrails.reachable_over_rate_band,
                high_possessions_threshold=self.guardrails.high_possessions_threshold,
            )
            adjusted = self.confidence_scorer.adjust_under_confidence(confidence, tier, context)
            confidence = adjusted.confidence
            tier = adjusted.tier
            guardrail_reasons.extend(adjusted.reasons)
            if tier == "PASS":
                action = TotalsAction.PASS

        meta = DecisionMeta(
            regime=regime.value,
            score=total_decision_score,
            compression_score=compression_score,
            chaos_score=chaos_score,
            runway_over_pressure=runway_over_pressure,
            over_required_rate=state.over_required_rate,
            tier=tier,
            guardrails=guardrail_reasons,
        )
        return LiveTotalsDecision(action=action.value, confidence=confidence, meta=meta)

    def _compression_score(self, state: LiveTotalsState, regime: Regime) -> float:
        time_factor = 1.0 - min(1.0, state.time_remaining_seconds / 720.0)
        margin_factor = min(1.0, abs(state.score_margin) / 20.0)
        possessions_factor = 1.0 - min(1.0, state.possessions_remaining / 14.0)
        leader_factor = max(0.0, min(1.0, state.leader_clock_pressure))
        raw = 0.42 * time_factor + 0.23 * margin_factor + 0.2 * possessions_factor + 0.15 * leader_factor
        if regime == Regime.RUNWAY:
            raw *= 0.7
        return max(0.0, min(1.0, raw))

    def _chaos_score(self, state: LiveTotalsState) -> float:
        raw = (state.foul_risk + state.trading_threes_risk + state.timeout_cluster_risk) / 3.0
        if state.time_remaining_seconds <= 120 and abs(state.score_margin) <= 8:
            raw += 0.15
        return max(0.0, min(1.0, raw))

    def _runway_over_pressure(self, state: LiveTotalsState, regime: Regime) -> float:
        safe_baseline = max(state.baseline_scoring_rate, 0.1)
        possession_pressure = min(1.0, state.possessions_remaining / 14.0)
        reachable_rate = max(0.0, 1.0 - (state.over_required_rate / (safe_baseline * 1.35)))
        time_pressure = min(1.0, state.time_remaining_seconds / 720.0)
        runway_strength = 0.45 * possession_pressure + 0.35 * reachable_rate + 0.20 * time_pressure
        if regime == Regime.COMPRESSION:
            runway_strength *= 0.55
        return -runway_strength

    def _burst_persistence_active(self, state: LiveTotalsState) -> bool:
        if state.burst_persistence_active:
            return True
        if state.recent_window_seconds is None or state.recent_scoring_rate is None:
            return False
        if state.recent_window_seconds > self.guardrails.burst_window_seconds:
            return False
        baseline = max(state.baseline_scoring_rate, 0.1)
        return state.recent_scoring_rate >= baseline * self.guardrails.burst_rate_multiplier

    def _tier_from_confidence(self, confidence: float, action: TotalsAction) -> str:
        if action == TotalsAction.PASS:
            return "PASS"
        if confidence >= self.guardrails.under_strong_threshold:
            return "STRONG"
        if confidence >= self.guardrails.under_medium_threshold:
            return "MEDIUM"
        if confidence >= self.guardrails.under_lean_threshold:
            return "LEAN"
        return "PASS"
