"""
Multi-Signal Confidence Scorer
================================
Combines multiple RLM signals into an overall confidence score and tier classification.

Tiers:
- TIER 1: 85%+ confidence → Full position
- TIER 2: 75%+ confidence → Partial position  
- LEAN: 60%+ confidence → Small position or watch
- PASS: <60% confidence → No bet
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from analysis.rlm_detector import RLMSignal


@dataclass
class ConfidenceScore:
    """Overall confidence assessment for a pick."""
    confidence: float  # 0.0 to 1.0
    tier: str  # "TIER_1", "TIER_2", "LEAN", "PASS"
    signals: List[str]  # List of signal types that fired
    signal_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "tier": self.tier,
            "signals": self.signals,
            "signal_count": self.signal_count,
        }


@dataclass
class UnderConfidenceContext:
    """Context for live-total under confidence guardrails."""

    regime: str
    possessions_remaining: float
    over_required_rate: float
    burst_persistence_active: bool
    reachable_over_rate_band: Tuple[float, float]
    high_possessions_threshold: float


@dataclass
class AdjustedConfidence:
    """Result of applying under-specific confidence guardrails."""

    confidence: float
    tier: str
    reasons: List[str]


class ConfidenceScorer:
    """
    Multi-signal confidence scoring engine.
    
    Strategy:
    - Need at least 2 signals to generate a pick
    - Average the confidence scores from all signals
    - Apply tier thresholds
    """
    
    def __init__(
        self,
        tier1_threshold: float = 0.85,
        tier2_threshold: float = 0.75,
        lean_threshold: float = 0.60,
        min_signals: int = 2
    ):
        """
        Args:
            tier1_threshold: Minimum confidence for TIER 1 (default 85%)
            tier2_threshold: Minimum confidence for TIER 2 (default 75%)
            lean_threshold: Minimum confidence for LEAN (default 60%)
            min_signals: Minimum number of signals required (default 2)
        """
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.lean_threshold = lean_threshold
        self.min_signals = min_signals
    
    def score(self, signals: List[RLMSignal]) -> ConfidenceScore:
        """
        Calculate overall confidence from multiple signals.
        
        Args:
            signals: List of RLMSignal objects (only detected=True signals)
        
        Returns:
            ConfidenceScore with tier classification
        """
        # Filter to only detected signals
        detected_signals = [s for s in signals if s.detected]
        
        if len(detected_signals) < self.min_signals:
            return ConfidenceScore(
                confidence=0.0,
                tier="PASS",
                signals=[],
                signal_count=len(detected_signals)
            )
        
        # Calculate average confidence from all signals
        # Weight stronger signals more heavily
        weighted_sum = 0.0
        weight_total = 0.0
        
        for signal in detected_signals:
            # Weight by signal confidence itself
            weight = signal.confidence
            weighted_sum += signal.confidence * weight
            weight_total += weight
        
        if weight_total == 0:
            confidence = 0.0
        else:
            confidence = weighted_sum / weight_total
        
        # Determine tier
        if confidence >= self.tier1_threshold:
            tier = "TIER_1"
        elif confidence >= self.tier2_threshold:
            tier = "TIER_2"
        elif confidence >= self.lean_threshold:
            tier = "LEAN"
        else:
            tier = "PASS"
        
        signal_types = [s.signal_type for s in detected_signals]
        
        return ConfidenceScore(
            confidence=confidence,
            tier=tier,
            signals=signal_types,
            signal_count=len(detected_signals)
        )
    
    def score_with_boost(
        self,
        primary_signals: List[RLMSignal],
        confirmation_signals: List[RLMSignal]
    ) -> ConfidenceScore:
        """
        Score with separate primary and confirmation signals.
        
        Primary signals are required triggers.
        Confirmation signals boost confidence but can't trigger alone.
        
        Args:
            primary_signals: List of primary RLMSignal objects
            confirmation_signals: List of confirmation RLMSignal objects
        
        Returns:
            ConfidenceScore with tier classification
        """
        # Filter to detected signals
        detected_primary = [s for s in primary_signals if s.detected]
        detected_confirmation = [s for s in confirmation_signals if s.detected]
        
        # Need at least one primary signal
        if len(detected_primary) == 0:
            return ConfidenceScore(
                confidence=0.0,
                tier="PASS",
                signals=[],
                signal_count=0
            )
        
        # Start with average of primary signals
        primary_confidence = sum(s.confidence for s in detected_primary) / len(detected_primary)
        
        # Boost by confirmation signals (diminishing returns)
        # First confirmation: +5%, second: +2.5%, third: +1.67%, etc.
        # This prevents confirmation signals from overwhelming primary signals
        confirmation_boost = 0.0
        for i, signal in enumerate(detected_confirmation):
            # First confirmation: +5%, second: +3%, third: +2%, etc.
            boost_factor = 0.05 / (i + 1)
            confirmation_boost += signal.confidence * boost_factor
        
        # Cap confirmation boost at +10%
        confirmation_boost = min(confirmation_boost, 0.10)
        
        # Final confidence (capped at 95%)
        confidence = min(0.95, primary_confidence + confirmation_boost)
        
        # Determine tier
        if confidence >= self.tier1_threshold:
            tier = "TIER_1"
        elif confidence >= self.tier2_threshold:
            tier = "TIER_2"
        elif confidence >= self.lean_threshold:
            tier = "LEAN"
        else:
            tier = "PASS"
        
        all_signals = detected_primary + detected_confirmation
        signal_types = [s.signal_type for s in all_signals]
        
        return ConfidenceScore(
            confidence=confidence,
            tier=tier,
            signals=signal_types,
            signal_count=len(all_signals)
        )

    def adjust_under_confidence(
        self,
        confidence: float,
        tier: str,
        context: UnderConfidenceContext,
    ) -> AdjustedConfidence:
        """Clamp under confidence based on runway and hard guardrails."""
        adjusted_confidence = confidence
        adjusted_tier = tier
        reasons: List[str] = []

        if context.regime == "RUNWAY" and adjusted_tier in {"MEDIUM", "STRONG", "TIER_1", "TIER_2"}:
            adjusted_confidence = min(adjusted_confidence, self.lean_threshold + 0.01)
            adjusted_tier = "LEAN"
            reasons.append("runway_caps_under_confidence")

        if context.possessions_remaining >= context.high_possessions_threshold and adjusted_tier == "STRONG":
            adjusted_confidence = min(adjusted_confidence, self.lean_threshold + 0.01)
            adjusted_tier = "LEAN"
            reasons.append("high_possessions_no_strong_under")

        lo, hi = context.reachable_over_rate_band
        if lo <= context.over_required_rate <= hi and adjusted_tier in {"MEDIUM", "STRONG", "TIER_1", "TIER_2"}:
            adjusted_confidence = min(adjusted_confidence, self.lean_threshold + 0.01)
            adjusted_tier = "LEAN"
            reasons.append("reachable_over_rate_clamp")

        if context.burst_persistence_active:
            adjusted_confidence = min(adjusted_confidence, max(0.0, self.lean_threshold - 0.01))
            adjusted_tier = "PASS"
            reasons.append("burst_persistence_blocks_under")

        return AdjustedConfidence(
            confidence=adjusted_confidence,
            tier=adjusted_tier,
            reasons=reasons,
        )
