"""Tests for universal all-quarter regime-based live totals classifier."""

from engine.live_totals_classifier import LiveTotalsClassifier, LiveTotalsState
from engine.regime import Regime, RegimeFeatures, classify_regime


def test_regime_detection_runway_neutral_compression():
    runway = RegimeFeatures(
        quarter=2,
        time_remaining_seconds=610,
        possessions_remaining=13,
        score_margin=4,
        quarter_scoring_rate=5.6,
        baseline_scoring_rate=5.2,
    )
    neutral = RegimeFeatures(
        quarter=2,
        time_remaining_seconds=300,
        possessions_remaining=9,
        score_margin=10,
        quarter_scoring_rate=4.9,
        baseline_scoring_rate=5.0,
    )
    compression = RegimeFeatures(
        quarter=4,
        time_remaining_seconds=95,
        possessions_remaining=4,
        score_margin=12,
        quarter_scoring_rate=4.1,
        baseline_scoring_rate=5.0,
    )

    assert classify_regime(runway) == Regime.RUNWAY
    assert classify_regime(neutral) == Regime.NEUTRAL
    assert classify_regime(compression) == Regime.COMPRESSION


def test_guardrail_high_possessions_no_strong_under():
    classifier = LiveTotalsClassifier()
    decision = classifier.classify(
        LiveTotalsState(
            quarter=2,
            time_remaining_seconds=320,
            possessions_remaining=11,
            score_margin=14,
            quarter_scoring_rate=4.0,
            baseline_scoring_rate=5.2,
            current_total=148.0,
            posted_total_line=202.5,
            base_edge=0.9,
            leader_clock_pressure=0.9,
        )
    )

    assert decision.action == "BET_UNDER"
    assert decision.meta.tier != "STRONG"
    assert "high_possessions_no_strong_under" in decision.meta.guardrails


def test_guardrail_reachable_over_rate_clamps_under_confidence():
    classifier = LiveTotalsClassifier()
    decision = classifier.classify(
        LiveTotalsState(
            quarter=3,
            time_remaining_seconds=360,
            possessions_remaining=9,
            score_margin=11,
            quarter_scoring_rate=4.2,
            baseline_scoring_rate=5.0,
            current_total=172.0,
            posted_total_line=202.0,
            base_edge=0.85,
            leader_clock_pressure=0.8,
        )
    )

    assert decision.meta.over_required_rate >= 4.8
    assert decision.meta.over_required_rate <= 5.8
    assert decision.meta.tier == "LEAN"
    assert "reachable_over_rate_clamp" in decision.meta.guardrails


def test_burst_persistence_blocks_fresh_under_entries():
    classifier = LiveTotalsClassifier()
    decision = classifier.classify(
        LiveTotalsState(
            quarter=4,
            time_remaining_seconds=150,
            possessions_remaining=6,
            score_margin=10,
            quarter_scoring_rate=4.3,
            baseline_scoring_rate=4.9,
            current_total=186.0,
            posted_total_line=216.5,
            base_edge=1.1,
            recent_window_seconds=60,
            recent_scoring_rate=6.2,
            leader_clock_pressure=0.95,
        )
    )

    assert decision.action == "PASS"
    assert decision.meta.tier == "PASS"
    assert "burst_persistence_blocks_under" in decision.meta.guardrails


def test_action_output_is_constrained():
    classifier = LiveTotalsClassifier()
    actions = {
        classifier.classify(
            LiveTotalsState(
                quarter=1,
                time_remaining_seconds=680,
                possessions_remaining=15,
                score_margin=1,
                quarter_scoring_rate=6.0,
                baseline_scoring_rate=5.2,
                current_total=31.0,
                posted_total_line=54.5,
                base_edge=-0.8,
            )
        ).action,
        classifier.classify(
            LiveTotalsState(
                quarter=2,
                time_remaining_seconds=120,
                possessions_remaining=4,
                score_margin=16,
                quarter_scoring_rate=3.7,
                baseline_scoring_rate=5.0,
                current_total=85.0,
                posted_total_line=106.5,
                base_edge=0.9,
                leader_clock_pressure=1.0,
            )
        ).action,
    }

    assert actions.issubset({"BET_OVER", "BET_UNDER", "PASS"})


def test_regression_naive_under_bias_is_rejected_in_runway_state():
    classifier = LiveTotalsClassifier()
    decision = classifier.classify(
        LiveTotalsState(
            quarter=3,
            time_remaining_seconds=560,
            possessions_remaining=13,
            score_margin=3,
            quarter_scoring_rate=4.6,
            baseline_scoring_rate=5.0,
            current_total=151.0,
            posted_total_line=197.0,
            base_edge=0.55,
            foul_risk=0.55,
            trading_threes_risk=0.45,
            timeout_cluster_risk=0.45,
        )
    )

    assert decision.meta.regime == "RUNWAY"
    assert decision.action in {"BET_OVER", "PASS"}
    assert not (decision.action == "BET_UNDER" and decision.meta.tier in {"MEDIUM", "STRONG"})
