"""
Tests for feature_engineering.py
===================================
Validates feature vector stability, missing-value handling, one-hot encoding,
and the full summary → TrainingExample transform.
"""

import math
import pytest
import numpy as np

from src.feature_engineering import FeatureEngineer, _safe_float
from src.io_schemas import (
    DecisionPoint,
    DecisionType,
    FeedbackSentiment,
    Outcome,
    SessionSummary,
    TrainingExample,
)


def _make_summary(**overrides) -> SessionSummary:
    """Helper: build a minimal SessionSummary for testing."""
    base = dict(
        session_id="feat_test",
        league="WNBA",
        market_type="Q2_total",
        total_messages=5,
        total_tokens=200,
        total_cost_usd=0.0002,
        final_decision=DecisionType.PASS,
        outcome=Outcome.UNKNOWN,
        feedback_sentiment=FeedbackSentiment.NEUTRAL,
    )
    base.update(overrides)
    return SessionSummary(**base)


class TestSafeFloat:
    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self):
        assert _safe_float(float("inf")) == 0.0

    def test_valid_returns_value(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_custom_default(self):
        assert _safe_float(None, default=-1.0) == -1.0


class TestFeatureEngineer:
    """Core tests for FeatureEngineer."""

    def setup_method(self):
        self.fe = FeatureEngineer()

    def test_feature_names_stable(self):
        names1 = self.fe.feature_names
        names2 = self.fe.feature_names
        assert names1 == names2

    def test_feature_names_contains_expected_keys(self):
        names = self.fe.feature_names
        assert "required_rate" in names
        assert "time_left_norm" in names
        assert "sentiment" in names
        assert "log_total_tokens" in names
        assert "decision_PASS" in names
        assert "league_WNBA" in names
        assert "market_Q2_total" in names

    def test_transform_returns_training_example(self):
        summary = _make_summary()
        example = self.fe.transform(summary)
        assert isinstance(example, TrainingExample)
        assert example.session_id == "feat_test"

    def test_feature_vector_has_all_names(self):
        summary = _make_summary()
        example = self.fe.transform(summary)
        for name in self.fe.feature_names:
            assert name in example.feature_vector, f"Missing feature: {name}"

    def test_no_nan_in_feature_vector(self):
        summary = _make_summary()
        example = self.fe.transform(summary)
        for k, v in example.feature_vector.items():
            assert not math.isnan(v), f"NaN in feature: {k}"
            assert not math.isinf(v), f"Inf in feature: {k}"

    def test_league_onehot_wnba(self):
        summary = _make_summary(league="WNBA")
        example = self.fe.transform(summary)
        assert example.feature_vector["league_WNBA"] == 1.0
        assert example.feature_vector["league_NBA"] == 0.0

    def test_league_onehot_nba(self):
        summary = _make_summary(league="NBA")
        example = self.fe.transform(summary)
        assert example.feature_vector["league_NBA"] == 1.0
        assert example.feature_vector["league_WNBA"] == 0.0

    def test_market_type_onehot(self):
        summary = _make_summary(market_type="Q2_total")
        example = self.fe.transform(summary)
        assert example.feature_vector["market_Q2_total"] == 1.0
        assert example.feature_vector["market_game_total"] == 0.0

    def test_decision_onehot_pass(self):
        summary = _make_summary(final_decision=DecisionType.PASS)
        example = self.fe.transform(summary)
        assert example.feature_vector["decision_PASS"] == 1.0
        assert example.feature_vector["decision_BET_OVER"] == 0.0

    def test_decision_onehot_bet_over(self):
        summary = _make_summary(final_decision=DecisionType.BET_OVER)
        example = self.fe.transform(summary)
        assert example.feature_vector["decision_BET_OVER"] == 1.0
        assert example.feature_vector["decision_PASS"] == 0.0

    def test_sentiment_strong_negative_score(self):
        summary = _make_summary(feedback_sentiment=FeedbackSentiment.STRONG_NEGATIVE)
        example = self.fe.transform(summary)
        assert example.feature_vector["sentiment"] == -2.0

    def test_sentiment_positive_score(self):
        summary = _make_summary(feedback_sentiment=FeedbackSentiment.POSITIVE)
        example = self.fe.transform(summary)
        assert example.feature_vector["sentiment"] == 1.0

    def test_weight_strong_negative_boosted(self):
        summary = _make_summary(feedback_sentiment=FeedbackSentiment.STRONG_NEGATIVE)
        example = self.fe.transform(summary)
        assert example.weight == 3.0

    def test_weight_known_outcome_boosted(self):
        summary = _make_summary(outcome=Outcome.WIN)
        example = self.fe.transform(summary)
        assert example.weight >= 1.5

    def test_weight_default(self):
        summary = _make_summary()
        example = self.fe.transform(summary)
        assert example.weight == 1.0

    def test_required_rate_from_decision_point(self):
        dp = DecisionPoint(turn_index=0, threshold_used=5.13)
        summary = _make_summary(decision_points=[dp])
        example = self.fe.transform(summary)
        assert example.feature_vector["required_rate"] == pytest.approx(5.13)

    def test_time_left_normalised(self):
        dp = DecisionPoint(turn_index=0, time_left_seconds=269.0)  # 4:29
        summary = _make_summary(decision_points=[dp])
        example = self.fe.transform(summary)
        expected = min(1.0, 269.0 / 720.0)
        assert example.feature_vector["time_left_norm"] == pytest.approx(expected)

    def test_time_left_capped_at_one(self):
        dp = DecisionPoint(turn_index=0, time_left_seconds=9999.0)
        summary = _make_summary(decision_points=[dp])
        example = self.fe.transform(summary)
        assert example.feature_vector["time_left_norm"] <= 1.0

    def test_log_total_tokens_positive(self):
        summary = _make_summary(total_tokens=500)
        example = self.fe.transform(summary)
        assert example.feature_vector["log_total_tokens"] > 0.0

    def test_transform_many(self):
        summaries = [_make_summary(session_id=f"s{i}") for i in range(5)]
        examples = self.fe.transform_many(summaries)
        assert len(examples) == 5

    def test_to_array_shape(self):
        summaries = [_make_summary(session_id=f"s{i}") for i in range(3)]
        examples = self.fe.transform_many(summaries)
        X = self.fe.to_array(examples)
        assert X.shape == (3, len(self.fe.feature_names))
        assert X.dtype == np.float32

    def test_to_array_empty(self):
        X = self.fe.to_array([])
        assert X.shape == (0, len(self.fe.feature_names))

    def test_labels_array(self):
        examples = [
            TrainingExample(
                session_id="x",
                feature_vector={"x": 1.0},
                label=DecisionType.BET_OVER,
            ),
            TrainingExample(
                session_id="x",
                feature_vector={"x": 0.0},
                label=DecisionType.PASS,
            ),
        ]
        y = self.fe.labels_array(examples)
        assert y[0] == DecisionType.BET_OVER
        assert y[1] == DecisionType.PASS

    def test_missing_decision_points_safe(self):
        """Summarizer with no decision points should produce a valid example."""
        summary = _make_summary(decision_points=[])
        example = self.fe.transform(summary)
        assert example.feature_vector["required_rate"] == 0.0
        assert example.feature_vector["time_left_norm"] == 0.0
