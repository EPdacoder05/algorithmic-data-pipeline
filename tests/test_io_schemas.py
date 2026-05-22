"""
Tests for io_schemas.py
========================
Validates Pydantic model construction, field constraints, and validation helpers.
"""

import math
import pytest
from pydantic import ValidationError

from src.io_schemas import (
    DecisionType,
    FeedbackSentiment,
    ModelMetrics,
    Outcome,
    Role,
    SessionSummary,
    TokenEvent,
    TrainingExample,
    validate_model_metrics,
    validate_session_summary,
    validate_token_event,
    validate_training_example,
)


class TestTokenEvent:
    """Schema validation tests for TokenEvent."""

    def _valid_data(self, **overrides):
        base = {
            "session_id": "sess_001",
            "message_id": 0,
            "role": "user",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        base.update(overrides)
        return base

    def test_valid_event_constructs(self):
        evt = TokenEvent(**self._valid_data())
        assert evt.session_id == "sess_001"
        assert evt.total_tokens == 15

    def test_total_tokens_mismatch_raises(self):
        with pytest.raises(ValidationError):
            TokenEvent(**self._valid_data(total_tokens=99))

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValidationError):
            TokenEvent(**self._valid_data(prompt_tokens=-1, total_tokens=4))

    def test_validate_helper_parses_dict(self):
        evt = validate_token_event(self._valid_data())
        assert isinstance(evt, TokenEvent)

    def test_role_enum_values(self):
        for role in ("user", "assistant", "system"):
            evt = TokenEvent(**self._valid_data(role=role))
            assert evt.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            TokenEvent(**self._valid_data(role="bot"))

    def test_optional_fields_default(self):
        evt = TokenEvent(**self._valid_data())
        assert evt.cost_usd == 0.0
        assert evt.market_context is None
        assert evt.line_snapshot is None
        assert evt.metadata == {}

    def test_event_id_is_unique(self):
        e1 = TokenEvent(**self._valid_data())
        e2 = TokenEvent(**self._valid_data())
        assert e1.event_id != e2.event_id


class TestSessionSummary:
    """Schema validation tests for SessionSummary."""

    def test_defaults(self):
        s = SessionSummary(session_id="s1")
        assert s.league == "UNKNOWN"
        assert s.total_tokens == 0
        assert s.outcome == Outcome.UNKNOWN
        assert s.feedback_sentiment == FeedbackSentiment.NEUTRAL

    def test_validate_helper(self):
        s = validate_session_summary({"session_id": "s2", "league": "NBA"})
        assert s.league == "NBA"

    def test_invalid_schema_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            validate_session_summary({"total_messages": -1})


class TestTrainingExample:
    """Schema validation tests for TrainingExample."""

    def _valid(self, **overrides):
        base = {
            "session_id": "s1",
            "feature_vector": {"required_rate": 5.1, "time_left_norm": 0.5},
            "label": "BET_OVER",
        }
        base.update(overrides)
        return base

    def test_valid_example(self):
        ex = TrainingExample(**self._valid())
        assert ex.label == DecisionType.BET_OVER

    def test_nan_feature_raises(self):
        with pytest.raises(ValidationError):
            TrainingExample(**self._valid(feature_vector={"x": float("nan")}))

    def test_inf_feature_raises(self):
        with pytest.raises(ValidationError):
            TrainingExample(**self._valid(feature_vector={"x": float("inf")}))

    def test_weight_positive(self):
        ex = TrainingExample(**self._valid(weight=2.5))
        assert ex.weight == 2.5

    def test_weight_zero_raises(self):
        with pytest.raises(ValidationError):
            TrainingExample(**self._valid(weight=0.0))


class TestModelMetrics:
    """Schema validation tests for ModelMetrics."""

    def test_defaults(self):
        m = ModelMetrics(model_version="v1")
        assert m.n_examples_seen == 0
        assert m.accuracy is None

    def test_validate_helper(self):
        m = validate_model_metrics({"model_version": "v2", "n_examples_seen": 100})
        assert m.n_examples_seen == 100
