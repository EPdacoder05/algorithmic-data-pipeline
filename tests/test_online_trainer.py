"""
Tests for online_trainer.py
=============================
Validates partial_fit updates, confidence gating, prediction routing,
metrics tracking, and model persistence round-trip.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.feature_engineering import FeatureEngineer
from src.io_schemas import (
    DecisionType,
    FeedbackSentiment,
    Outcome,
    SessionSummary,
    TrainingExample,
)
from src.online_trainer import OnlineTrainer


def _make_example(label: str = "PASS", weight: float = 1.0) -> TrainingExample:
    """Create a minimal TrainingExample with a trivial feature vector."""
    fe = FeatureEngineer()
    from src.io_schemas import DecisionPoint
    from src.session_summarizer import SessionSummarizer
    from src.token_tracker import TokenTracker

    summary = SessionSummary(
        session_id="trainer_test",
        league="WNBA",
        market_type="Q2_total",
        total_tokens=100,
        final_decision=label,
        feedback_sentiment=FeedbackSentiment.NEUTRAL,
    )
    example = fe.transform(summary)
    example.label = label
    example.weight = weight
    return example


def _make_examples(n: int = 10) -> list[TrainingExample]:
    """Create a balanced set of training examples for smoke tests."""
    labels = [DecisionType.BET_OVER, DecisionType.BET_UNDER, DecisionType.PASS]
    return [_make_example(labels[i % 3]) for i in range(n)]


class TestOnlineTrainerInit:
    def test_init_creates_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            trainer = OnlineTrainer(model_dir=model_dir)
            assert model_dir.exists()

    def test_initial_state_unfitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            assert not trainer._is_fitted

    def test_predict_before_fit_returns_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            fe = FeatureEngineer()
            x = np.zeros(len(fe.feature_names), dtype=np.float32)
            action, prob = trainer.predict(x)
            assert action == "PASS"
            assert prob == 0.0


class TestOnlineTrainerUpdate:
    def test_update_single_example_fits_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.update([_make_example("PASS")])
            assert trainer._is_fitted

    def test_update_increments_n_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.update(_make_examples(6))
            assert trainer._n_examples == 6

    def test_update_empty_list_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.update([])
            assert not trainer._is_fitted

    def test_multiple_update_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.update(_make_examples(6))
            trainer.update(_make_examples(6))
            assert trainer._n_examples == 12

    def test_update_persists_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            trainer = OnlineTrainer(model_dir=model_dir)
            trainer.update(_make_examples(6))
            model_files = list(model_dir.glob("*.joblib"))
            assert len(model_files) >= 1


class TestOnlineTrainerPredict:
    def test_predict_returns_valid_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp), confidence_threshold=0.0)
            trainer.update(_make_examples(9))
            fe = FeatureEngineer()
            x = np.zeros(len(fe.feature_names), dtype=np.float32)
            action, prob = trainer.predict(x)
            assert action in ("BET_OVER", "BET_UNDER", "PASS")
            assert 0.0 <= prob <= 1.0

    def test_low_confidence_forces_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Set threshold so high that any real prediction falls below it
            trainer = OnlineTrainer(model_dir=Path(tmp), confidence_threshold=0.999)
            trainer.update(_make_examples(9))
            fe = FeatureEngineer()
            x = np.zeros(len(fe.feature_names), dtype=np.float32)
            action, prob = trainer.predict(x)
            # Should be PASS because confidence < 0.999
            assert action == "PASS"


class TestOnlineTrainerMetrics:
    def test_metrics_before_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            metrics = trainer.compute_metrics()
            assert metrics.n_examples_seen == 0
            assert metrics.accuracy is None

    def test_record_outcome_updates_rolling(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.record_outcome("PASS", "PASS")
            trainer.record_outcome("BET_OVER", "PASS")
            metrics = trainer.compute_metrics()
            assert metrics.accuracy == pytest.approx(0.5)

    def test_segment_metrics_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.record_outcome("PASS", "PASS", segment_key="WNBA::Q2_total")
            trainer.record_outcome("BET_OVER", "PASS", segment_key="WNBA::Q2_total")
            metrics = trainer.compute_metrics()
            assert "WNBA::Q2_total" in metrics.segment_metrics
            assert metrics.segment_metrics["WNBA::Q2_total"]["total"] == 2.0

    def test_model_version_increments(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(model_dir=Path(tmp))
            trainer.update(_make_examples(6))
            v1 = trainer._version
            trainer.update(_make_examples(6))
            assert trainer._version > v1


class TestOnlineTrainerPersistence:
    def test_load_latest_on_init(self):
        """Trainer should resume from persisted state on re-init."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            t1 = OnlineTrainer(model_dir=model_dir)
            t1.update(_make_examples(9))
            n = t1._n_examples
            version = t1._version

            # New trainer instance in same dir should load the saved model
            t2 = OnlineTrainer(model_dir=model_dir)
            assert t2._n_examples == n
            assert t2._version == version
