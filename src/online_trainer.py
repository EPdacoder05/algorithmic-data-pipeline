"""
Online Trainer
==============
Maintains an :class:`sklearn.linear_model.SGDClassifier` that is updated via
``partial_fit`` after each new session summary is labelled.

Design decisions
----------------
* SGDClassifier with ``loss="log_loss"`` provides calibrated probabilities via
  ``predict_proba`` without requiring a full refit.
* The model is persisted to disk with ``joblib`` after every update so that
  state is never lost between process restarts.
* A confidence gate converts low-probability predictions to ``PASS``.
* Segment-level metrics (by league + market_type) track performance separately.
* Model versions are numbered sequentially (v1, v2, …) so any version can be
  rolled back.

Usage example::

    fe = FeatureEngineer()
    trainer = OnlineTrainer()

    for summary in new_summaries:
        example = fe.transform(summary)
        trainer.update([example])

    action, prob = trainer.predict(fe.to_array([example])[0])
    metrics = trainer.compute_metrics()
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.io_schemas import DecisionType, ModelMetrics, TrainingExample

logger = logging.getLogger(__name__)

# All three possible action labels — order matters for LabelEncoder
_CLASSES = ["BET_OVER", "BET_UNDER", "PASS"]


class OnlineTrainer:
    """
    Online classifier that wraps :class:`sklearn.linear_model.SGDClassifier`.

    Parameters
    ----------
    model_dir:
        Directory where serialized models are written.  Defaults to
        ``settings.MODEL_DIR``.
    confidence_threshold:
        Minimum ``predict_proba`` value for a non-PASS action.  Defaults to
        ``settings.CONFIDENCE_THRESHOLD``.
    learning_rate:
        SGD learning rate.  Defaults to ``settings.LEARNING_RATE``.
    accuracy_window:
        Rolling window size for accuracy tracking.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
        learning_rate: Optional[float] = None,
        accuracy_window: int = 50,
    ) -> None:
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.learning_rate = learning_rate or settings.LEARNING_RATE
        self.accuracy_window = accuracy_window

        # Classifier
        self._clf = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=self.learning_rate,
            warm_start=True,
            random_state=42,
            max_iter=1,
        )
        self._is_fitted = False
        self._n_examples = 0
        self._version = 1

        # Rolling accuracy window (stores 1 if correct, 0 if wrong)
        self._rolling_correct: Deque[int] = deque(maxlen=accuracy_window)

        # Segment-level counters  {segment_key: {"correct": int, "total": int}}
        self._segment_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

        # Attempt to load latest persisted model on startup
        self._try_load_latest()

    # ------------------------------------------------------------------ public

    def update(self, examples: List[TrainingExample]) -> None:
        """
        Run one ``partial_fit`` step with the supplied examples.

        Parameters
        ----------
        examples:
            List of labelled :class:`~src.io_schemas.TrainingExample` objects.
            Must contain at least one item.
        """
        if not examples:
            logger.warning("update() called with empty example list — skipping.")
            return

        fe = FeatureEngineer()
        X = fe.to_array(examples)
        y = np.array([e.label for e in examples], dtype=object)
        weights = np.array([e.weight for e in examples], dtype=float)

        # On the very first call we must pass all known classes so that the
        # classifier's internal class registry is initialised correctly.
        if not self._is_fitted:
            self._clf.partial_fit(X, y, classes=np.array(_CLASSES, dtype=object), sample_weight=weights)
            self._is_fitted = True
        else:
            self._clf.partial_fit(X, y, sample_weight=weights)

        self._n_examples += len(examples)
        logger.info(
            "partial_fit: %d new examples (total seen: %d)",
            len(examples),
            self._n_examples,
        )
        self._version += 1
        self._persist()

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        """
        Predict the recommended action for a single feature vector.

        Returns
        -------
        (action, probability)
            *action* is one of ``BET_OVER``, ``BET_UNDER``, ``PASS``.
            *probability* is the model's confidence in the chosen action.

        If the model is not yet fitted, ``("PASS", 0.0)`` is returned.
        """
        if not self._is_fitted:
            return "PASS", 0.0

        x2d = x.reshape(1, -1)
        proba = self._clf.predict_proba(x2d)[0]
        classes = self._clf.classes_
        best_idx = int(np.argmax(proba))
        best_action = str(classes[best_idx])
        best_prob = float(proba[best_idx])

        if best_prob < self.confidence_threshold:
            return "PASS", best_prob

        return best_action, best_prob

    def record_outcome(
        self,
        predicted: str,
        actual: str,
        segment_key: Optional[str] = None,
    ) -> None:
        """
        Record whether a prediction was correct for rolling accuracy tracking.

        Parameters
        ----------
        predicted:
            The action that was predicted (output of :meth:`predict`).
        actual:
            The true label (ground truth from settlement).
        segment_key:
            Optional segment identifier (e.g. ``"WNBA::Q2_total"``).
        """
        correct = int(predicted == actual)
        self._rolling_correct.append(correct)
        if segment_key:
            self._segment_stats[segment_key]["correct"] += correct
            self._segment_stats[segment_key]["total"] += 1

    def compute_metrics(self) -> ModelMetrics:
        """Return a :class:`~src.io_schemas.ModelMetrics` snapshot."""
        accuracy: Optional[float] = None
        if self._rolling_correct:
            accuracy = round(sum(self._rolling_correct) / len(self._rolling_correct), 4)

        class_counts: Dict[str, int] = {}
        if self._is_fitted:
            for cls, count in zip(
                self._clf.classes_,
                np.bincount(
                    np.searchsorted(self._clf.classes_, self._clf.classes_),
                    minlength=len(self._clf.classes_),
                ),
            ):
                class_counts[str(cls)] = int(count)

        segment_metrics: Dict[str, Dict[str, float]] = {}
        for seg, stats in self._segment_stats.items():
            total = stats["total"]
            segment_metrics[seg] = {
                "accuracy": round(stats["correct"] / total, 4) if total else 0.0,
                "total": float(total),
            }

        return ModelMetrics(
            model_version=f"{settings.MODEL_VERSION_PREFIX}{self._version}",
            n_examples_seen=self._n_examples,
            accuracy=accuracy,
            accuracy_window=self.accuracy_window,
            class_counts=class_counts,
            segment_metrics=segment_metrics,
        )

    # ---------------------------------------------------------------- persist

    def _model_path(self, version: Optional[int] = None) -> Path:
        v = version or self._version
        return self.model_dir / f"online_trainer_{settings.MODEL_VERSION_PREFIX}{v}.joblib"

    def _persist(self) -> None:
        """Save the current model state to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self._model_path()
        state = {
            "clf": self._clf,
            "is_fitted": self._is_fitted,
            "n_examples": self._n_examples,
            "version": self._version,
            "rolling_correct": list(self._rolling_correct),
            "segment_stats": dict(self._segment_stats),
        }
        joblib.dump(state, path)
        logger.info("Model persisted → %s", path)

    def _try_load_latest(self) -> None:
        """Load the highest-versioned model from disk if one exists."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(self.model_dir.glob("online_trainer_*.joblib"))
        if not candidates:
            return
        path = candidates[-1]
        try:
            state = joblib.load(path)
            self._clf = state["clf"]
            self._is_fitted = state["is_fitted"]
            self._n_examples = state["n_examples"]
            self._version = state["version"]
            self._rolling_correct = deque(state["rolling_correct"], maxlen=self.accuracy_window)
            self._segment_stats = defaultdict(
                lambda: {"correct": 0, "total": 0},
                state["segment_stats"],
            )
            logger.info("Loaded model from %s (version %d)", path, self._version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load model from %s: %s — starting fresh.", path, exc)
