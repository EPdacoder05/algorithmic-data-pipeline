"""
Feature Engineering
===================
Transforms a :class:`~src.io_schemas.SessionSummary` into a flat numeric
feature vector ready for sklearn's ``partial_fit``.

Feature groups
--------------
required_rate       -- pts/min threshold extracted from the last decision point.
game_state          -- time remaining (normalised), line value.
league_onehot       -- binary indicators for each known league.
market_type_onehot  -- binary indicators for each known market type.
decision_type       -- one-hot for the session's final decision.
sentiment           -- numeric encoding of user feedback sentiment.
token_stats         -- total tokens, cost estimate (log-scaled).

All missing values are replaced with a safe default (0.0) and the output dict
always contains the same keys in the same order, guaranteeing a stable feature
space across invocations.

Usage example::

    fe = FeatureEngineer()
    example = fe.transform(summary)          # → TrainingExample
    X = fe.to_array([example1, example2])    # → np.ndarray, shape (2, n_features)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from src.config import settings
from src.io_schemas import (
    DecisionType,
    FeedbackSentiment,
    Outcome,
    SessionSummary,
    TrainingExample,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment → float mapping
# ---------------------------------------------------------------------------
_SENTIMENT_SCORES: Dict[str, float] = {
    FeedbackSentiment.STRONG_NEGATIVE: -2.0,
    FeedbackSentiment.NEGATIVE: -1.0,
    FeedbackSentiment.NEUTRAL: 0.0,
    FeedbackSentiment.POSITIVE: 1.0,
    FeedbackSentiment.STRONG_POSITIVE: 2.0,
}

# Max quarter time in seconds (12-min quarter NBA/WNBA)
_MAX_QUARTER_SECONDS = 720.0

# Line normalisation reference (sensible upper bound for totals)
_LINE_NORM = 250.0

# Token cost log-scale cap
_LOG_TOKEN_CAP = 100_000.0


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    """Return *value* if it is a finite number, else *default*."""
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return float(value)


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """
    Builds a fixed-width, numeric feature vector from a
    :class:`~src.io_schemas.SessionSummary`.

    Parameters
    ----------
    known_leagues:
        Ordered list of league codes for one-hot encoding.  Defaults to
        ``settings.KNOWN_LEAGUES``.
    known_market_types:
        Ordered list of market type codes.  Defaults to
        ``settings.KNOWN_MARKET_TYPES``.
    """

    def __init__(
        self,
        known_leagues: Optional[List[str]] = None,
        known_market_types: Optional[List[str]] = None,
    ) -> None:
        self.known_leagues = known_leagues or list(settings.KNOWN_LEAGUES)
        self.known_market_types = known_market_types or list(settings.KNOWN_MARKET_TYPES)

    # ------------------------------------------------------------------ public

    @property
    def feature_names(self) -> List[str]:
        """Stable ordered list of feature names."""
        _dt_values = [dt.value for dt in (DecisionType.BET_OVER, DecisionType.BET_UNDER, DecisionType.PASS, DecisionType.UNKNOWN)]
        names = [
            "required_rate",
            "time_left_norm",
            "line_norm",
            "sentiment",
            "log_total_tokens",
            "log_total_cost",
        ]
        names += [f"decision_{v}" for v in _dt_values]
        names += [f"league_{lg}" for lg in self.known_leagues]
        names += [f"market_{mt}" for mt in self.known_market_types]
        return names

    def transform(self, summary: SessionSummary) -> TrainingExample:
        """
        Convert a :class:`SessionSummary` into a :class:`TrainingExample`.

        The label is taken from ``summary.final_decision``.  The weight is
        boosted for strongly negative feedback (high-signal correction events).
        """
        fv = self._build_feature_vector(summary)
        weight = self._compute_weight(summary)

        return TrainingExample(
            session_id=summary.session_id,
            feature_vector=fv,
            label=DecisionType(summary.final_decision),
            outcome=Outcome(summary.outcome),
            weight=weight,
            metadata={"summary_id": summary.summary_id},
        )

    def transform_many(self, summaries: List[SessionSummary]) -> List[TrainingExample]:
        """Batch transform a list of summaries."""
        return [self.transform(s) for s in summaries]

    def to_array(self, examples: List[TrainingExample]) -> np.ndarray:
        """
        Stack feature vectors into a 2-D numpy array.

        Rows correspond to examples; columns to :attr:`feature_names`.
        """
        if not examples:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        rows = [
            [_safe_float(ex.feature_vector.get(name, 0.0)) for name in self.feature_names]
            for ex in examples
        ]
        return np.array(rows, dtype=np.float32)

    def labels_array(self, examples: List[TrainingExample]) -> np.ndarray:
        """Return label strings as a 1-D numpy array."""
        return np.array([ex.label for ex in examples], dtype=object)

    # ----------------------------------------------------------------- private

    def _build_feature_vector(self, summary: SessionSummary) -> Dict[str, float]:
        """Build the flat feature dict for a summary."""
        fv: Dict[str, float] = {}

        # --- required_rate from last decision point with a threshold ---
        req_rate = 0.0
        for dp in reversed(summary.decision_points):
            if dp.threshold_used is not None:
                req_rate = _safe_float(dp.threshold_used)
                break
        fv["required_rate"] = req_rate

        # --- game state ---
        time_left = 0.0
        for dp in reversed(summary.decision_points):
            if dp.time_left_seconds is not None:
                time_left = _safe_float(dp.time_left_seconds)
                break
        fv["time_left_norm"] = min(1.0, time_left / _MAX_QUARTER_SECONDS)

        line = 0.0
        for dp in reversed(summary.decision_points):
            if dp.line_at_decision is not None:
                line = _safe_float(dp.line_at_decision)
                break
        if line == 0.0 and summary.metadata.get("last_line_snapshot"):
            line = _safe_float(summary.metadata["last_line_snapshot"])
        fv["line_norm"] = min(1.0, line / _LINE_NORM)

        # --- sentiment ---
        fv["sentiment"] = _SENTIMENT_SCORES.get(summary.feedback_sentiment, 0.0)

        # --- token stats (log-scaled to reduce magnitude) ---
        fv["log_total_tokens"] = math.log1p(
            min(_safe_float(summary.total_tokens), _LOG_TOKEN_CAP)
        )
        fv["log_total_cost"] = math.log1p(_safe_float(summary.total_cost_usd) * 1_000)

        # --- final decision one-hot ---
        for dt in (DecisionType.BET_OVER, DecisionType.BET_UNDER, DecisionType.PASS, DecisionType.UNKNOWN):
            key = f"decision_{dt.value}"
            fv[key] = 1.0 if summary.final_decision == dt.value else 0.0

        # --- league one-hot ---
        for lg in self.known_leagues:
            fv[f"league_{lg}"] = 1.0 if summary.league == lg else 0.0

        # --- market type one-hot ---
        for mt in self.known_market_types:
            fv[f"market_{mt}"] = 1.0 if summary.market_type == mt else 0.0

        return fv

    @staticmethod
    def _compute_weight(summary: SessionSummary) -> float:
        """
        Assign a sample weight.

        Strong-negative feedback = weight 3.0 (high-signal error correction).
        Negative feedback        = weight 2.0.
        Known outcome            = weight 1.5.
        Default                  = 1.0.
        """
        weight = 1.0
        if summary.feedback_sentiment == FeedbackSentiment.STRONG_NEGATIVE:
            weight = 3.0
        elif summary.feedback_sentiment == FeedbackSentiment.NEGATIVE:
            weight = 2.0
        if summary.outcome in (Outcome.WIN, Outcome.LOSS):
            weight = max(weight, 1.5)
        return weight
