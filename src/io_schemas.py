"""
I/O Schemas
===========
Pydantic v2 data-contracts shared across all pipeline modules.

Contracts
---------
TokenEvent       -- one row per chat message; token counts + metadata
SessionSummary   -- structured summary produced from a batch of TokenEvents
TrainingExample  -- ML-ready feature vector derived from a SessionSummary
ModelMetrics     -- snapshot of online-model performance
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DecisionType(str, Enum):
    BET_OVER = "BET_OVER"
    BET_UNDER = "BET_UNDER"
    PASS = "PASS"
    UNKNOWN = "UNKNOWN"


class Outcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    UNKNOWN = "UNKNOWN"


class FeedbackSentiment(str, Enum):
    STRONG_NEGATIVE = "strong_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    STRONG_POSITIVE = "strong_positive"


# ---------------------------------------------------------------------------
# TokenEvent
# ---------------------------------------------------------------------------


class TokenEvent(BaseModel):
    """
    One token-usage record per chat message.

    Fields
    ------
    event_id        : Unique identifier for this log row.
    session_id      : Groups all messages from one betting session.
    message_id      : Ordering key within the session (0-indexed).
    timestamp       : UTC time the message was processed.
    role            : Who sent the message (user / assistant / system).
    prompt_tokens   : Tokens in the prompt/context at this turn.
    completion_tokens: Tokens in the assistant reply (0 for user turns).
    total_tokens    : prompt_tokens + completion_tokens.
    cost_usd        : Estimated USD cost for this turn.
    rolling_session_tokens: Cumulative tokens in the session so far.
    market_context  : Free-form market identifier (e.g. "WNBA Q2 total 38.5").
    line_snapshot   : Line value at message time (e.g. 38.5).
    metadata        : Any additional key-value pairs for extensibility.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message_id: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: Role
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0, default=0.0)
    rolling_session_tokens: int = Field(ge=0, default=0)
    market_context: Optional[str] = None
    line_snapshot: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_total(self) -> "TokenEvent":
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens + completion_tokens ({expected})."
            )
        return self

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# SessionSummary
# ---------------------------------------------------------------------------


class DecisionPoint(BaseModel):
    """A single decision recorded during a session."""

    turn_index: int
    decision_type: DecisionType = DecisionType.UNKNOWN
    side: Optional[str] = None           # e.g. "over", "under"
    confidence: Optional[float] = None   # 0.0–1.0 if available
    threshold_used: Optional[float] = None
    time_left_seconds: Optional[float] = None
    score_snapshot: Optional[str] = None
    line_at_decision: Optional[float] = None


class SessionSummary(BaseModel):
    """
    Structured summary derived from a list of TokenEvents.

    Fields
    ------
    summary_id      : Unique identifier for this summary.
    session_id      : Links back to the raw events.
    created_at      : UTC time the summary was generated.
    league          : Sport/league code (NBA, WNBA, …).
    market_type     : Market kind (Q2_total, game_total, …).
    total_messages  : Number of chat turns in the session.
    total_tokens    : Cumulative token count across all turns.
    total_cost_usd  : Cumulative cost estimate.
    decision_points : Extracted picks / pass decisions.
    final_decision  : Last definitive decision in the session.
    outcome         : Settlement result if known.
    feedback_sentiment: Dominant user-feedback sentiment.
    raw_event_count : Number of TokenEvent rows that fed this summary.
    metadata        : Free-form extras.
    """

    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    league: str = "UNKNOWN"
    market_type: str = "OTHER"
    total_messages: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0, default=0)
    total_cost_usd: float = Field(ge=0.0, default=0.0)
    decision_points: List[DecisionPoint] = Field(default_factory=list)
    final_decision: DecisionType = DecisionType.UNKNOWN
    outcome: Outcome = Outcome.UNKNOWN
    feedback_sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    raw_event_count: int = Field(ge=0, default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


class TrainingExample(BaseModel):
    """
    One labelled training row ready for the online learner.

    The feature_vector is a flat dict whose keys are stable across calls so
    that FeatureEngineer can produce a consistent numpy array.
    """

    example_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_vector: Dict[str, float] = Field(default_factory=dict)
    label: DecisionType
    outcome: Outcome = Outcome.UNKNOWN
    weight: float = Field(default=1.0, gt=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature_vector")
    @classmethod
    def _no_nan_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        import math

        bad = [k for k, val in v.items() if math.isnan(val) or math.isinf(val)]
        if bad:
            raise ValueError(f"Feature vector contains NaN/Inf for keys: {bad}")
        return v

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# ModelMetrics
# ---------------------------------------------------------------------------


class ModelMetrics(BaseModel):
    """
    Performance snapshot of the online model.

    segment_metrics maps a segment key (e.g. "NBA::Q2_total") to its metrics dict.
    """

    metrics_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str
    n_examples_seen: int = Field(ge=0, default=0)
    accuracy: Optional[float] = None
    accuracy_window: int = Field(default=50, description="Rolling window size for accuracy.")
    class_counts: Dict[str, int] = Field(default_factory=dict)
    segment_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_token_event(data: Dict[str, Any]) -> TokenEvent:
    """Parse and validate a raw dict as a :class:`TokenEvent`.

    Raises ``pydantic.ValidationError`` on failure with a descriptive message.
    """
    return TokenEvent.model_validate(data)


def validate_session_summary(data: Dict[str, Any]) -> SessionSummary:
    """Parse and validate a raw dict as a :class:`SessionSummary`."""
    return SessionSummary.model_validate(data)


def validate_training_example(data: Dict[str, Any]) -> TrainingExample:
    """Parse and validate a raw dict as a :class:`TrainingExample`."""
    return TrainingExample.model_validate(data)


def validate_model_metrics(data: Dict[str, Any]) -> ModelMetrics:
    """Parse and validate a raw dict as a :class:`ModelMetrics`."""
    return ModelMetrics.model_validate(data)
