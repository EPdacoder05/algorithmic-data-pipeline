"""
Pipeline Configuration
======================
All runtime settings for the token-tracking → summarization → online-learning
pipeline.  Values may be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """
    Central configuration for the chat-pipeline.

    Environment variable names match the field names (upper-case).
    Example override::

        PIPELINE_MAX_TOKENS=8192 python -m src.cli ingest-events …
    """

    # ------------------------------------------------------------------ paths
    DATA_DIR: Path = Field(
        default=Path("data"),
        description="Root directory for all pipeline artifacts.",
    )
    EVENTS_DIR: Path = Field(
        default=Path("data/token_events"),
        description="Where .jsonl token-event logs are stored.",
    )
    SUMMARIES_DIR: Path = Field(
        default=Path("data/session_summaries"),
        description="Where session-summary .jsonl files are stored.",
    )
    TRAINING_DIR: Path = Field(
        default=Path("data/training_sets"),
        description="Where built training-set .jsonl files land.",
    )
    MODEL_DIR: Path = Field(
        default=Path("data/models"),
        description="Serialised model artefacts (joblib).",
    )

    # --------------------------------------------------------------- tokeniser
    # Simple character-based approximation by default (no external dependency).
    # Switch to "tiktoken" to use OpenAI-compatible counting.
    TOKENIZER_BACKEND: str = Field(
        default="simple",
        description="Token counting backend.  'simple' = ~4 chars/token heuristic.",
    )
    TIKTOKEN_ENCODING: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name (used when TOKENIZER_BACKEND='tiktoken').",
    )

    # ---------------------------------------------------------------- guardrails
    MAX_SESSION_TOKENS: int = Field(
        default=16_000,
        description="Alert threshold: log a WARNING when a session exceeds this many tokens.",
    )
    MAX_MESSAGE_TOKENS: int = Field(
        default=4_096,
        description="Alert threshold: log a WARNING when a single message exceeds this.",
    )

    # ------------------------------------------------------------------- costs
    # Provider-agnostic cost hooks (USD per 1 000 tokens).
    COST_PER_1K_PROMPT: float = Field(
        default=0.001,
        description="USD cost per 1 000 prompt tokens.",
    )
    COST_PER_1K_COMPLETION: float = Field(
        default=0.002,
        description="USD cost per 1 000 completion tokens.",
    )

    # --------------------------------------------------------- online learning
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.55,
        description="Minimum class probability to emit BET_OVER / BET_UNDER (else PASS).",
    )
    LEARNING_RATE: float = Field(
        default=0.01,
        description="SGD learning rate passed to SGDClassifier.",
    )
    CLASS_LABELS: List[str] = Field(
        default=["BET_OVER", "BET_UNDER", "PASS"],
        description="Ordered list of action labels emitted by the online trainer.",
    )
    MODEL_VERSION_PREFIX: str = Field(
        default="v",
        description="Prefix for versioned model artefact filenames.",
    )
    SEGMENT_TRACK_FIELDS: List[str] = Field(
        default=["league", "market_type"],
        description="Feature fields used to bucket segment-level metrics.",
    )

    # ---------------------------------------------------------------- leagues
    KNOWN_LEAGUES: List[str] = Field(
        default=["NBA", "WNBA", "NCAAB", "EURO", "OTHER"],
        description="All recognised league codes for one-hot encoding.",
    )
    KNOWN_MARKET_TYPES: List[str] = Field(
        default=["Q1_total", "Q2_total", "Q3_total", "Q4_total", "game_total", "OTHER"],
        description="All recognised market types for one-hot encoding.",
    )

    @field_validator("TOKENIZER_BACKEND")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        allowed = {"simple", "tiktoken"}
        if v not in allowed:
            raise ValueError(f"TOKENIZER_BACKEND must be one of {allowed}, got {v!r}")
        return v

    model_config = {
        "env_prefix": "PIPELINE_",
        "env_file": ".env",
        "case_sensitive": False,
    }


# Module-level singleton — importable everywhere as `from src.config import settings`
settings = PipelineSettings()
