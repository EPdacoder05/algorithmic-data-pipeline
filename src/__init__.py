"""
Token-size + chat-summary + online-learning pipeline for live betting decision quality.

Modules
-------
config              Pipeline configuration
io_schemas          Pydantic data contracts (TokenEvent, SessionSummary, …)
token_tracker       Per-message token logging with JSONL persistence
session_summarizer  Rule-based session summarizer
feature_engineering Feature transforms for the online model
online_trainer      SGDClassifier with partial_fit loop
cli                 Click CLI entry-points
"""

__version__ = "0.1.0"
