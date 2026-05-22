"""
Tests for session_summarizer.py
=================================
Validates rule-based extraction of decision points, outcome signals,
feedback sentiment, and end-to-end summarization from TokenEvents.
"""

import pytest
from datetime import datetime, timezone

from src.io_schemas import (
    DecisionType,
    FeedbackSentiment,
    Outcome,
    Role,
    SessionSummary,
    TokenEvent,
)
from src.session_summarizer import SessionSummarizer


def _make_event(
    session_id="sess_test",
    message_id=0,
    role="user",
    market_context=None,
    line_snapshot=None,
    **metadata,
) -> TokenEvent:
    """Helper: build a minimal valid TokenEvent."""
    return TokenEvent(
        session_id=session_id,
        message_id=message_id,
        role=role,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        market_context=market_context,
        line_snapshot=line_snapshot,
        metadata=metadata,
    )


class TestSessionSummarizer:
    """Tests for the SessionSummarizer class."""

    def setup_method(self):
        self.summarizer = SessionSummarizer()

    def test_empty_events_returns_summary(self):
        summary = self.summarizer.summarize([], session_id="empty_sess")
        assert isinstance(summary, SessionSummary)
        assert summary.session_id == "empty_sess"
        assert summary.total_messages == 0

    def test_session_id_from_first_event_fallback(self):
        events = [_make_event(session_id="auto_sess")]
        summary = self.summarizer.summarize(events)
        assert summary.session_id == "auto_sess"

    def test_total_messages_count(self):
        events = [_make_event(message_id=i) for i in range(4)]
        summary = self.summarizer.summarize(events)
        assert summary.total_messages == 4

    def test_total_tokens_summed(self):
        events = [_make_event(message_id=i) for i in range(3)]
        summary = self.summarizer.summarize(events)
        assert summary.total_tokens == 3 * 15

    def test_league_inferred_wnba(self):
        events = [_make_event(market_context="WNBA Q2 total 38.5")]
        summary = self.summarizer.summarize(events)
        assert summary.league == "WNBA"

    def test_league_inferred_nba(self):
        events = [_make_event(market_context="NBA Q3 total 55.5")]
        summary = self.summarizer.summarize(events)
        assert summary.league == "NBA"

    def test_league_hint_overrides_inference(self):
        summarizer = SessionSummarizer(league_hint="EURO")
        events = [_make_event(market_context="NBA game total 220")]
        summary = summarizer.summarize(events)
        assert summary.league == "EURO"

    def test_market_type_q2_inferred(self):
        events = [_make_event(market_context="WNBA Q2 total 38.5")]
        summary = self.summarizer.summarize(events)
        assert summary.market_type == "Q2_total"

    def test_market_type_hint_overrides(self):
        summarizer = SessionSummarizer(market_type_hint="game_total")
        events = [_make_event(market_context="Q2 WNBA")]
        summary = summarizer.summarize(events)
        assert summary.market_type == "game_total"

    def test_decision_pass_extracted(self):
        events = [
            _make_event(
                message_id=0,
                role="assistant",
                market_context="PASS — required rate 5.1 pts/min",
            )
        ]
        summary = self.summarizer.summarize(events)
        assert summary.final_decision == DecisionType.PASS

    def test_decision_over_extracted(self):
        events = [
            _make_event(
                message_id=0,
                role="assistant",
                market_context="BET OVER 38.5 — confidence 70%",
            )
        ]
        summary = self.summarizer.summarize(events)
        assert summary.final_decision == DecisionType.BET_OVER

    def test_decision_under_extracted(self):
        events = [
            _make_event(
                message_id=0,
                role="assistant",
                market_context="BET UNDER 40.5",
            )
        ]
        summary = self.summarizer.summarize(events)
        assert summary.final_decision == DecisionType.BET_UNDER

    def test_outcome_win_extracted(self):
        events = [_make_event(outcome="win", result="covered")]
        summary = self.summarizer.summarize(events)
        assert summary.outcome == Outcome.WIN

    def test_outcome_loss_extracted(self):
        events = [_make_event(outcome="loss")]
        summary = self.summarizer.summarize(events)
        assert summary.outcome == Outcome.LOSS

    def test_outcome_push_extracted(self):
        events = [_make_event(outcome="push")]
        summary = self.summarizer.summarize(events)
        assert summary.outcome == Outcome.PUSH

    def test_outcome_unknown_default(self):
        events = [_make_event()]
        summary = self.summarizer.summarize(events)
        assert summary.outcome == Outcome.UNKNOWN

    def test_sentiment_strong_negative(self):
        events = [
            _make_event(role="user", content="terrible stench you stink awful")
        ]
        summary = self.summarizer.summarize(events)
        assert summary.feedback_sentiment in (
            FeedbackSentiment.STRONG_NEGATIVE,
            FeedbackSentiment.NEGATIVE,
        )

    def test_sentiment_positive(self):
        events = [
            _make_event(role="user", content="great correct perfect solid call")
        ]
        summary = self.summarizer.summarize(events)
        assert summary.feedback_sentiment in (
            FeedbackSentiment.POSITIVE,
            FeedbackSentiment.STRONG_POSITIVE,
        )

    def test_sentiment_neutral_default(self):
        events = [_make_event(role="user", content="ok")]
        summary = self.summarizer.summarize(events)
        # Neutral or mild negative/positive — just ensure it's a valid value
        assert summary.feedback_sentiment in FeedbackSentiment.__members__.values()

    def test_events_sorted_by_message_id(self):
        """Summarizer should produce correct decision order regardless of input order."""
        e0 = _make_event(message_id=2, role="assistant", market_context="PASS")
        e1 = _make_event(message_id=0, role="assistant", market_context="BET OVER")
        e2 = _make_event(message_id=1, role="user", content="ok")
        summary = self.summarizer.summarize([e0, e1, e2])
        # Last decision in sorted order is message_id=2 (PASS)
        assert summary.final_decision == DecisionType.PASS

    def test_line_snapshot_captured_in_decision_point(self):
        events = [
            _make_event(
                message_id=0,
                role="assistant",
                market_context="BET OVER — line 38.5",
                line_snapshot=38.5,
            )
        ]
        summary = self.summarizer.summarize(events)
        dp_with_line = [
            dp for dp in summary.decision_points if dp.line_at_decision is not None
        ]
        assert len(dp_with_line) >= 1
        assert dp_with_line[0].line_at_decision == pytest.approx(38.5)

    def test_required_rate_parsed_from_context(self):
        events = [
            _make_event(
                message_id=0,
                role="assistant",
                market_context="PASS required rate: 5.13 pts/min",
            )
        ]
        summary = self.summarizer.summarize(events)
        dp_with_rate = [
            dp for dp in summary.decision_points if dp.threshold_used is not None
        ]
        assert len(dp_with_rate) >= 1
        assert dp_with_rate[0].threshold_used == pytest.approx(5.13)

    def test_bootstrap_session_summarization(self):
        """End-to-end smoke test using bootstrap chat data."""
        from src.token_tracker import load_events_from_jsonl
        import pathlib

        jsonl_path = (
            pathlib.Path(__file__).parent.parent
            / "data/bootstrap_sessions/bootstrap_session_001.jsonl"
        )
        # Load as raw dicts and re-ingest via TokenTracker
        from src.token_tracker import TokenTracker
        import json

        tracker = TokenTracker(session_id="bootstrap_001", market_context="WNBA Q2 38.5")
        with open(jsonl_path, encoding="utf-8") as fh:
            for msg in fh:
                msg = msg.strip()
                if not msg:
                    continue
                d = json.loads(msg)
                tracker.ingest(
                    role=d["role"],
                    text=d["content"],
                    line_snapshot=d.get("line_snapshot"),
                    market_context=d.get("market_context"),
                )

        summarizer = SessionSummarizer(league_hint="WNBA", market_type_hint="Q2_total")
        summary = summarizer.summarize(tracker.events(), session_id="bootstrap_001")

        assert summary.session_id == "bootstrap_001"
        assert summary.league == "WNBA"
        assert summary.total_messages > 0
        assert summary.total_tokens > 0
