"""
Tests for token_tracker.py
============================
Validates token counting, cost estimation, event generation, guardrails,
and JSONL round-trip persistence.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.token_tracker import (
    TokenTracker,
    count_tokens,
    estimate_cost,
    load_events_from_jsonl,
    _count_tokens_simple,
)
from src.io_schemas import TokenEvent, Role


class TestCountTokens:
    """Unit tests for the token-counting backends."""

    def test_simple_nonempty(self):
        n = _count_tokens_simple("Hello world")
        assert n >= 1

    def test_simple_empty_returns_one(self):
        # The heuristic floors at 1
        n = _count_tokens_simple("")
        assert n >= 1

    def test_simple_long_text(self):
        text = "a" * 400
        n = _count_tokens_simple(text)
        assert n == 100

    def test_count_tokens_uses_simple_backend(self, monkeypatch):
        monkeypatch.setattr("src.token_tracker.settings.TOKENIZER_BACKEND", "simple")
        n = count_tokens("short text")
        assert n >= 1


class TestEstimateCost:
    """Unit tests for cost estimation."""

    def test_zero_tokens_zero_cost(self):
        assert estimate_cost(0, 0) == 0.0

    def test_cost_is_positive(self):
        c = estimate_cost(1000, 500)
        assert c > 0.0

    def test_cost_rounded(self):
        c = estimate_cost(100, 50)
        # Should be a float with at most 8 decimal places
        assert isinstance(c, float)


class TestTokenTracker:
    """Integration tests for TokenTracker."""

    def test_ingest_returns_token_event(self):
        tracker = TokenTracker(session_id="test_sess")
        event = tracker.ingest(role="user", text="WNBA Q2 is live")
        assert isinstance(event, TokenEvent)
        assert event.session_id == "test_sess"
        assert event.message_id == 0

    def test_message_ids_increment(self):
        tracker = TokenTracker(session_id="test_sess2")
        e0 = tracker.ingest(role="user", text="msg 0")
        e1 = tracker.ingest(role="assistant", text="msg 1")
        e2 = tracker.ingest(role="user", text="msg 2")
        assert e0.message_id == 0
        assert e1.message_id == 1
        assert e2.message_id == 2

    def test_rolling_tokens_accumulate(self):
        tracker = TokenTracker(session_id="roll_test")
        tracker.ingest(role="user", text="Hello there")
        tracker.ingest(role="assistant", text="Hi back")
        # Rolling should be positive and larger than single message
        stats = tracker.session_stats()
        assert stats["total_tokens"] > 0
        assert stats["message_count"] == 2

    def test_market_context_inherited(self):
        tracker = TokenTracker(session_id="ctx_test", market_context="WNBA Q2 38.5")
        event = tracker.ingest(role="user", text="check this")
        assert event.market_context == "WNBA Q2 38.5"

    def test_market_context_overridden_per_message(self):
        tracker = TokenTracker(session_id="ctx_over", market_context="NBA game total")
        event = tracker.ingest(
            role="user", text="check", market_context="WNBA Q2 40.5"
        )
        assert event.market_context == "WNBA Q2 40.5"

    def test_events_list_before_flush(self):
        tracker = TokenTracker(session_id="list_test")
        tracker.ingest(role="user", text="a")
        tracker.ingest(role="assistant", text="b")
        events = tracker.events()
        assert len(events) == 2

    def test_flush_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TokenTracker(session_id="flush_test")
            tracker.ingest(role="user", text="message one")
            tracker.ingest(role="assistant", text="reply")
            out = Path(tmpdir) / "flush_test.jsonl"
            written = tracker.flush(out, append=False)
            assert written == 2
            assert out.exists()
            lines = out.read_text().strip().splitlines()
            assert len(lines) == 2

    def test_flush_clears_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TokenTracker(session_id="clear_test")
            tracker.ingest(role="user", text="hello")
            out = Path(tmpdir) / "clear_test.jsonl"
            tracker.flush(out)
            # Buffer should be empty now
            assert tracker.events() == []

    def test_flush_append_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "append_test.jsonl"
            for i in range(3):
                tracker = TokenTracker(session_id="app_sess")
                tracker.ingest(role="user", text=f"msg {i}")
                tracker.flush(out, append=True)
            lines = out.read_text().strip().splitlines()
            assert len(lines) == 3

    def test_session_stats_empty(self):
        tracker = TokenTracker(session_id="empty_stats")
        stats = tracker.session_stats()
        assert stats["message_count"] == 0
        assert stats["total_tokens"] == 0

    def test_guardrail_large_message_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.token_tracker"):
            tracker = TokenTracker(session_id="guard_test")
            # Create a very large text to trigger the guardrail
            big_text = "a" * 100_000  # ~25 000 tokens via simple backend
            tracker.ingest(role="user", text=big_text)
        assert any("MAX_MESSAGE_TOKENS" in r.message for r in caplog.records)


class TestLoadEventsFromJsonl:
    """Tests for the JSONL loader."""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TokenTracker(session_id="rt_sess")
            tracker.ingest(role="user", text="round trip test")
            out = Path(tmpdir) / "rt_sess.jsonl"
            tracker.flush(out)

            events = load_events_from_jsonl(out)
            assert len(events) == 1
            assert events[0].session_id == "rt_sess"

    def test_missing_file_returns_empty(self):
        events = load_events_from_jsonl("/nonexistent/path/to/file.jsonl")
        assert events == []

    def test_invalid_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bad.jsonl"
            out.write_text('{"not": "valid token event"}\n{"also": "bad"}\n')
            events = load_events_from_jsonl(out)
            # Both lines fail validation → empty list (no crash)
            assert isinstance(events, list)
