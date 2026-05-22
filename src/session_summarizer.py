"""
Session Summarizer
==================
Produces :class:`~src.io_schemas.SessionSummary` objects from a list of
:class:`~src.io_schemas.TokenEvent` records using deterministic, rule-based
extraction — no external LLM required.

Extraction logic
----------------
Decision points:
    Scans ``market_context`` and ``metadata`` fields for keywords
    (over/under/pass, confidence levels, required-rate figures).

State snapshots:
    Reads ``line_snapshot``, ``metadata.time_left``, ``metadata.score``.

Outcome signals:
    Looks for settlement keywords (win/loss/push/cover) in metadata or
    assistant message content stored in metadata.

Feedback sentiment:
    Classifies user turns based on negative/positive keyword density.

Usage example::

    from src.session_summarizer import SessionSummarizer
    from src.token_tracker import load_events_from_jsonl

    events = load_events_from_jsonl("data/token_events/sess_001.jsonl")
    summarizer = SessionSummarizer()
    summary = summarizer.summarize(events, session_id="sess_001")
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.io_schemas import (
    DecisionPoint,
    DecisionType,
    FeedbackSentiment,
    Outcome,
    Role,
    SessionSummary,
    TokenEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets for rule-based classification
# ---------------------------------------------------------------------------

_OVER_PATTERNS = re.compile(r"\b(bet[\s_]over|over|o\d+\.?\d*)\b", re.IGNORECASE)
_UNDER_PATTERNS = re.compile(r"\b(bet[\s_]under|under|u\d+\.?\d*)\b", re.IGNORECASE)
_PASS_PATTERNS = re.compile(r"\bpass\b", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
_REQUIRED_RATE_PATTERN = re.compile(
    r"required[\s_]rate[:\s]+([\d.]+)\s*(?:pts?/min)?", re.IGNORECASE
)
_LINE_PATTERN = re.compile(r"line[\s:]+(\d+\.?\d*)", re.IGNORECASE)
_TIME_PATTERN = re.compile(r"(\d+:\d+)\s*left", re.IGNORECASE)
_SCORE_PATTERN = re.compile(r"score[:\s]+([^\.,;\n]{1,40})", re.IGNORECASE)

# Settlement keywords → Outcome
_WIN_KEYWORDS = {"win", "won", "correct", "cover", "covered", "hit"}
_LOSS_KEYWORDS = {"loss", "lost", "wrong", "miss", "missed", "no cover"}
_PUSH_KEYWORDS = {"push", "pushed", "tie", "tied", "no action"}

# Sentiment keywords
_STRONG_NEG = {"terrible", "stench", "ewww", "worst", "idiot", "stink", "fix it", "awful"}
_NEGATIVE = {"wrong", "bad", "incorrect", "no", "don't", "dont", "off", "error"}
_POSITIVE = {"right", "good", "correct", "great", "nice", "yes", "perfect", "solid"}
_STRONG_POS = {"amazing", "excellent", "perfect", "love it", "fire", "🔥"}


# ---------------------------------------------------------------------------
# SessionSummarizer
# ---------------------------------------------------------------------------


class SessionSummarizer:
    """
    Rule-based summarizer: converts raw :class:`TokenEvent` lists into a
    :class:`~src.io_schemas.SessionSummary`.

    Parameters
    ----------
    league_hint:
        Optional override for the league field when it cannot be inferred from
        the events (e.g. ``"WNBA"``).
    market_type_hint:
        Optional override for ``market_type``.
    """

    def __init__(
        self,
        league_hint: Optional[str] = None,
        market_type_hint: Optional[str] = None,
    ) -> None:
        self.league_hint = league_hint
        self.market_type_hint = market_type_hint

    # ------------------------------------------------------------------ public

    def summarize(
        self,
        events: List[TokenEvent],
        session_id: Optional[str] = None,
    ) -> SessionSummary:
        """
        Summarize a list of events for one session.

        Parameters
        ----------
        events:
            All :class:`TokenEvent` records for the session in chronological
            order (ascending ``message_id``).
        session_id:
            Falls back to ``events[0].session_id`` when not provided.

        Returns
        -------
        SessionSummary
        """
        if not events:
            sid = session_id or "unknown"
            logger.warning("Summarizing empty event list for session %s", sid)
            return SessionSummary(session_id=sid)

        sid = session_id or events[0].session_id
        events = sorted(events, key=lambda e: e.message_id)

        league = self.league_hint or self._infer_league(events)
        market_type = self.market_type_hint or self._infer_market_type(events)
        decision_points = self._extract_decisions(events)
        final_decision = self._final_decision(decision_points)
        outcome = self._extract_outcome(events)
        sentiment = self._classify_sentiment(events)

        total_tokens = sum(e.total_tokens for e in events)
        total_cost = round(sum(e.cost_usd for e in events), 6)

        summary = SessionSummary(
            session_id=sid,
            league=league,
            market_type=market_type,
            total_messages=len(events),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            decision_points=decision_points,
            final_decision=final_decision,
            outcome=outcome,
            feedback_sentiment=sentiment,
            raw_event_count=len(events),
            metadata={
                "first_market_context": events[0].market_context,
                "last_line_snapshot": events[-1].line_snapshot,
            },
        )
        logger.debug("Summary for session %s: %s", sid, summary.model_dump_json())
        return summary

    # ----------------------------------------------------------------- private

    def _infer_league(self, events: List[TokenEvent]) -> str:
        """Scan market_context fields for a league name."""
        for event in events:
            ctx = (event.market_context or "").upper()
            if "WNBA" in ctx:
                return "WNBA"
            if "NBA" in ctx:
                return "NBA"
            if "NCAAB" in ctx or "COLLEGE" in ctx:
                return "NCAAB"
            if any(k in ctx for k in ("EURO", "EUR", "EUROLEAGUE")):
                return "EURO"
        return "OTHER"

    def _infer_market_type(self, events: List[TokenEvent]) -> str:
        """Scan context fields for a market/quarter identifier."""
        for event in events:
            ctx = (event.market_context or "").upper()
            for q in ("Q1", "Q2", "Q3", "Q4"):
                if q in ctx:
                    return f"{q}_total"
            if "GAME" in ctx or "FULL" in ctx:
                return "game_total"
        return "OTHER"

    def _extract_decisions(self, events: List[TokenEvent]) -> List[DecisionPoint]:
        """Parse decision points from assistant messages."""
        points: List[DecisionPoint] = []
        for event in events:
            # Decisions come from assistant turns or metadata
            text_sources = [event.market_context or ""]
            if event.metadata:
                text_sources.append(str(event.metadata.get("content", "")))
                text_sources.append(str(event.metadata.get("text", "")))
            combined = " ".join(text_sources)

            decision, side = self._parse_decision_type(combined)
            if decision == DecisionType.UNKNOWN and event.role != Role.USER:
                continue  # Skip turns with no detectable decision

            confidence = self._parse_confidence(combined)
            required_rate = self._parse_required_rate(combined)
            line = event.line_snapshot or self._parse_line(combined)
            time_left = self._parse_time_left_seconds(combined)
            score = self._parse_score(combined)

            dp = DecisionPoint(
                turn_index=event.message_id,
                decision_type=decision,
                side=side,
                confidence=confidence,
                threshold_used=required_rate,
                time_left_seconds=time_left,
                score_snapshot=score,
                line_at_decision=line,
            )
            points.append(dp)

        return points

    def _final_decision(self, decision_points: List[DecisionPoint]) -> DecisionType:
        """Return the last non-UNKNOWN decision."""
        for dp in reversed(decision_points):
            if dp.decision_type != DecisionType.UNKNOWN:
                return dp.decision_type
        return DecisionType.UNKNOWN

    def _extract_outcome(self, events: List[TokenEvent]) -> Outcome:
        """Scan metadata and market_context for settlement signals."""
        for event in reversed(events):
            text = " ".join(
                [
                    (event.market_context or "").lower(),
                    str(event.metadata.get("outcome", "")).lower(),
                    str(event.metadata.get("result", "")).lower(),
                ]
            )
            words = set(text.split())
            if words & _WIN_KEYWORDS:
                return Outcome.WIN
            if words & _LOSS_KEYWORDS:
                return Outcome.LOSS
            if words & _PUSH_KEYWORDS:
                return Outcome.PUSH
        return Outcome.UNKNOWN

    def _classify_sentiment(self, events: List[TokenEvent]) -> FeedbackSentiment:
        """
        Compute dominant user-feedback sentiment from USER turns.

        Scoring:
            strong_negative word  →  -2
            negative word         →  -1
            positive word         →  +1
            strong_positive word  →  +2
        """
        score = 0
        user_events = [e for e in events if e.role == Role.USER]
        for event in user_events:
            text = " ".join(
                [
                    str(event.metadata.get("content", "")),
                    str(event.metadata.get("text", "")),
                    event.market_context or "",
                ]
            ).lower()
            words = set(re.findall(r"\w+", text))

            for phrase in _STRONG_NEG:
                if phrase in text:
                    score -= 2
            for phrase in _STRONG_POS:
                if phrase in text:
                    score += 2
            score -= len(words & _NEGATIVE)
            score += len(words & _POSITIVE)

        if score <= -4:
            return FeedbackSentiment.STRONG_NEGATIVE
        if score <= -1:
            return FeedbackSentiment.NEGATIVE
        if score >= 4:
            return FeedbackSentiment.STRONG_POSITIVE
        if score >= 1:
            return FeedbackSentiment.POSITIVE
        return FeedbackSentiment.NEUTRAL

    # ----------------------------------------------------------------- parsers

    @staticmethod
    def _parse_decision_type(text: str) -> Tuple[DecisionType, Optional[str]]:
        if _PASS_PATTERNS.search(text):
            return DecisionType.PASS, None
        if _OVER_PATTERNS.search(text):
            return DecisionType.BET_OVER, "over"
        if _UNDER_PATTERNS.search(text):
            return DecisionType.BET_UNDER, "under"
        return DecisionType.UNKNOWN, None

    @staticmethod
    def _parse_confidence(text: str) -> Optional[float]:
        m = _CONFIDENCE_PATTERN.search(text)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
        return None

    @staticmethod
    def _parse_required_rate(text: str) -> Optional[float]:
        m = _REQUIRED_RATE_PATTERN.search(text)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _parse_line(text: str) -> Optional[float]:
        m = _LINE_PATTERN.search(text)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _parse_time_left_seconds(text: str) -> Optional[float]:
        m = _TIME_PATTERN.search(text)
        if m:
            parts = m.group(1).split(":")
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_score(text: str) -> Optional[str]:
        m = _SCORE_PATTERN.search(text)
        if m:
            return m.group(1).strip()
        return None
