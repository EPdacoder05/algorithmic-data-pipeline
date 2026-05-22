"""
Token Tracker
=============
Ingests chat turns, counts tokens, maintains rolling session stats, estimates
costs, enforces guardrails, and writes newline-delimited JSON to disk with
atomic append semantics.

Usage example::

    tracker = TokenTracker(session_id="sess_001")
    event = tracker.ingest(role="user", text="WNBA Q2, 4:29 left, line 38.5")
    tracker.ingest(role="assistant", text="PASS — required rate too high.")
    tracker.flush("data/token_events/sess_001.jsonl")
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import settings
from src.io_schemas import Role, TokenEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-counting backends
# ---------------------------------------------------------------------------


def _count_tokens_simple(text: str) -> int:
    """
    Heuristic token counter (~4 chars per token).

    Used when ``PIPELINE_TOKENIZER_BACKEND=simple`` (the default).  Avoids any
    external dependency.
    """
    return max(1, len(text) // 4)


def _count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Exact token counter using the *tiktoken* library.

    Raises ``ImportError`` if tiktoken is not installed.
    """
    import tiktoken  # optional dependency

    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def count_tokens(text: str) -> int:
    """
    Count tokens in *text* using the backend configured in
    ``settings.TOKENIZER_BACKEND``.
    """
    if settings.TOKENIZER_BACKEND == "tiktoken":
        return _count_tokens_tiktoken(text, settings.TIKTOKEN_ENCODING)
    return _count_tokens_simple(text)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate USD cost for a single turn.

    Rates are read from :data:`src.config.settings` so they can be overridden
    via environment variables without code changes.
    """
    prompt_cost = (prompt_tokens / 1_000) * settings.COST_PER_1K_PROMPT
    completion_cost = (completion_tokens / 1_000) * settings.COST_PER_1K_COMPLETION
    return round(prompt_cost + completion_cost, 8)


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """
    Tracks per-session token usage and produces :class:`~src.io_schemas.TokenEvent`
    records.

    Parameters
    ----------
    session_id:
        Stable identifier grouping all messages from one betting session.
    market_context:
        Optional human-readable market description to attach to every event
        (can be overridden per message).
    """

    def __init__(
        self,
        session_id: str,
        market_context: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.default_market_context = market_context
        self._events: List[TokenEvent] = []
        self._rolling_tokens: int = 0
        self._next_message_id: int = 0

    # ------------------------------------------------------------------ public

    def ingest(
        self,
        role: str,
        text: str,
        prompt_text: Optional[str] = None,
        line_snapshot: Optional[float] = None,
        market_context: Optional[str] = None,
        **extra_metadata,
    ) -> TokenEvent:
        """
        Process one chat turn and return the resulting :class:`TokenEvent`.

        Parameters
        ----------
        role:
            ``"user"``, ``"assistant"``, or ``"system"``.
        text:
            The full text of this message (used for completion token count).
        prompt_text:
            Text that was provided *as context* to the model this turn.  When
            ``None``, prompt tokens are estimated from ``text`` length.
        line_snapshot:
            Market line value at the time of this message.
        market_context:
            Overrides the session-level ``default_market_context`` for this
            message only.
        **extra_metadata:
            Arbitrary key-value pairs persisted in the event's ``metadata``.
        """
        role_enum = Role(role)
        completion_tokens = count_tokens(text)
        prompt_tokens = count_tokens(prompt_text) if prompt_text else count_tokens(text)
        total_tokens = prompt_tokens + completion_tokens
        self._rolling_tokens += total_tokens

        self._check_guardrails(completion_tokens, prompt_tokens)

        event = TokenEvent(
            session_id=self.session_id,
            message_id=self._next_message_id,
            timestamp=datetime.now(timezone.utc),
            role=role_enum,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=estimate_cost(prompt_tokens, completion_tokens),
            rolling_session_tokens=self._rolling_tokens,
            market_context=market_context or self.default_market_context,
            line_snapshot=line_snapshot,
            metadata=extra_metadata or {},
        )
        self._events.append(event)
        self._next_message_id += 1
        return event

    def session_stats(self) -> dict:
        """Return a summary dict of the current session's token usage."""
        if not self._events:
            return {
                "session_id": self.session_id,
                "message_count": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }
        return {
            "session_id": self.session_id,
            "message_count": len(self._events),
            "total_tokens": self._rolling_tokens,
            "total_cost_usd": round(sum(e.cost_usd for e in self._events), 6),
            "prompt_tokens": sum(e.prompt_tokens for e in self._events),
            "completion_tokens": sum(e.completion_tokens for e in self._events),
        }

    def flush(self, path: str | Path, append: bool = True) -> int:
        """
        Write all buffered events to a ``.jsonl`` file atomically.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created as needed.
        append:
            When ``True`` (default), events are appended to an existing file.
            When ``False``, the file is overwritten.

        Returns
        -------
        int
            Number of events written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        written = 0
        with open(path, mode, encoding="utf-8") as fh:
            # Acquire an exclusive advisory lock for atomic append
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                for event in self._events:
                    fh.write(event.model_dump_json() + "\n")
                    written += 1
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

        logger.info("Flushed %d token events to %s", written, path)
        self._events.clear()
        return written

    def events(self) -> List[TokenEvent]:
        """Return a copy of buffered events (does not clear the buffer)."""
        return list(self._events)

    # ----------------------------------------------------------------- private

    def _check_guardrails(self, completion_tokens: int, prompt_tokens: int) -> None:
        msg_tokens = completion_tokens + prompt_tokens
        if msg_tokens > settings.MAX_MESSAGE_TOKENS:
            logger.warning(
                "Session %s: message exceeds MAX_MESSAGE_TOKENS (%d > %d).",
                self.session_id,
                msg_tokens,
                settings.MAX_MESSAGE_TOKENS,
            )
        if self._rolling_tokens > settings.MAX_SESSION_TOKENS:
            logger.warning(
                "Session %s: rolling total exceeds MAX_SESSION_TOKENS (%d > %d).",
                self.session_id,
                self._rolling_tokens,
                settings.MAX_SESSION_TOKENS,
            )


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_events_from_jsonl(path: str | Path) -> List[TokenEvent]:
    """
    Load all :class:`~src.io_schemas.TokenEvent` records from a ``.jsonl`` file.

    Invalid lines are logged and skipped rather than raising.
    """
    events: List[TokenEvent] = []
    path = Path(path)
    if not path.exists():
        logger.warning("Events file not found: %s", path)
        return events

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TokenEvent.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping invalid event on line %d of %s: %s", lineno, path, exc)

    return events
