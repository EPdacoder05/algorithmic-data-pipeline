"""
CLI — Token+Chat+Online-Learning Pipeline
==========================================
Entry-points for operating the full pipeline from the command line.

Commands
--------
ingest-events       Ingest a raw chat JSONL and write token events.
summarize-session   Summarise one session's token-event log.
build-training-set  Convert session summaries into a training JSONL.
train-online        Run partial_fit on a training JSONL.
evaluate            Print current model metrics.

Quick example::

    # 1. Ingest a raw session
    python -m src.cli ingest-events \\
        --input data/bootstrap_sessions/bootstrap_session_001.jsonl \\
        --session-id sess_bootstrap_001

    # 2. Summarise
    python -m src.cli summarize-session \\
        --session-id sess_bootstrap_001 \\
        --league WNBA --market-type Q2_total

    # 3. Build training set
    python -m src.cli build-training-set

    # 4. Train
    python -m src.cli train-online

    # 5. Evaluate
    python -m src.cli evaluate
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.io_schemas import (
    SessionSummary,
    TokenEvent,
    TrainingExample,
    validate_session_summary,
    validate_token_event,
    validate_training_example,
)
from src.online_trainer import OnlineTrainer
from src.session_summarizer import SessionSummarizer
from src.token_tracker import TokenTracker, load_events_from_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option("0.1.0", prog_name="pipeline")
def cli() -> None:
    """Token-size + chat-summary + online-learning pipeline for live betting."""


# ---------------------------------------------------------------------------
# ingest-events
# ---------------------------------------------------------------------------


@cli.command("ingest-events")
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSONL file of raw chat messages.  "
         "Each line must be a JSON object with at least "
         "'role' (user|assistant|system) and 'content' fields.",
)
@click.option(
    "--session-id", "-s",
    required=True,
    help="Stable identifier for this session (used in filenames and schemas).",
)
@click.option(
    "--market-context", "-m",
    default=None,
    help="Optional market descriptor attached to every event, e.g. 'WNBA Q2 38.5'.",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory to write the .jsonl event log.  Default: {settings.EVENTS_DIR}",
)
def ingest_events(
    input_path: str,
    session_id: str,
    market_context: Optional[str],
    out_dir: Optional[str],
) -> None:
    """
    Ingest a raw chat JSONL and write per-message token events.

    Each line of INPUT must be JSON with keys ``role`` and ``content``.
    Optional keys: ``line_snapshot`` (float), ``market_context`` (str).

    \b
    Example
    -------
        python -m src.cli ingest-events \\
            --input data/bootstrap_sessions/bootstrap_session_001.jsonl \\
            --session-id sess_bootstrap_001 \\
            --market-context "WNBA Q2 total 38.5"
    """
    out_dir_path = Path(out_dir) if out_dir else settings.EVENTS_DIR
    out_file = out_dir_path / f"{session_id}.jsonl"

    tracker = TokenTracker(session_id=session_id, market_context=market_context)

    count = 0
    with open(input_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                click.echo(f"[WARN] Skipping invalid JSON on line {lineno}: {exc}", err=True)
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            tracker.ingest(
                role=role,
                text=content,
                line_snapshot=msg.get("line_snapshot"),
                market_context=msg.get("market_context", market_context),
                content=content,
            )
            count += 1

    written = tracker.flush(out_file, append=False)
    click.echo(f"Ingested {count} messages → {written} events written to {out_file}")


# ---------------------------------------------------------------------------
# summarize-session
# ---------------------------------------------------------------------------


@cli.command("summarize-session")
@click.option(
    "--session-id", "-s",
    required=True,
    help="Session identifier.  Looks for <EVENTS_DIR>/<session-id>.jsonl.",
)
@click.option(
    "--events-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory containing token-event logs.  Default: {settings.EVENTS_DIR}",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory to write the summary JSONL.  Default: {settings.SUMMARIES_DIR}",
)
@click.option("--league", default=None, help="League hint, e.g. WNBA.")
@click.option("--market-type", default=None, help="Market type hint, e.g. Q2_total.")
def summarize_session(
    session_id: str,
    events_dir: Optional[str],
    out_dir: Optional[str],
    league: Optional[str],
    market_type: Optional[str],
) -> None:
    """
    Summarise one session's token-event log into a structured SessionSummary.

    \b
    Example
    -------
        python -m src.cli summarize-session \\
            --session-id sess_bootstrap_001 \\
            --league WNBA \\
            --market-type Q2_total
    """
    events_dir_path = Path(events_dir) if events_dir else settings.EVENTS_DIR
    events_file = events_dir_path / f"{session_id}.jsonl"
    out_dir_path = Path(out_dir) if out_dir else settings.SUMMARIES_DIR
    out_file = out_dir_path / f"{session_id}_summary.jsonl"

    events = load_events_from_jsonl(events_file)
    if not events:
        click.echo(f"[ERROR] No events found at {events_file}", err=True)
        raise SystemExit(1)

    summarizer = SessionSummarizer(league_hint=league, market_type_hint=market_type)
    summary = summarizer.summarize(events, session_id=session_id)

    out_dir_path.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(summary.model_dump_json() + "\n")

    click.echo(f"Summary written to {out_file}")
    click.echo(json.dumps(json.loads(summary.model_dump_json()), indent=2))


# ---------------------------------------------------------------------------
# build-training-set
# ---------------------------------------------------------------------------


@cli.command("build-training-set")
@click.option(
    "--summaries-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory containing session summary JSONLs.  Default: {settings.SUMMARIES_DIR}",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Output directory for the training JSONL.  Default: {settings.TRAINING_DIR}",
)
@click.option(
    "--out-file",
    default="training_set.jsonl",
    show_default=True,
    help="Output filename inside OUT_DIR.",
)
def build_training_set(
    summaries_dir: Optional[str],
    out_dir: Optional[str],
    out_file: str,
) -> None:
    """
    Convert all session summaries into a unified training JSONL.

    \b
    Example
    -------
        python -m src.cli build-training-set
    """
    summaries_dir_path = Path(summaries_dir) if summaries_dir else settings.SUMMARIES_DIR
    out_dir_path = Path(out_dir) if out_dir else settings.TRAINING_DIR
    out_path = out_dir_path / out_file

    summary_files = sorted(summaries_dir_path.glob("*_summary.jsonl"))
    if not summary_files:
        click.echo(f"[WARN] No summary files found in {summaries_dir_path}", err=True)
        raise SystemExit(1)

    fe = FeatureEngineer()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_path, "w", encoding="utf-8") as out_fh:
        for sf in summary_files:
            with open(sf, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        summary = validate_session_summary(json.loads(line))
                    except Exception as exc:  # noqa: BLE001
                        click.echo(f"[WARN] Skipping invalid summary in {sf}: {exc}", err=True)
                        continue
                    example = fe.transform(summary)
                    out_fh.write(example.model_dump_json() + "\n")
                    count += 1

    click.echo(f"Built {count} training examples → {out_path}")


# ---------------------------------------------------------------------------
# train-online
# ---------------------------------------------------------------------------


@cli.command("train-online")
@click.option(
    "--training-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a training JSONL produced by build-training-set.",
)
@click.option(
    "--model-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory for model persistence.  Default: {settings.MODEL_DIR}",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    type=int,
    help="Mini-batch size for partial_fit updates.",
)
def train_online(
    training_file: Optional[str],
    model_dir: Optional[str],
    batch_size: int,
) -> None:
    """
    Run partial_fit on a training JSONL and persist the updated model.

    \b
    Example
    -------
        python -m src.cli train-online \\
            --training-file data/training_sets/training_set.jsonl \\
            --batch-size 16
    """
    if training_file is None:
        training_file = str(settings.TRAINING_DIR / "training_set.jsonl")

    trainer = OnlineTrainer(model_dir=Path(model_dir) if model_dir else None)
    fe = FeatureEngineer()

    batch: list[TrainingExample] = []
    total = 0

    with open(training_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                example = validate_training_example(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[WARN] Skipping invalid example: {exc}", err=True)
                continue
            batch.append(example)
            if len(batch) >= batch_size:
                trainer.update(batch)
                total += len(batch)
                batch = []

    if batch:
        trainer.update(batch)
        total += len(batch)

    metrics = trainer.compute_metrics()
    click.echo(f"Trained on {total} examples.  Model version: {metrics.model_version}")
    click.echo(json.dumps(json.loads(metrics.model_dump_json()), indent=2))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@cli.command("evaluate")
@click.option(
    "--model-dir",
    default=None,
    type=click.Path(file_okay=False),
    help=f"Directory containing persisted models.  Default: {settings.MODEL_DIR}",
)
def evaluate(model_dir: Optional[str]) -> None:
    """
    Print current model metrics (version, accuracy, segment breakdown).

    \b
    Example
    -------
        python -m src.cli evaluate
    """
    trainer = OnlineTrainer(model_dir=Path(model_dir) if model_dir else None)
    metrics = trainer.compute_metrics()
    click.echo(json.dumps(json.loads(metrics.model_dump_json()), indent=2))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
