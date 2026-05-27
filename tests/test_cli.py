"""
CLI smoke tests
================
Tests the five CLI commands using Click's test runner so no real I/O or model
persistence is required for the critical paths.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.cli import cli


BOOTSTRAP_JSONL = Path(__file__).parent.parent / "data/bootstrap_sessions/bootstrap_session_001.jsonl"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_dirs(tmp_path):
    """Yield a dict of temporary directories matching pipeline structure."""
    dirs = {
        "events": tmp_path / "token_events",
        "summaries": tmp_path / "session_summaries",
        "training": tmp_path / "training_sets",
        "models": tmp_path / "models",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    return dirs


class TestIngestEventsCommand:
    def test_ingest_bootstrap_session(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", str(BOOTSTRAP_JSONL),
                "--session-id", "smoke_sess_001",
                "--out-dir", str(tmp_dirs["events"]),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "events written" in result.output
        events_file = tmp_dirs["events"] / "smoke_sess_001.jsonl"
        assert events_file.exists()

    def test_ingest_writes_valid_jsonl(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", str(BOOTSTRAP_JSONL),
                "--session-id", "valid_jsonl_test",
                "--out-dir", str(tmp_dirs["events"]),
            ],
        )
        assert result.exit_code == 0
        events_file = tmp_dirs["events"] / "valid_jsonl_test.jsonl"
        lines = events_file.read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "session_id" in obj

    def test_ingest_nonexistent_input_fails(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", "/nonexistent/file.jsonl",
                "--session-id", "fail_test",
            ],
        )
        assert result.exit_code != 0


class TestSummarizeSessionCommand:
    def _ingest(self, runner, tmp_dirs, session_id="smz_sess"):
        runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", str(BOOTSTRAP_JSONL),
                "--session-id", session_id,
                "--out-dir", str(tmp_dirs["events"]),
            ],
        )
        return session_id

    def test_summarize_produces_file(self, runner, tmp_dirs):
        sid = self._ingest(runner, tmp_dirs)
        result = runner.invoke(
            cli,
            [
                "summarize-session",
                "--session-id", sid,
                "--events-dir", str(tmp_dirs["events"]),
                "--out-dir", str(tmp_dirs["summaries"]),
                "--league", "WNBA",
                "--market-type", "Q2_total",
            ],
        )
        assert result.exit_code == 0, result.output
        summary_file = tmp_dirs["summaries"] / f"{sid}_summary.jsonl"
        assert summary_file.exists()

    def test_summarize_missing_session_fails(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            [
                "summarize-session",
                "--session-id", "nonexistent_sess",
                "--events-dir", str(tmp_dirs["events"]),
            ],
        )
        assert result.exit_code != 0


class TestBuildTrainingSetCommand:
    def _prepare(self, runner, tmp_dirs, session_id="bts_sess"):
        runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", str(BOOTSTRAP_JSONL),
                "--session-id", session_id,
                "--out-dir", str(tmp_dirs["events"]),
            ],
        )
        runner.invoke(
            cli,
            [
                "summarize-session",
                "--session-id", session_id,
                "--events-dir", str(tmp_dirs["events"]),
                "--out-dir", str(tmp_dirs["summaries"]),
                "--league", "WNBA",
            ],
        )

    def test_build_produces_training_jsonl(self, runner, tmp_dirs):
        self._prepare(runner, tmp_dirs)
        result = runner.invoke(
            cli,
            [
                "build-training-set",
                "--summaries-dir", str(tmp_dirs["summaries"]),
                "--out-dir", str(tmp_dirs["training"]),
            ],
        )
        assert result.exit_code == 0, result.output
        out_file = tmp_dirs["training"] / "training_set.jsonl"
        assert out_file.exists()
        lines = out_file.read_text().strip().splitlines()
        assert len(lines) >= 1

    def test_build_no_summaries_fails(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            [
                "build-training-set",
                "--summaries-dir", str(tmp_dirs["summaries"]),
            ],
        )
        assert result.exit_code != 0


class TestTrainOnlineCommand:
    def _prepare_training_file(self, runner, tmp_dirs, session_id="train_sess"):
        runner.invoke(
            cli,
            [
                "ingest-events",
                "--input", str(BOOTSTRAP_JSONL),
                "--session-id", session_id,
                "--out-dir", str(tmp_dirs["events"]),
            ],
        )
        runner.invoke(
            cli,
            [
                "summarize-session",
                "--session-id", session_id,
                "--events-dir", str(tmp_dirs["events"]),
                "--out-dir", str(tmp_dirs["summaries"]),
            ],
        )
        runner.invoke(
            cli,
            [
                "build-training-set",
                "--summaries-dir", str(tmp_dirs["summaries"]),
                "--out-dir", str(tmp_dirs["training"]),
            ],
        )

    def test_train_online_runs_to_completion(self, runner, tmp_dirs):
        self._prepare_training_file(runner, tmp_dirs)
        training_file = tmp_dirs["training"] / "training_set.jsonl"
        result = runner.invoke(
            cli,
            [
                "train-online",
                "--training-file", str(training_file),
                "--model-dir", str(tmp_dirs["models"]),
                "--batch-size", "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Trained on" in result.output

    def test_train_online_persists_model(self, runner, tmp_dirs):
        self._prepare_training_file(runner, tmp_dirs)
        training_file = tmp_dirs["training"] / "training_set.jsonl"
        runner.invoke(
            cli,
            [
                "train-online",
                "--training-file", str(training_file),
                "--model-dir", str(tmp_dirs["models"]),
            ],
        )
        model_files = list(tmp_dirs["models"].glob("*.joblib"))
        assert len(model_files) >= 1


class TestEvaluateCommand:
    def test_evaluate_unfitted_model(self, runner, tmp_dirs):
        result = runner.invoke(
            cli,
            ["evaluate", "--model-dir", str(tmp_dirs["models"])],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "model_version" in data

    def test_evaluate_after_training(self, runner, tmp_dirs):
        # Ingest → summarize → build → train → evaluate
        for sid in ["e2e_sess"]:
            runner.invoke(
                cli,
                [
                    "ingest-events",
                    "--input", str(BOOTSTRAP_JSONL),
                    "--session-id", sid,
                    "--out-dir", str(tmp_dirs["events"]),
                ],
            )
            runner.invoke(
                cli,
                [
                    "summarize-session",
                    "--session-id", sid,
                    "--events-dir", str(tmp_dirs["events"]),
                    "--out-dir", str(tmp_dirs["summaries"]),
                ],
            )
        runner.invoke(
            cli,
            [
                "build-training-set",
                "--summaries-dir", str(tmp_dirs["summaries"]),
                "--out-dir", str(tmp_dirs["training"]),
            ],
        )
        runner.invoke(
            cli,
            [
                "train-online",
                "--training-file", str(tmp_dirs["training"] / "training_set.jsonl"),
                "--model-dir", str(tmp_dirs["models"]),
            ],
        )
        result = runner.invoke(
            cli,
            ["evaluate", "--model-dir", str(tmp_dirs["models"])],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["n_examples_seen"] >= 1
