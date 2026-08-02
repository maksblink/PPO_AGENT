from __future__ import annotations

from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_and_eval.training.cli as cli
from train_and_eval.checkpoints.persistence import (
    PersistedCheckpoint,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
    EvaluationStatus,
    RunStatus,
)
from train_and_eval.evaluation.persistence import (
    PersistedEvaluationState,
)
from train_and_eval.training.execution import (
    ExactPPOTrainingResult,
)
from train_and_eval.training.persistence import (
    PersistedRunState,
)
from train_and_eval.training.service import (
    TrainingServiceResult,
)


class FakeEngine:
    def __init__(self) -> None:
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1


def _checkpoint(
    *,
    checkpoint_id: int,
    run_step: int,
    save_reason: CheckpointSaveReason,
) -> PersistedCheckpoint:
    return PersistedCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=51,
        run_step=run_step,
        model_step=100 + run_step,
        save_reason=save_reason,
        relative_path=(
            "artifacts/runs/00000051/checkpoints/"
            f"step_{run_step:012d}_{save_reason.value}.zip"
        ),
        absolute_path=Path(
            f"/tmp/{checkpoint_id}.zip"
        ),
        sha256="a" * 64,
        size_bytes=1234,
    )


def _evaluation(
    *,
    evaluation_id: int,
    checkpoint_id: int,
    score: float,
) -> PersistedEvaluationState:
    return PersistedEvaluationState(
        evaluation_id=evaluation_id,
        checkpoint_id=checkpoint_id,
        status=EvaluationStatus.COMPLETED,
        steps_expected=100,
        steps_completed=100,
        started_at=None,
        finished_at=None,
        balanced_score=score,
        error_type=None,
        error_message=None,
    )


def _result(
    *,
    stopped_early: bool = False,
    source_checkpoint_id: int | None = None,
) -> TrainingServiceResult:
    periodic = _checkpoint(
        checkpoint_id=91,
        run_step=50,
        save_reason=CheckpointSaveReason.PERIODIC,
    )
    final = _checkpoint(
        checkpoint_id=92,
        run_step=(80 if stopped_early else 100),
        save_reason=CheckpointSaveReason.FINAL,
    )
    best_evaluation = _evaluation(
        evaluation_id=301,
        checkpoint_id=periodic.checkpoint_id,
        score=0.25,
    )
    final_evaluation = _evaluation(
        evaluation_id=302,
        checkpoint_id=final.checkpoint_id,
        score=0.20,
    )
    completed_steps = (
        80 if stopped_early else 100
    )
    reason = (
        "balanced_score did not improve for 2 evaluations"
        if stopped_early
        else None
    )

    return TrainingServiceResult(
        run=PersistedRunState(
            run_id=51,
            name="cli-test-run",
            status=RunStatus.COMPLETED,
            training_steps_requested=100,
            training_steps_completed=completed_steps,
            data_epochs_completed=Decimal(
                "1.25000000"
            ),
            stopped_early=stopped_early,
            early_stop_reason=reason,
            started_at=None,
            finished_at=None,
            error_type=None,
            error_message=None,
        ),
        checkpoint=final,
        checkpoints=(periodic, final),
        evaluations=(
            best_evaluation,
            final_evaluation,
        ),
        best_checkpoint=periodic,
        best_evaluation=best_evaluation,
        training=ExactPPOTrainingResult(
            model_steps_before=1000,
            model_steps_after=(
                1000 + completed_steps
            ),
            local_steps_requested=100,
            local_steps_completed=completed_steps,
            rollout_sizes=(32, 32, 32, 4),
            stopped_early=stopped_early,
        ),
        early_stop_reason=reason,
        source_checkpoint_id=source_checkpoint_id,
    )


def _args(**overrides):
    values = {
        "config": "configs/experiments/example.yml",
        "verbose": 1,
        "log_interval": 1,
        "progress_bar": False,
        "traceback": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_executes_training_and_prints_success_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine()
    session_factory = object()

    monkeypatch.setattr(
        cli,
        "require_clean_git",
        lambda: object(),
    )
    calls: list[object] = []

    monkeypatch.setattr(
        cli,
        "create_database_engine",
        lambda: engine,
    )
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda received_engine: (
            calls.append(received_engine)
            or session_factory
        ),
    )

    def train(received_factory, **kwargs):
        calls.append(received_factory)
        calls.append(kwargs)
        return _result()

    monkeypatch.setattr(
        cli,
        "train_ppo_run",
        train,
    )

    stdout = StringIO()
    stderr = StringIO()
    exit_code = cli.execute_training_cli(
        _args(
            verbose=2,
            log_interval=3,
            progress_bar=True,
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert engine.dispose_calls == 1
    assert calls[0] is engine
    assert calls[1] is session_factory
    assert calls[2] == {
        "config_path": (
            "configs/experiments/example.yml"
        ),
        "verbose": 2,
        "log_interval": 3,
        "progress_bar": True,
    }

    output = stdout.getvalue()
    assert "Training run: OK" in output
    assert "Run ID:                 51" in output
    assert "Continuation source:    fresh" in output
    assert "Completed steps:        100" in output
    assert "Stopped early:          no" in output
    assert "Best balanced score:    0.25000000" in output
    assert "Best checkpoint ID:     91" in output
    assert "Final checkpoint ID:    92" in output
    assert "step_000000000100_final.zip" in output


def test_prints_resume_and_early_stopping_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        cli,
        "require_clean_git",
        lambda: object(),
    )
    monkeypatch.setattr(
        cli,
        "create_database_engine",
        lambda: engine,
    )
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda received_engine: object(),
    )
    monkeypatch.setattr(
        cli,
        "train_ppo_run",
        lambda *args, **kwargs: _result(
            stopped_early=True,
            source_checkpoint_id=77,
        ),
    )

    stdout = StringIO()
    exit_code = cli.execute_training_cli(
        _args(),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "Continuation source:    checkpoint 77" in output
    assert "Completed steps:        80" in output
    assert "Stopped early:          yes" in output
    assert (
        "Early-stop reason:     balanced_score did not improve"
        in output
    )
    assert "Final checkpoint step:  80" in output


def test_returns_one_and_prints_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine()
    error = RuntimeError(
        "synthetic training failure"
    )
    error.add_note(
        "failed state persistence also failed"
    )

    monkeypatch.setattr(
        cli,
        "require_clean_git",
        lambda: object(),
    )
    monkeypatch.setattr(
        cli,
        "create_database_engine",
        lambda: engine,
    )
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda received_engine: object(),
    )
    monkeypatch.setattr(
        cli,
        "train_ppo_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            error
        ),
    )

    stdout = StringIO()
    stderr = StringIO()
    exit_code = cli.execute_training_cli(
        _args(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert engine.dispose_calls == 1
    output = stderr.getvalue()
    assert "Training run: FAILED" in output
    assert "Error type: RuntimeError" in output
    assert "Error: synthetic training failure" in output
    assert (
        "Note: failed state persistence also failed"
        in output
    )


def test_returns_130_for_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine()

    monkeypatch.setattr(
        cli,
        "require_clean_git",
        lambda: object(),
    )
    monkeypatch.setattr(
        cli,
        "create_database_engine",
        lambda: engine,
    )
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda received_engine: object(),
    )
    monkeypatch.setattr(
        cli,
        "train_ppo_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt()
        ),
    )

    stderr = StringIO()
    exit_code = cli.execute_training_cli(
        _args(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 130
    assert engine.dispose_calls == 1
    assert "Training run: INTERRUPTED" in stderr.getvalue()


def test_dirty_git_is_rejected_before_database_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "require_clean_git",
        lambda: (_ for _ in ()).throw(
            RuntimeError("dirty repository")
        ),
    )
    monkeypatch.setattr(
        cli,
        "create_database_engine",
        lambda: (_ for _ in ()).throw(
            AssertionError(
                "Database setup must not run."
            )
        ),
    )

    stderr = StringIO()
    exit_code = cli.execute_training_cli(
        _args(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert "dirty repository" in stderr.getvalue()


def test_main_parses_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def execute(args):
        captured.update(vars(args))
        return 7

    monkeypatch.setattr(
        cli,
        "execute_training_cli",
        execute,
    )

    exit_code = cli.main(
        [
            "--config",
            "experiment.yml",
            "--verbose",
            "2",
            "--log-interval",
            "5",
            "--progress-bar",
            "--traceback",
        ]
    )

    assert exit_code == 7
    assert captured == {
        "config": "experiment.yml",
        "verbose": 2,
        "log_interval": 5,
        "progress_bar": True,
        "traceback": True,
    }


def test_parser_rejects_nonpositive_log_interval() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(
            [
                "--config",
                "experiment.yml",
                "--log-interval",
                "0",
            ]
        )

    assert captured.value.code == 2
