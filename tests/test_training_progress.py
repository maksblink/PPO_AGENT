from __future__ import annotations

from io import StringIO

import pytest

import train_and_eval.training.progress as progress_module
from train_and_eval.training.progress import (
    LiveTrainingProgress,
    ValidationMetricSnapshot,
)


def _validation_snapshot() -> ValidationMetricSnapshot:
    return ValidationMetricSnapshot(
        evaluation_id=9,
        checkpoint_id=11,
        run_step=100,
        trigger="final",
        balanced_score=0.12,
        agent_return=0.20,
        always_long_return=0.10,
        always_short_return=-0.10,
        agent_max_drawdown=-0.05,
        profit_factor=1.3,
        win_rate=0.55,
        market_exposure=0.40,
        round_trips=42,
    )


def test_live_progress_keeps_training_metrics_during_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(
        progress_module.time,
        "monotonic",
        lambda: clock[0],
    )
    stream = StringIO()
    progress = LiveTrainingProgress(
        stream,
        minimum_refresh_seconds=0.0,
    )

    progress.start(
        run_id=7,
        run_name="example-run",
        requested_steps=100,
        model_steps_before=500,
    )
    clock[0] = 110.0
    progress.training_update(
        completed_steps=100,
        model_steps=600,
        rollouts_completed=3,
        metrics={
            "approx_kl": 0.002,
            "clip_fraction": 0.03,
            "entropy_loss": -0.25,
            "explained_variance": 0.4,
            "learning_rate": 0.0003,
            "policy_gradient_loss": -0.01,
            "value_loss": 0.02,
            "n_updates": 30,
        },
    )
    progress.validation_started(
        run_step=100,
        checkpoint_id=11,
        trigger="final",
        expected_steps=1000,
    )
    clock[0] = 120.0
    progress.validation_update(
        completed_steps=400,
        expected_steps=1000,
    )

    output = stream.getvalue()
    assert "Phase: VALIDATION" in output
    assert "TRAIN — latest PPO update" in output
    assert "approx_kl 0.00200" in output
    assert "40.00%" in output
    assert "400/1,000" in output
    assert "Training 00:10" in output
    assert "Run wall 00:20" in output
    assert "validation time 00:10" in output


def test_live_progress_renders_completed_validation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(
        progress_module.time,
        "monotonic",
        lambda: clock[0],
    )
    stream = StringIO()
    progress = LiveTrainingProgress(
        stream,
        minimum_refresh_seconds=0.0,
    )

    progress.start(
        run_id=7,
        run_name="example-run",
        requested_steps=100,
        model_steps_before=500,
    )
    clock[0] = 110.0
    progress.training_update(
        completed_steps=100,
        model_steps=600,
        rollouts_completed=3,
        metrics={"n_updates": 30},
    )
    progress.validation_started(
        run_step=100,
        checkpoint_id=11,
        trigger="final",
        expected_steps=1000,
    )
    clock[0] = 130.0
    progress.validation_update(
        completed_steps=1000,
        expected_steps=1000,
    )
    progress.validation_completed(
        _validation_snapshot()
    )
    progress.finish(
        completed_steps=100,
        stopped_early=False,
    )

    output = stream.getvalue()
    assert "Phase: COMPLETED" in output
    assert "100.00%" in output
    assert "Completed in 00:20" in output
    assert "balanced_score 0.12000" in output
    assert "profit_factor 1.30000" in output
    assert "round_trips 42" in output
    assert "Run wall 00:30" in output
    assert "validation time 00:20" in output
    assert "\x1b[" in output
