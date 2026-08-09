from __future__ import annotations

from io import StringIO

from train_and_eval.training.progress import (
    LiveTrainingProgress,
    ValidationMetricSnapshot,
)


def test_live_progress_renders_training_and_validation_metrics() -> None:
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
    progress.training_update(
        completed_steps=25,
        model_steps=525,
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
        run_step=25,
        checkpoint_id=11,
        trigger="scheduled",
    )
    progress.validation_completed(
        ValidationMetricSnapshot(
            evaluation_id=9,
            checkpoint_id=11,
            run_step=25,
            trigger="scheduled",
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
    )

    output = stream.getvalue()
    assert "PPO RUN #7  example-run" in output
    assert "25.00%" in output
    assert "25/100" in output
    assert "TRAIN (latest PPO update)" in output
    assert "approx_kl 0.00200" in output
    assert "VALIDATION: completed (scheduled, evaluation 9)" in output
    assert "balanced_score 0.12000" in output
    assert "profit_factor 1.30000" in output
    assert "round_trips 42" in output
    assert "\x1b[" in output


def test_finish_leaves_final_progress_at_one_hundred_percent() -> None:
    stream = StringIO()
    progress = LiveTrainingProgress(
        stream,
        minimum_refresh_seconds=0.0,
    )
    progress.start(
        run_id=1,
        run_name="done",
        requested_steps=10,
        model_steps_before=0,
    )
    progress.finish(
        completed_steps=10,
        stopped_early=False,
    )

    output = stream.getvalue()
    assert "Phase: COMPLETED" in output
    assert "100.00%" in output
    assert "10/10" in output
