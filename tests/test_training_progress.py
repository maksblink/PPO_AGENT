from __future__ import annotations

from io import StringIO

import pytest

import train_and_eval.training.progress as progress_module
from train_and_eval.training.progress import (
    LiveTrainingProgress,
    TrainingPreflightSnapshot,
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
    assert "evaluation time 00:10" in output


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
    assert "evaluation time 00:20" in output
    assert "\x1b[" in output


def test_live_progress_renders_training_preflight() -> None:
    stream = StringIO()
    progress = LiveTrainingProgress(
        stream,
        minimum_refresh_seconds=0.0,
    )

    progress.preflight(
        TrainingPreflightSnapshot(
            n_steps=2_048,
            batch_size=1_024,
            original_steps_per_data_epoch=69_243,
            trimmed_training_data_steps=635,
            effective_steps_per_data_epoch=68_608,
            duration_unit="data_epochs",
            duration_amount=1,
            raw_requested_steps=68_608,
            trimmed_requested_steps=0,
            resolved_requested_steps=68_608,
            training_batch_count=67,
            checkpoint_requested_steps=10_000,
            checkpoint_rollout_count=4,
            checkpoint_resolved_steps=8_192,
            checkpoint_trimmed_steps=1_808,
            evaluation_requested_steps=20_000,
            evaluation_rollout_count=9,
            evaluation_resolved_steps=18_432,
            evaluation_trimmed_steps=1_568,
            schedule_event_steps=(
                8_192,
                16_384,
                18_432,
                24_576,
                32_768,
                36_864,
                40_960,
                49_152,
                55_296,
                57_344,
                65_536,
                68_608,
            ),
            periodic_checkpoint_writes=11,
            periodic_evaluations=3,
            total_checkpoint_writes=12,
            training_segment_count=12,
            all_segments_batch_aligned=True,
        )
    )

    output = stream.getvalue()

    assert "PPO TRAINING RESOLUTION" in output
    assert "69,243" in output
    assert "69,243 % 1,024 = 635" in output
    assert "68,608" in output
    assert "floor(68,608 / 1,024) = 67 complete batches" in output
    assert "floor(10,000 / 2,048) = 4" in output
    assert "resolved: 8,192" in output
    assert "floor(20,000 / 2,048) = 9" in output
    assert "resolved: 18,432" in output
    assert "periodic checkpoint writes: 11" in output
    assert "periodic evaluation events: 3" in output
    assert "total checkpoint writes: 12" in output
    assert "training segments: 12" in output
    assert "all segments batch-aligned: YES" in output
    assert "8,192" in output
    assert "68,608" in output


def test_long_event_schedule_is_compact_but_exactly_counted() -> None:
    steps = tuple(
        index * 2_048
        for index in range(1, 31)
    )

    text = LiveTrainingProgress._event_steps_text(steps)

    assert "2,048" in text
    assert "20,480" in text
    assert "43,008" in text
    assert "61,440" in text
    assert "10 events omitted" in text
    assert "22,528" not in text
    assert "40,960" not in text
