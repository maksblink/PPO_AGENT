from __future__ import annotations

import pytest

from train_and_eval.training.scheduling import (
    TrainingScheduleError,
    build_training_schedule,
)


def test_unifies_checkpoint_and_evaluation_boundaries() -> None:
    schedule = build_training_schedule(
        total_steps=10,
        checkpoint_every_steps=4,
        eval_every_steps=6,
    )

    assert [
        event.run_step
        for event in schedule
    ] == [4, 6, 8, 10]
    assert [
        event.checkpoint_due
        for event in schedule
    ] == [True, True, True, True]
    assert [
        event.evaluation_due
        for event in schedule
    ] == [False, True, False, True]
    assert [
        event.final
        for event in schedule
    ] == [False, False, False, True]


def test_final_step_replaces_duplicate_periodic_work() -> None:
    schedule = build_training_schedule(
        total_steps=10,
        checkpoint_every_steps=5,
        eval_every_steps=5,
    )

    assert [
        event.run_step
        for event in schedule
    ] == [5, 10]
    assert schedule[-1].final is True
    assert schedule[-1].evaluation_due is True


def test_short_run_still_gets_final_checkpoint_and_evaluation() -> None:
    schedule = build_training_schedule(
        total_steps=3,
        checkpoint_every_steps=100,
        eval_every_steps=100,
    )

    assert len(schedule) == 1
    assert schedule[0].run_step == 3
    assert schedule[0].checkpoint_due is True
    assert schedule[0].evaluation_due is True
    assert schedule[0].final is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_steps", 0),
        ("checkpoint_every_steps", 0),
        ("eval_every_steps", 0),
    ],
)
def test_rejects_nonpositive_schedule_values(
    field: str,
    value: int,
) -> None:
    arguments = {
        "total_steps": 10,
        "checkpoint_every_steps": 4,
        "eval_every_steps": 6,
    }
    arguments[field] = value

    with pytest.raises(
        TrainingScheduleError,
        match="positive integer",
    ):
        build_training_schedule(
            **arguments
        )


def test_aligns_periodic_boundaries_to_rollout_steps() -> None:
    schedule = build_training_schedule(
        total_steps=68_608,
        checkpoint_every_steps=10_000,
        eval_every_steps=20_000,
        rollout_steps=2_048,
    )

    assert [
        event.run_step
        for event in schedule
    ] == [
        10_240,
        20_480,
        30_720,
        40_960,
        51_200,
        61_440,
        68_608,
    ]

    assert all(
        event.run_step % 2_048 == 0
        for event in schedule[:-1]
    )
