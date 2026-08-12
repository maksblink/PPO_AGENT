from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TrainingScheduleError(ValueError):
    """Raised when periodic training events cannot be scheduled."""


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """One exact local run step at which persistence work is due."""

    run_step: int
    checkpoint_due: bool
    evaluation_due: bool
    final: bool


def _positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise TrainingScheduleError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TrainingScheduleError(
            f"{name} must be an integer."
        ) from error

    if result != value or result < 1:
        raise TrainingScheduleError(
            f"{name} must be a positive integer."
        )

    return result


@dataclass(frozen=True, slots=True)
class PeriodicIntervalResolution:
    requested_steps: int
    rollout_steps: int
    rollout_count: int
    resolved_steps: int
    trimmed_steps: int


def resolve_periodic_interval(
    *,
    requested_steps: int,
    rollout_steps: int,
    name: str,
) -> PeriodicIntervalResolution:
    resolved_requested = _positive_integer(
        requested_steps,
        name=name,
    )
    resolved_rollout = _positive_integer(
        rollout_steps,
        name="rollout_steps",
    )

    rollout_count = (
        resolved_requested
        // resolved_rollout
    )

    if rollout_count < 1:
        raise TrainingScheduleError(
            f"{name}={resolved_requested}; "
            f"floor({resolved_requested} / {resolved_rollout}) "
            f"= {rollout_count} complete rollouts. "
            "At least one complete PPO rollout is required. "
            f"Minimum valid value: {resolved_rollout} steps."
        )

    resolved_steps = (
        rollout_count
        * resolved_rollout
    )

    return PeriodicIntervalResolution(
        requested_steps=resolved_requested,
        rollout_steps=resolved_rollout,
        rollout_count=rollout_count,
        resolved_steps=resolved_steps,
        trimmed_steps=(
            resolved_requested
            - resolved_steps
        ),
    )


def build_training_schedule(
    *,
    total_steps: int,
    checkpoint_every_steps: int,
    eval_every_steps: int,
    rollout_steps: int | None = None,
) -> tuple[TrainingEvent, ...]:
    """
    Build exact periodic checkpoint and validation-evaluation boundaries.

    Every evaluation boundary also persists a checkpoint, because an
    Evaluation row always references one immutable Checkpoint. The final
    local step is represented once: it receives a FINAL checkpoint and a
    FINAL evaluation instead of duplicate periodic work at the same step.
    """
    resolved_total = _positive_integer(
        total_steps,
        name="total_steps",
    )
    resolved_checkpoint_interval = _positive_integer(
        checkpoint_every_steps,
        name="checkpoint_every_steps",
    )
    resolved_eval_interval = _positive_integer(
        eval_every_steps,
        name="eval_every_steps",
    )

    if rollout_steps is not None:
        checkpoint_resolution = resolve_periodic_interval(
            requested_steps=resolved_checkpoint_interval,
            rollout_steps=rollout_steps,
            name="checkpoint_every_steps",
        )
        evaluation_resolution = resolve_periodic_interval(
            requested_steps=resolved_eval_interval,
            rollout_steps=rollout_steps,
            name="eval_every_steps",
        )

        resolved_checkpoint_interval = (
            checkpoint_resolution.resolved_steps
        )
        resolved_eval_interval = (
            evaluation_resolution.resolved_steps
        )

    checkpoint_steps = set(
        range(
            resolved_checkpoint_interval,
            resolved_total,
            resolved_checkpoint_interval,
        )
    )
    evaluation_steps = set(
        range(
            resolved_eval_interval,
            resolved_total,
            resolved_eval_interval,
        )
    )

    event_steps = sorted(
        checkpoint_steps
        | evaluation_steps
        | {resolved_total}
    )

    return tuple(
        TrainingEvent(
            run_step=step,
            checkpoint_due=(
                step in checkpoint_steps
                or step in evaluation_steps
                or step == resolved_total
            ),
            evaluation_due=(
                step in evaluation_steps
                or step == resolved_total
            ),
            final=(
                step == resolved_total
            ),
        )
        for step in event_steps
    )
