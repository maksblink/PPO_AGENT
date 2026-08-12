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
        resolved_rollout_steps = _positive_integer(
            rollout_steps,
            name="rollout_steps",
        )

        def align_interval(interval: int) -> int:
            rollout_count = max(
                1,
                round(
                    interval
                    / resolved_rollout_steps
                ),
            )
            return (
                rollout_count
                * resolved_rollout_steps
            )

        resolved_checkpoint_interval = align_interval(
            resolved_checkpoint_interval
        )
        resolved_eval_interval = align_interval(
            resolved_eval_interval
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
