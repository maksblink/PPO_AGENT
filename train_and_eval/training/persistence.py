from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, TypeVar

from sqlalchemy.orm import Session

from train_and_eval.database.models import (
    Checkpoint,
    ContinuationMode,
    Run,
    RunStatus,
    TrainingDurationUnit,
    utc_now,
)
from train_and_eval.market_data.split_market_data import (
    ChronologicalMarketDataSplit,
)
from train_and_eval.reproducibility import CleanGitState
from train_and_eval.run_config import (
    FreshContinuationSection,
    LoadedRunConfig,
    ResumeContinuationSection,
)


SessionFactory = Callable[[], Session]
EnumT = TypeVar("EnumT", bound=enum.Enum)

_EPOCH_QUANTUM = Decimal("0.00000001")


class RunPersistenceError(RuntimeError):
    """Base error for training-run persistence failures."""


class RunIdentityError(RunPersistenceError):
    """Raised when a run definition or progress value is invalid."""


class RunNotFoundError(RunPersistenceError):
    """Raised when the requested run does not exist."""


class RunSourceCheckpointNotFoundError(RunPersistenceError):
    """Raised when a resumed run references a missing checkpoint."""


class RunStateTransitionError(RunPersistenceError):
    """Raised for an invalid run lifecycle transition."""


@dataclass(frozen=True, slots=True)
class PersistedRunState:
    run_id: int
    name: str
    status: RunStatus
    training_steps_requested: int
    training_steps_completed: int
    data_epochs_completed: Decimal
    stopped_early: bool
    early_stop_reason: str | None
    started_at: datetime | None
    finished_at: datetime | None
    error_type: str | None
    error_message: str | None


def _integer_value(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise RunIdentityError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RunIdentityError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise RunIdentityError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise RunIdentityError(
            f"{name} must be at least {minimum}."
        )

    return result


def _enum_value(
    enum_type: type[EnumT],
    value: EnumT | str,
    *,
    name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value

    raw_value = getattr(value, "value", value)

    try:
        return enum_type(str(raw_value))
    except ValueError as error:
        allowed = ", ".join(
            str(member.value)
            for member in enum_type
        )
        raise RunIdentityError(
            f"Unsupported {name} {raw_value!r}. "
            f"Allowed values: {allowed}."
        ) from error


def _status_value(run: Run) -> RunStatus:
    return _enum_value(
        RunStatus,
        run.status,
        name="run status",
    )


def _snapshot(run: Run) -> PersistedRunState:
    return PersistedRunState(
        run_id=int(run.id),
        name=str(run.name),
        status=_status_value(run),
        training_steps_requested=int(
            run.training_steps_requested
        ),
        training_steps_completed=int(
            run.training_steps_completed
        ),
        data_epochs_completed=Decimal(
            run.data_epochs_completed
        ),
        stopped_early=bool(
            run.stopped_early
        ),
        early_stop_reason=(
            run.early_stop_reason
        ),
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_type=run.error_type,
        error_message=run.error_message,
    )


def _load_run_for_update(
    session: Session,
    run_id: int,
) -> Run:
    run = session.get(
        Run,
        run_id,
        with_for_update=True,
    )

    if run is None:
        raise RunNotFoundError(
            f"Run {run_id} does not exist."
        )

    return run


def _data_epochs_completed(
    *,
    training_steps_completed: int,
    steps_per_data_epoch: int,
) -> Decimal:
    return (
        Decimal(training_steps_completed)
        / Decimal(steps_per_data_epoch)
    ).quantize(
        _EPOCH_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def create_pending_run(
    session_factory: SessionFactory,
    *,
    loaded_config: LoadedRunConfig,
    split: ChronologicalMarketDataSplit,
    git_state: CleanGitState,
    training_steps_requested: int,
    source_checkpoint_id: int | None = None,
) -> PersistedRunState:
    """Create one pending training run from verified immutable inputs."""
    config = loaded_config.config
    manifest_entry = loaded_config.data_manifest_entry

    if manifest_entry is None:
        raise RunIdentityError(
            "Run config must be loaded with market-data verification enabled."
        )

    manifest_path = manifest_entry.get("path")
    manifest_sha256 = manifest_entry.get("sha256")
    manifest_rows = manifest_entry.get("rows")

    if manifest_entry.get("status") != "okay":
        raise RunIdentityError(
            "Market-data manifest entry must have status='okay'."
        )

    if manifest_path != config.data.path:
        raise RunIdentityError(
            "Manifest data path does not match the run config."
        )

    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise RunIdentityError(
            "Manifest does not contain a valid data SHA-256."
        )

    if manifest_rows != (
        split.train_rows + split.validation_rows
    ):
        raise RunIdentityError(
            "Manifest row count does not match the chronological split."
        )

    if split.split_index != split.train_rows:
        raise RunIdentityError(
            "Chronological split_index must equal train_rows."
        )

    resolved_training_steps_requested = _integer_value(
        training_steps_requested,
        name="training_steps_requested",
        minimum=1,
    )

    normalized_config_json = json.loads(
        loaded_config.normalized_json
    )
    normalized_config_sha256 = hashlib.sha256(
        loaded_config.normalized_json.encode("utf-8")
    ).hexdigest()

    continuation = config.continuation

    if isinstance(continuation, FreshContinuationSection):
        continuation_mode = ContinuationMode.FRESH

        if source_checkpoint_id is not None:
            raise RunIdentityError(
                "A fresh run cannot reference a source checkpoint."
            )

        resolved_source_checkpoint_id = None

    elif isinstance(continuation, ResumeContinuationSection):
        continuation_mode = ContinuationMode.RESUME

        if config.run.name == continuation.source_run:
            raise RunIdentityError(
                "A resumed run must have a different name "
                "than its source run."
            )

        if source_checkpoint_id is None:
            raise RunIdentityError(
                "A resumed run requires source_checkpoint_id."
            )

        resolved_source_checkpoint_id = _integer_value(
            source_checkpoint_id,
            name="source_checkpoint_id",
            minimum=1,
        )

    else:
        raise RunIdentityError(
            "Unsupported continuation configuration."
        )

    with session_factory() as session:
        try:
            if resolved_source_checkpoint_id is not None:
                source_checkpoint = session.get(
                    Checkpoint,
                    resolved_source_checkpoint_id,
                )

                if source_checkpoint is None:
                    raise RunSourceCheckpointNotFoundError(
                        "Source checkpoint "
                        f"{resolved_source_checkpoint_id} does not exist."
                    )

                source_run = session.get(
                    Run,
                    int(source_checkpoint.run_id),
                )

                if source_run is None:
                    raise RunSourceCheckpointNotFoundError(
                        "Source checkpoint run "
                        f"{source_checkpoint.run_id} does not exist."
                    )

                if source_run.name != continuation.source_run:
                    raise RunIdentityError(
                        "Source checkpoint does not belong to the "
                        "run named by continuation.source_run."
                    )

            run = Run(
                name=config.run.name,
                status=RunStatus.PENDING,
                continuation_mode=continuation_mode,
                source_checkpoint_id=resolved_source_checkpoint_id,
                git_commit=git_state.commit,
                git_branch=git_state.branch,
                config_schema_version=config.config_schema_version,
                seed=int(config.run.seed),
                config_sha256=loaded_config.sha256,
                normalized_config_sha256=(
                    normalized_config_sha256
                ),
                raw_config_yaml=loaded_config.raw_yaml,
                normalized_config_json=normalized_config_json,
                data_path=config.data.path,
                data_sha256=manifest_sha256,
                duration_unit=_enum_value(
                    TrainingDurationUnit,
                    config.training.duration_unit,
                    name="duration_unit",
                ),
                duration_amount=int(
                    config.training.duration_amount
                ),
                split_index=int(split.split_index),
                train_rows=int(split.train_rows),
                validation_rows=int(split.validation_rows),
                steps_per_data_epoch=int(
                    split.steps_per_data_epoch
                ),
                training_steps_requested=int(
                resolved_training_steps_requested
            ),
            training_steps_completed=0,
                data_epochs_completed=Decimal("0"),
                stopped_early=False,
                early_stop_reason=None,
                started_at=None,
                finished_at=None,
                error_type=None,
                error_message=None,
            )

            session.add(run)
            session.flush()
            state = _snapshot(run)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def mark_run_running(
    session_factory: SessionFactory,
    *,
    run_id: int,
) -> PersistedRunState:
    """Transition one pending run to running."""
    resolved_id = _integer_value(
        run_id,
        name="run_id",
        minimum=1,
    )

    with session_factory() as session:
        try:
            run = _load_run_for_update(
                session,
                resolved_id,
            )
            current_status = _status_value(run)

            if current_status != RunStatus.PENDING:
                raise RunStateTransitionError(
                    f"Run {resolved_id} cannot move from "
                    f"{current_status.value} to running."
                )

            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            run.finished_at = None
            run.stopped_early = False
            run.early_stop_reason = None
            run.error_type = None
            run.error_message = None

            session.flush()
            state = _snapshot(run)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def update_run_progress(
    session_factory: SessionFactory,
    *,
    run_id: int,
    training_steps_completed: int,
) -> PersistedRunState:
    """Persist monotonic local training progress for one running run."""
    resolved_id = _integer_value(
        run_id,
        name="run_id",
        minimum=1,
    )
    resolved_steps = _integer_value(
        training_steps_completed,
        name="training_steps_completed",
        minimum=0,
    )

    with session_factory() as session:
        try:
            run = _load_run_for_update(
                session,
                resolved_id,
            )
            current_status = _status_value(run)

            if current_status != RunStatus.RUNNING:
                raise RunStateTransitionError(
                    f"Run {resolved_id} cannot update progress "
                    f"while {current_status.value}."
                )

            previous_steps = int(
                run.training_steps_completed
            )
            requested_steps = int(
                run.training_steps_requested
            )

            if resolved_steps < previous_steps:
                raise RunIdentityError(
                    "training_steps_completed cannot decrease."
                )

            if resolved_steps > requested_steps:
                raise RunIdentityError(
                    "training_steps_completed cannot exceed "
                    "training_steps_requested."
                )

            run.training_steps_completed = resolved_steps
            run.data_epochs_completed = _data_epochs_completed(
                training_steps_completed=resolved_steps,
                steps_per_data_epoch=int(
                    run.steps_per_data_epoch
                ),
            )

            session.flush()
            state = _snapshot(run)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def complete_run(
    session_factory: SessionFactory,
    *,
    run_id: int,
    stopped_early: bool = False,
    early_stop_reason: str | None = None,
) -> PersistedRunState:
    """Mark a fully progressed or intentionally early-stopped run completed."""
    resolved_id = _integer_value(
        run_id,
        name="run_id",
        minimum=1,
    )

    with session_factory() as session:
        try:
            run = _load_run_for_update(
                session,
                resolved_id,
            )
            current_status = _status_value(run)

            if current_status != RunStatus.RUNNING:
                raise RunStateTransitionError(
                    f"Run {resolved_id} cannot move from "
                    f"{current_status.value} to completed."
                )

            completed_steps = int(
                run.training_steps_completed
            )
            requested_steps = int(
                run.training_steps_requested
            )

            if not isinstance(
                stopped_early,
                bool,
            ):
                raise RunIdentityError(
                    "stopped_early must be a boolean."
                )

            resolved_reason: str | None = None

            if stopped_early:
                if not isinstance(
                    early_stop_reason,
                    str,
                ):
                    raise RunIdentityError(
                        "early_stop_reason must be a string "
                        "when stopped_early is true."
                    )

                resolved_reason = (
                    early_stop_reason.strip()
                )

                if not resolved_reason:
                    raise RunIdentityError(
                        "early_stop_reason must not be empty."
                    )

                if not 0 < completed_steps < requested_steps:
                    raise RunStateTransitionError(
                        "An early-stopped run must persist at least "
                        "one step and fewer than all requested steps."
                    )
            else:
                if early_stop_reason is not None:
                    raise RunIdentityError(
                        "early_stop_reason must be null when "
                        "stopped_early is false."
                    )

                if completed_steps != requested_steps:
                    raise RunStateTransitionError(
                        "A run cannot complete before all requested "
                        "training steps are persisted unless it is "
                        "explicitly marked as early-stopped."
                    )

            run.status = RunStatus.COMPLETED
            run.finished_at = utc_now()
            run.stopped_early = stopped_early
            run.early_stop_reason = resolved_reason
            run.error_type = None
            run.error_message = None

            session.flush()
            state = _snapshot(run)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def fail_run(
    session_factory: SessionFactory,
    *,
    run_id: int,
    error: BaseException,
    training_steps_completed: int | None = None,
) -> PersistedRunState:
    """Mark a pending or running run as failed."""
    resolved_id = _integer_value(
        run_id,
        name="run_id",
        minimum=1,
    )

    if not isinstance(error, BaseException):
        raise RunIdentityError(
            "error must be an exception."
        )

    with session_factory() as session:
        try:
            run = _load_run_for_update(
                session,
                resolved_id,
            )
            current_status = _status_value(run)

            if current_status not in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
            }:
                raise RunStateTransitionError(
                    f"Run {resolved_id} cannot move from "
                    f"{current_status.value} to failed."
                )

            if training_steps_completed is None:
                resolved_steps = int(
                    run.training_steps_completed
                )
            else:
                resolved_steps = _integer_value(
                    training_steps_completed,
                    name="training_steps_completed",
                    minimum=0,
                )

            previous_steps = int(
                run.training_steps_completed
            )
            requested_steps = int(
                run.training_steps_requested
            )

            if resolved_steps < previous_steps:
                raise RunIdentityError(
                    "training_steps_completed cannot decrease."
                )

            if resolved_steps > requested_steps:
                raise RunIdentityError(
                    "training_steps_completed cannot exceed "
                    "training_steps_requested."
                )

            error_type = (
                type(error).__name__.strip()
                or "Exception"
            )[:255]
            error_message = str(error).strip()

            if not error_message:
                error_message = repr(error).strip()

            if not error_message:
                error_message = (
                    "Training failed without an error message."
                )

            now = utc_now()

            if run.started_at is None:
                run.started_at = now

            run.status = RunStatus.FAILED
            run.finished_at = now
            run.stopped_early = False
            run.early_stop_reason = None
            run.training_steps_completed = resolved_steps
            run.data_epochs_completed = _data_epochs_completed(
                training_steps_completed=resolved_steps,
                steps_per_data_epoch=int(
                    run.steps_per_data_epoch
                ),
            )
            run.error_type = error_type
            run.error_message = error_message

            session.flush()
            state = _snapshot(run)
            session.commit()

        except Exception:
            session.rollback()
            raise

    return state
