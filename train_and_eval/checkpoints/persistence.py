from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from train_and_eval.artifact_storage.storage import (
    ArtifactStorage,
)
from train_and_eval.database.models import (
    Checkpoint,
    CheckpointSaveReason,
    Run,
)


class CheckpointPersistenceError(RuntimeError):
    """Base error for checkpoint persistence failures."""


class CheckpointRunNotFoundError(
    CheckpointPersistenceError
):
    """Raised when the target run does not exist."""


class CheckpointIdentityError(
    CheckpointPersistenceError
):
    """Raised when checkpoint identity fields are invalid."""


@dataclass(frozen=True, slots=True)
class PersistedCheckpoint:
    checkpoint_id: int
    run_id: int
    run_step: int
    model_step: int
    save_reason: CheckpointSaveReason
    relative_path: str
    absolute_path: Path
    sha256: str
    size_bytes: int


SessionFactory = Callable[[], Session]


def _integer_value(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise CheckpointIdentityError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CheckpointIdentityError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise CheckpointIdentityError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise CheckpointIdentityError(
            f"{name} must be at least {minimum}."
        )

    return result


def _checkpoint_save_reason(
    value: CheckpointSaveReason | str,
) -> CheckpointSaveReason:
    if isinstance(value, CheckpointSaveReason):
        return value

    raw_value = getattr(
        value,
        "value",
        value,
    )

    try:
        return CheckpointSaveReason(
            str(raw_value)
        )
    except ValueError as error:
        allowed = ", ".join(
            reason.value
            for reason in CheckpointSaveReason
        )

        raise CheckpointIdentityError(
            "Unsupported checkpoint save_reason "
            f"{raw_value!r}. Allowed values: {allowed}."
        ) from error


def persist_checkpoint_file(
    session_factory: SessionFactory,
    storage: ArtifactStorage,
    *,
    source_path: str | Path,
    run_id: int,
    run_step: int,
    model_step: int,
    save_reason: CheckpointSaveReason | str,
) -> PersistedCheckpoint:
    """
    Store one immutable checkpoint file and create its database row.

    The source file must already be complete.

    If construction or database flush fails, the newly stored file is
    removed. If commit itself fails, the file is retained because the
    final database commit outcome may be ambiguous after a connection
    failure.
    """
    resolved_run_id = _integer_value(
        run_id,
        name="run_id",
        minimum=1,
    )
    resolved_run_step = _integer_value(
        run_step,
        name="run_step",
        minimum=0,
    )
    resolved_model_step = _integer_value(
        model_step,
        name="model_step",
        minimum=0,
    )
    resolved_save_reason = (
        _checkpoint_save_reason(
            save_reason
        )
    )

    if resolved_model_step < resolved_run_step:
        raise CheckpointIdentityError(
            "model_step must be greater than or "
            "equal to run_step."
        )

    with session_factory() as session:
        run = session.get(
            Run,
            resolved_run_id,
        )

        if run is None:
            raise CheckpointRunNotFoundError(
                f"Run {resolved_run_id} does not exist."
            )

        stored_artifact = (
            storage.store_checkpoint_file(
                source_path,
                run_id=resolved_run_id,
                run_step=resolved_run_step,
                save_reason=resolved_save_reason,
            )
        )

        try:
            checkpoint = Checkpoint(
                run_id=resolved_run_id,
                run_step=resolved_run_step,
                model_step=resolved_model_step,
                save_reason=resolved_save_reason,
                relative_path=(
                    stored_artifact.relative_path
                ),
                sha256=stored_artifact.sha256,
                size_bytes=(
                    stored_artifact.size_bytes
                ),
            )

            session.add(checkpoint)
            session.flush()

            checkpoint_id = int(
                checkpoint.id
            )

        except Exception:
            session.rollback()

            stored_artifact.absolute_path.unlink(
                missing_ok=True
            )

            raise

        try:
            session.commit()
        except Exception:
            session.rollback()

            # Do not remove the artifact here. A failed commit call can
            # be ambiguous if PostgreSQL committed but the connection
            # was interrupted before confirmation reached the client.
            raise

    return PersistedCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=resolved_run_id,
        run_step=resolved_run_step,
        model_step=resolved_model_step,
        save_reason=resolved_save_reason,
        relative_path=(
            stored_artifact.relative_path
        ),
        absolute_path=(
            stored_artifact.absolute_path
        ),
        sha256=stored_artifact.sha256,
        size_bytes=stored_artifact.size_bytes,
    )
