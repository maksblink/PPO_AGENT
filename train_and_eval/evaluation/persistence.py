from __future__ import annotations

import enum
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy.orm import Session

from train_and_eval.database.models import (
    Checkpoint,
    Evaluation,
    EvaluationCurrentStreakType,
    EvaluationDataScope,
    EvaluationPolicyMode,
    EvaluationStatus,
    EvaluationTrigger,
    utc_now,
)
from train_and_eval.evaluation.metrics import (
    EvaluationMetrics,
)
from train_and_eval.evaluation.runner import (
    EvaluationRunResult,
)


SessionFactory = Callable[[], Session]
EnumT = TypeVar(
    "EnumT",
    bound=enum.Enum,
)

_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)
_GIT_COMMIT_PATTERN = re.compile(
    r"^[0-9a-f]{7,64}$"
)

_EVALUATION_METRIC_FIELDS = tuple(
    field.name
    for field in fields(
        EvaluationMetrics
    )
)


class EvaluationPersistenceError(RuntimeError):
    """Base error for evaluation persistence failures."""


class EvaluationIdentityError(
    EvaluationPersistenceError
):
    """Raised when evaluation identity fields are invalid."""


class EvaluationCheckpointNotFoundError(
    EvaluationPersistenceError
):
    """Raised when the checkpoint does not exist."""


class EvaluationNotFoundError(
    EvaluationPersistenceError
):
    """Raised when the evaluation does not exist."""


class EvaluationStateTransitionError(
    EvaluationPersistenceError
):
    """Raised for an invalid evaluation status transition."""


class EvaluationResultMismatchError(
    EvaluationPersistenceError
):
    """Raised when a result does not match its database row."""


@dataclass(frozen=True, slots=True)
class PersistedEvaluationState:
    evaluation_id: int
    checkpoint_id: int
    status: EvaluationStatus
    steps_expected: int
    steps_completed: int
    started_at: datetime | None
    finished_at: datetime | None
    balanced_score: float | None
    error_type: str | None
    error_message: str | None


def _integer_value(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise EvaluationIdentityError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvaluationIdentityError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise EvaluationIdentityError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise EvaluationIdentityError(
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

    raw_value = getattr(
        value,
        "value",
        value,
    )

    try:
        return enum_type(
            str(raw_value)
        )
    except ValueError as error:
        allowed = ", ".join(
            str(member.value)
            for member in enum_type
        )

        raise EvaluationIdentityError(
            f"Unsupported {name} "
            f"{raw_value!r}. Allowed values: "
            f"{allowed}."
        ) from error


def _nonempty_text(
    value: Any,
    *,
    name: str,
    maximum_length: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise EvaluationIdentityError(
            f"{name} must be a string."
        )

    result = value.strip()

    if not result:
        raise EvaluationIdentityError(
            f"{name} must not be empty."
        )

    if (
        maximum_length is not None
        and len(result) > maximum_length
    ):
        raise EvaluationIdentityError(
            f"{name} must contain at most "
            f"{maximum_length} characters."
        )

    return result


def _aware_datetime(
    value: Any,
    *,
    name: str,
) -> datetime:
    if not isinstance(value, datetime):
        raise EvaluationIdentityError(
            f"{name} must be a datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise EvaluationIdentityError(
            f"{name} must be timezone-aware."
        )

    return value


def _sha256_value(
    value: Any,
) -> str:
    result = _nonempty_text(
        value,
        name="data_sha256",
        maximum_length=64,
    )

    if not _SHA256_PATTERN.fullmatch(
        result
    ):
        raise EvaluationIdentityError(
            "data_sha256 must contain exactly "
            "64 lowercase hexadecimal characters."
        )

    return result


def _git_commit_value(
    value: Any,
) -> str:
    result = _nonempty_text(
        value,
        name="git_commit",
        maximum_length=64,
    )

    if not _GIT_COMMIT_PATTERN.fullmatch(
        result
    ):
        raise EvaluationIdentityError(
            "git_commit must be a lowercase "
            "hexadecimal Git commit identifier."
        )

    return result


def _threshold_fields(
    *,
    policy_mode: EvaluationPolicyMode,
    threshold_action: int | None,
    probability_threshold: float | None,
) -> tuple[int | None, float | None]:
    if policy_mode in {
        EvaluationPolicyMode.DETERMINISTIC_ARGMAX,
        EvaluationPolicyMode.STOCHASTIC_SAMPLE,
    }:
        if threshold_action is not None:
            raise EvaluationIdentityError(
                "threshold_action must be null "
                "for non-threshold policy modes."
            )

        if probability_threshold is not None:
            raise EvaluationIdentityError(
                "probability_threshold must be null "
                "for non-threshold policy modes."
            )

        return None, None

    if probability_threshold is None:
        raise EvaluationIdentityError(
            "probability_threshold is required "
            "for probability_threshold mode."
        )

    if isinstance(
        probability_threshold,
        bool,
    ):
        raise EvaluationIdentityError(
            "probability_threshold must be numeric."
        )

    try:
        resolved_threshold = float(
            probability_threshold
        )
    except (TypeError, ValueError) as error:
        raise EvaluationIdentityError(
            "probability_threshold must be numeric."
        ) from error

    if not math.isfinite(
        resolved_threshold
    ):
        raise EvaluationIdentityError(
            "probability_threshold must be finite."
        )

    if not 0.0 <= resolved_threshold <= 1.0:
        raise EvaluationIdentityError(
            "probability_threshold must be "
            "between 0 and 1."
        )

    resolved_action: int | None = None

    if threshold_action is not None:
        resolved_action = _integer_value(
            threshold_action,
            name="threshold_action",
            minimum=0,
        )

        if resolved_action != 1:
            raise EvaluationIdentityError(
                "threshold_action must be 1 or null."
            )

    return (
        resolved_action,
        resolved_threshold,
    )


def _status_value(
    evaluation: Evaluation,
) -> EvaluationStatus:
    return _enum_value(
        EvaluationStatus,
        evaluation.status,
        name="evaluation status",
    )


def _snapshot(
    evaluation: Evaluation,
) -> PersistedEvaluationState:
    return PersistedEvaluationState(
        evaluation_id=int(
            evaluation.id
        ),
        checkpoint_id=int(
            evaluation.checkpoint_id
        ),
        status=_status_value(
            evaluation
        ),
        steps_expected=int(
            evaluation.steps_expected
        ),
        steps_completed=int(
            evaluation.steps_completed
        ),
        started_at=evaluation.started_at,
        finished_at=evaluation.finished_at,
        balanced_score=(
            None
            if evaluation.balanced_score
            is None
            else float(
                evaluation.balanced_score
            )
        ),
        error_type=evaluation.error_type,
        error_message=(
            evaluation.error_message
        ),
    )


def _load_evaluation_for_update(
    session: Session,
    evaluation_id: int,
) -> Evaluation:
    evaluation = session.get(
        Evaluation,
        evaluation_id,
        with_for_update=True,
    )

    if evaluation is None:
        raise EvaluationNotFoundError(
            f"Evaluation {evaluation_id} "
            "does not exist."
        )

    return evaluation


def create_pending_evaluation(
    session_factory: SessionFactory,
    *,
    checkpoint_id: int,
    trigger: EvaluationTrigger | str,
    policy_mode: EvaluationPolicyMode | str,
    threshold_action: int | None,
    probability_threshold: float | None,
    seed: int,
    data_scope: EvaluationDataScope | str,
    data_path: str,
    data_sha256: str,
    data_rows: int,
    evaluation_start_index: int,
    evaluation_end_index: int,
    evaluation_start_at: datetime,
    evaluation_end_at: datetime,
    lookback_rows: int,
    git_commit: str,
    git_branch: str,
) -> PersistedEvaluationState:
    """Create one immutable pending evaluation definition."""
    resolved_checkpoint_id = _integer_value(
        checkpoint_id,
        name="checkpoint_id",
        minimum=1,
    )
    resolved_trigger = _enum_value(
        EvaluationTrigger,
        trigger,
        name="trigger",
    )
    resolved_policy_mode = _enum_value(
        EvaluationPolicyMode,
        policy_mode,
        name="policy_mode",
    )
    (
        resolved_threshold_action,
        resolved_probability_threshold,
    ) = _threshold_fields(
        policy_mode=resolved_policy_mode,
        threshold_action=threshold_action,
        probability_threshold=(
            probability_threshold
        ),
    )
    resolved_seed = _integer_value(
        seed,
        name="seed",
        minimum=0,
    )
    resolved_data_scope = _enum_value(
        EvaluationDataScope,
        data_scope,
        name="data_scope",
    )
    resolved_data_path = _nonempty_text(
        data_path,
        name="data_path",
    )
    resolved_data_sha256 = (
        _sha256_value(
            data_sha256
        )
    )
    resolved_data_rows = _integer_value(
        data_rows,
        name="data_rows",
        minimum=1,
    )
    resolved_start_index = _integer_value(
        evaluation_start_index,
        name="evaluation_start_index",
        minimum=0,
    )
    resolved_end_index = _integer_value(
        evaluation_end_index,
        name="evaluation_end_index",
        minimum=1,
    )
    resolved_lookback_rows = (
        _integer_value(
            lookback_rows,
            name="lookback_rows",
            minimum=1,
        )
    )
    resolved_start_at = _aware_datetime(
        evaluation_start_at,
        name="evaluation_start_at",
    )
    resolved_end_at = _aware_datetime(
        evaluation_end_at,
        name="evaluation_end_at",
    )
    resolved_git_commit = (
        _git_commit_value(
            git_commit
        )
    )
    resolved_git_branch = _nonempty_text(
        git_branch,
        name="git_branch",
        maximum_length=255,
    )

    if (
        resolved_end_index
        <= resolved_start_index
    ):
        raise EvaluationIdentityError(
            "evaluation_end_index must be "
            "greater than evaluation_start_index."
        )

    if (
        resolved_end_index
        > resolved_data_rows
    ):
        raise EvaluationIdentityError(
            "evaluation_end_index cannot exceed "
            "data_rows."
        )

    if (
        resolved_start_index
        < resolved_lookback_rows
    ):
        raise EvaluationIdentityError(
            "evaluation_start_index does not leave "
            "enough earlier rows for lookback."
        )

    if resolved_end_at < resolved_start_at:
        raise EvaluationIdentityError(
            "evaluation_end_at cannot be earlier "
            "than evaluation_start_at."
        )

    steps_expected = (
        resolved_end_index
        - resolved_start_index
    )

    with session_factory() as session:
        try:
            checkpoint = session.get(
                Checkpoint,
                resolved_checkpoint_id,
            )

            if checkpoint is None:
                raise (
                    EvaluationCheckpointNotFoundError(
                        "Checkpoint "
                        f"{resolved_checkpoint_id} "
                        "does not exist."
                    )
                )

            evaluation = Evaluation(
                checkpoint_id=(
                    resolved_checkpoint_id
                ),
                status=EvaluationStatus.PENDING,
                trigger=resolved_trigger,
                policy_mode=(
                    resolved_policy_mode
                ),
                threshold_action=(
                    resolved_threshold_action
                ),
                probability_threshold=(
                    resolved_probability_threshold
                ),
                seed=resolved_seed,
                data_scope=(
                    resolved_data_scope
                ),
                data_path=resolved_data_path,
                data_sha256=(
                    resolved_data_sha256
                ),
                data_rows=resolved_data_rows,
                evaluation_start_index=(
                    resolved_start_index
                ),
                evaluation_end_index=(
                    resolved_end_index
                ),
                evaluation_start_at=(
                    resolved_start_at
                ),
                evaluation_end_at=(
                    resolved_end_at
                ),
                lookback_rows=(
                    resolved_lookback_rows
                ),
                steps_expected=(
                    steps_expected
                ),
                steps_completed=0,
                started_at=None,
                finished_at=None,
                git_commit=(
                    resolved_git_commit
                ),
                git_branch=(
                    resolved_git_branch
                ),
                error_type=None,
                error_message=None,
            )

            session.add(evaluation)
            session.flush()

            state = _snapshot(
                evaluation
            )

            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def mark_evaluation_running(
    session_factory: SessionFactory,
    *,
    evaluation_id: int,
) -> PersistedEvaluationState:
    """Transition one pending evaluation to running."""
    resolved_id = _integer_value(
        evaluation_id,
        name="evaluation_id",
        minimum=1,
    )

    with session_factory() as session:
        try:
            evaluation = (
                _load_evaluation_for_update(
                    session,
                    resolved_id,
                )
            )

            current_status = _status_value(
                evaluation
            )

            if current_status != (
                EvaluationStatus.PENDING
            ):
                raise (
                    EvaluationStateTransitionError(
                        "Evaluation "
                        f"{resolved_id} cannot move "
                        f"from {current_status.value} "
                        "to running."
                    )
                )

            evaluation.status = (
                EvaluationStatus.RUNNING
            )
            evaluation.started_at = utc_now()
            evaluation.finished_at = None
            evaluation.error_type = None
            evaluation.error_message = None

            session.flush()

            state = _snapshot(
                evaluation
            )

            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def _apply_metrics(
    evaluation: Evaluation,
    metrics: EvaluationMetrics,
) -> None:
    metric_values = metrics.as_dict()

    if set(metric_values) != set(
        _EVALUATION_METRIC_FIELDS
    ):
        raise EvaluationResultMismatchError(
            "EvaluationMetrics fields changed "
            "unexpectedly."
        )

    missing_columns = [
        name
        for name in _EVALUATION_METRIC_FIELDS
        if not hasattr(
            Evaluation,
            name,
        )
    ]

    if missing_columns:
        raise EvaluationResultMismatchError(
            "Evaluation table is missing metric "
            "columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    for name in _EVALUATION_METRIC_FIELDS:
        value = metric_values[name]

        if name == "current_streak_type":
            value = (
                EvaluationCurrentStreakType(
                    str(value)
                )
            )

        setattr(
            evaluation,
            name,
            value,
        )


def complete_evaluation(
    session_factory: SessionFactory,
    *,
    evaluation_id: int,
    result: EvaluationRunResult,
) -> PersistedEvaluationState:
    """Persist all metrics and mark a running evaluation completed."""
    resolved_id = _integer_value(
        evaluation_id,
        name="evaluation_id",
        minimum=1,
    )

    with session_factory() as session:
        try:
            evaluation = (
                _load_evaluation_for_update(
                    session,
                    resolved_id,
                )
            )

            current_status = _status_value(
                evaluation
            )

            if current_status != (
                EvaluationStatus.RUNNING
            ):
                raise (
                    EvaluationStateTransitionError(
                        "Evaluation "
                        f"{resolved_id} cannot move "
                        f"from {current_status.value} "
                        "to completed."
                    )
                )

            if (
                int(result.steps_completed)
                != int(
                    evaluation.steps_expected
                )
            ):
                raise (
                    EvaluationResultMismatchError(
                        "Result steps_completed does "
                        "not match steps_expected."
                    )
                )

            if (
                int(
                    result
                    .evaluation_start_index
                )
                != int(
                    evaluation
                    .evaluation_start_index
                )
                or int(
                    result
                    .evaluation_end_index
                )
                != int(
                    evaluation
                    .evaluation_end_index
                )
                or int(
                    result.lookback_rows
                )
                != int(
                    evaluation.lookback_rows
                )
            ):
                raise (
                    EvaluationResultMismatchError(
                        "Result evaluation range does "
                        "not match the database row."
                    )
                )

            _apply_metrics(
                evaluation,
                result.metrics,
            )

            evaluation.steps_completed = (
                int(result.steps_completed)
            )
            evaluation.status = (
                EvaluationStatus.COMPLETED
            )
            evaluation.finished_at = utc_now()
            evaluation.error_type = None
            evaluation.error_message = None

            session.flush()

            state = _snapshot(
                evaluation
            )

            session.commit()

        except Exception:
            session.rollback()
            raise

    return state


def fail_evaluation(
    session_factory: SessionFactory,
    *,
    evaluation_id: int,
    error: BaseException,
    steps_completed: int | None = None,
) -> PersistedEvaluationState:
    """Mark a pending or running evaluation as failed."""
    resolved_id = _integer_value(
        evaluation_id,
        name="evaluation_id",
        minimum=1,
    )

    if not isinstance(
        error,
        BaseException,
    ):
        raise EvaluationIdentityError(
            "error must be an exception."
        )

    with session_factory() as session:
        try:
            evaluation = (
                _load_evaluation_for_update(
                    session,
                    resolved_id,
                )
            )

            current_status = _status_value(
                evaluation
            )

            if current_status not in {
                EvaluationStatus.PENDING,
                EvaluationStatus.RUNNING,
            }:
                raise (
                    EvaluationStateTransitionError(
                        "Evaluation "
                        f"{resolved_id} cannot move "
                        f"from {current_status.value} "
                        "to failed."
                    )
                )

            if steps_completed is None:
                resolved_steps = int(
                    evaluation.steps_completed
                )
            else:
                resolved_steps = (
                    _integer_value(
                        steps_completed,
                        name="steps_completed",
                        minimum=0,
                    )
                )

            if resolved_steps > int(
                evaluation.steps_expected
            ):
                raise EvaluationIdentityError(
                    "steps_completed cannot exceed "
                    "steps_expected."
                )

            error_type = (
                type(error).__name__.strip()
                or "Exception"
            )[:255]

            error_message = str(
                error
            ).strip()

            if not error_message:
                error_message = repr(
                    error
                ).strip()

            if not error_message:
                error_message = (
                    "Evaluation failed without "
                    "an error message."
                )

            now = utc_now()

            if evaluation.started_at is None:
                evaluation.started_at = now

            evaluation.status = (
                EvaluationStatus.FAILED
            )
            evaluation.finished_at = now
            evaluation.steps_completed = (
                resolved_steps
            )
            evaluation.error_type = (
                error_type
            )
            evaluation.error_message = (
                error_message
            )

            session.flush()

            state = _snapshot(
                evaluation
            )

            session.commit()

        except Exception:
            session.rollback()
            raise

    return state
