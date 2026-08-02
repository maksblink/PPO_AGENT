from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from train_and_eval.database.models import (
    Checkpoint,
    ContinuationMode,
    Run,
    RunStatus,
)


class RunRegistryError(RuntimeError):
    """Base error for read-only run-registry queries."""


class RunRegistryNotFoundError(RunRegistryError):
    """Raised when a requested run does not exist."""


def _with_history():
    return selectinload(
        Run.checkpoints
    ).selectinload(
        Checkpoint.evaluations
    )


def load_runs(
    session: Session,
    *,
    status: RunStatus | None = None,
    continuation_mode: ContinuationMode | None = None,
    name_contains: str | None = None,
    stopped_early: bool = False,
    limit: int = 20,
) -> list[Run]:
    """Load recent runs together with checkpoint/evaluation history."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")

    statement = (
        select(Run)
        .options(_with_history())
        .order_by(
            Run.created_at.desc(),
            Run.id.desc(),
        )
        .limit(limit)
    )

    if status is not None:
        statement = statement.where(
            Run.status == status
        )

    if continuation_mode is not None:
        statement = statement.where(
            Run.continuation_mode
            == continuation_mode
        )

    if name_contains:
        statement = statement.where(
            Run.name.icontains(
                name_contains,
                autoescape=True,
            )
        )

    if stopped_early:
        statement = statement.where(
            Run.stopped_early.is_(True)
        )

    return list(
        session.scalars(statement).all()
    )


def load_run(
    session: Session,
    *,
    run_id: int,
) -> Run:
    """Load one run and all checkpoint/evaluation history."""
    if run_id < 1:
        raise ValueError("run_id must be a positive integer")

    statement = (
        select(Run)
        .options(_with_history())
        .where(Run.id == run_id)
    )
    run = session.scalars(
        statement
    ).one_or_none()

    if run is None:
        raise RunRegistryNotFoundError(
            f"Run {run_id} does not exist."
        )

    return run
