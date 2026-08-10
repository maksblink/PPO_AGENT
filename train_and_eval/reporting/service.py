from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

from sqlalchemy import select

from train_and_eval.artifact_storage.storage import DEFAULT_ARTIFACTS_DIRECTORY
from train_and_eval.database.models import Checkpoint, Evaluation, EvaluationStatus, Run
from train_and_eval.evaluation.metrics import EvaluationMetrics
from train_and_eval.evaluation.service import replay_run_validation_checkpoint
from train_and_eval.reporting.artifacts import (
    evaluation_artifact_directory,
    persist_evaluation_source_artifacts,
    render_evaluation_plots,
    render_run_level_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReportGenerationError(RuntimeError):
    """Raised when a post-hoc report cannot be regenerated safely."""


def _metric_value(value):
    return getattr(value, "value", value)


def _assert_replay_matches_evaluation(result, evaluation: Evaluation) -> None:
    if int(result.steps_completed) != int(evaluation.steps_completed):
        raise ReportGenerationError(
            "Offline replay step count does not match the persisted evaluation."
        )

    for field in fields(EvaluationMetrics):
        name = field.name
        replayed = _metric_value(getattr(result.metrics, name))
        persisted = _metric_value(getattr(evaluation, name))

        if replayed is None or persisted is None:
            if replayed is not None or persisted is not None:
                raise ReportGenerationError(
                    f"Offline replay metric {name} does not match persisted nullability."
                )
            continue

        if isinstance(replayed, (int, float)) and isinstance(persisted, (int, float)):
            if not math.isclose(
                float(replayed),
                float(persisted),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ReportGenerationError(
                    f"Offline replay metric {name} differs: "
                    f"persisted={persisted!r}, replayed={replayed!r}."
                )
        elif replayed != persisted:
            raise ReportGenerationError(
                f"Offline replay metric {name} differs: "
                f"persisted={persisted!r}, replayed={replayed!r}."
            )


def generate_run_report(
    session_factory,
    *,
    run_id: int,
    project_root: str | Path = PROJECT_ROOT,
    artifacts_directory: str | Path = DEFAULT_ARTIFACTS_DIRECTORY,
    evaluation_id: int | None = None,
) -> tuple[Path, ...]:
    """Regenerate run/evaluation reports without inserting DB evaluations."""
    root = Path(project_root).expanduser().resolve()
    outputs: list[Path] = list(
        render_run_level_artifacts(
            session_factory,
            run_id=run_id,
            project_root=root,
            artifacts_directory=artifacts_directory,
        )
    )

    with session_factory() as session:
        run = session.get(Run, int(run_id))
        if run is None:
            raise ReportGenerationError(f"Run {run_id} does not exist.")
        statement = (
            select(Evaluation, Checkpoint)
            .join(Checkpoint, Checkpoint.id == Evaluation.checkpoint_id)
            .where(
                Checkpoint.run_id == int(run_id),
                Evaluation.status == EvaluationStatus.COMPLETED,
            )
            .order_by(Checkpoint.model_step, Evaluation.id)
        )
        if evaluation_id is not None:
            statement = statement.where(Evaluation.id == int(evaluation_id))
        rows = list(session.execute(statement).all())

    if evaluation_id is not None and not rows:
        raise ReportGenerationError(
            f"Completed evaluation {evaluation_id} does not belong to run {run_id}."
        )

    for evaluation, checkpoint in rows:
        directory = evaluation_artifact_directory(
            root, artifacts_directory, run_id, int(evaluation.id)
        )
        source_paths = (
            directory / "trajectory.parquet",
            directory / "trade_events.parquet",
            directory / "metrics.json",
        )
        if not all(path.exists() for path in source_paths):
            result = replay_run_validation_checkpoint(
                session_factory=session_factory,
                checkpoint_id=int(checkpoint.id),
                project_root=root,
                artifacts_directory=artifacts_directory,
                policy_mode=evaluation.policy_mode,
                threshold_action=evaluation.threshold_action,
                probability_threshold=evaluation.probability_threshold,
                seed=int(evaluation.seed),
            )
            _assert_replay_matches_evaluation(result, evaluation)
            directory = persist_evaluation_source_artifacts(
                result,
                project_root=root,
                artifacts_directory=artifacts_directory,
                run_id=run_id,
                evaluation_id=int(evaluation.id),
                checkpoint_id=int(checkpoint.id),
                metadata={
                    "git_commit": str(run.git_commit),
                    "git_branch": str(run.git_branch),
                    "data_sha256": str(run.data_sha256),
                    "checkpoint_sha256": str(checkpoint.sha256),
                    "replayed_offline": True,
                },
            )
        outputs.extend(render_evaluation_plots(directory))
        outputs.extend(
            path for path in (
                directory / "trajectory.parquet",
                directory / "trade_events.parquet",
                directory / "metrics.json",
            ) if path.exists()
        )

    # Stable de-duplication while preserving useful display order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in outputs:
        resolved = Path(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)
