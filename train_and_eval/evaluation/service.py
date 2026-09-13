from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from pydantic import ValidationError

from train_and_eval.artifact_storage.storage import (
    DEFAULT_ARTIFACTS_DIRECTORY,
)
from sqlalchemy.orm import Session

from train_and_eval.database.models import (
    Checkpoint,
    EvaluationDataScope,
    EvaluationPolicyMode,
    EvaluationTrigger,
    Run,
)
from train_and_eval.environment.contexts import (
    get_context_definition,
)
from train_and_eval.evaluation.persistence import (
    PersistedEvaluationState,
    complete_evaluation,
    create_pending_evaluation,
    fail_evaluation,
    mark_evaluation_running,
)
from train_and_eval.evaluation.runner import (
    EvaluationRunResult,
    run_ppo_evaluation,
)
from train_and_eval.reporting.artifacts import (
    evaluation_artifact_directory,
    persist_evaluation_source_artifacts,
    render_evaluation_result_plots,
)
from train_and_eval.market_data.load_market_data import (
    DATA_DIRECTORY,
    MANIFEST_PATH,
    load_market_data,
)
from train_and_eval.ppo.checkpoints import (
    load_persisted_ppo_checkpoint,
)
from train_and_eval.reproducibility import (
    require_clean_git,
)
from train_and_eval.run_config import (
    RunConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvaluationServiceError(RuntimeError):
    """Base error for checkpoint evaluation services."""


class EvaluationSourceNotFoundError(
    EvaluationServiceError
):
    """Raised when a checkpoint or its source run does not exist."""


class EvaluationSourceMismatchError(
    EvaluationServiceError
):
    """Raised when archived source metadata is inconsistent."""


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationSource:
    """Immutable checkpoint and source-run data needed for evaluation."""

    id: int
    run_id: int
    relative_path: str
    sha256: str
    size_bytes: int
    model_step: int

    run_seed: int
    config_schema_version: int
    normalized_config_json: dict[str, Any]

    data_path: str
    data_sha256: str
    split_index: int
    train_rows: int
    validation_rows: int
    run_git_commit: str
    run_git_branch: str
    window_metadata: dict[str, Any] | None = None


def _nonnegative_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise EvaluationServiceError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvaluationServiceError(
            f"{name} must be an integer."
        ) from error

    if result != value or result < 0:
        raise EvaluationServiceError(
            f"{name} must be a nonnegative integer."
        )

    return result


def _positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise EvaluationServiceError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvaluationServiceError(
            f"{name} must be an integer."
        ) from error

    if result != value or result < 1:
        raise EvaluationServiceError(
            f"{name} must be a positive integer."
        )

    return result


def _load_source(
    session_factory,
    *,
    checkpoint_id: int,
) -> CheckpointEvaluationSource:
    with session_factory() as session:
        checkpoint = session.get(
            Checkpoint,
            checkpoint_id,
        )

        if checkpoint is None:
            raise EvaluationSourceNotFoundError(
                f"Checkpoint {checkpoint_id} "
                "does not exist."
            )

        run = session.get(
            Run,
            int(checkpoint.run_id),
        )

        if run is None:
            raise EvaluationSourceNotFoundError(
                "Source run "
                f"{checkpoint.run_id} does not exist."
            )

        normalized_config = (
            run.normalized_config_json
        )

        if not isinstance(
            normalized_config,
            dict,
        ):
            raise EvaluationSourceMismatchError(
                "Source run normalized_config_json "
                "is not a JSON object."
            )

        return CheckpointEvaluationSource(
            id=int(checkpoint.id),
            run_id=int(checkpoint.run_id),
            relative_path=str(
                checkpoint.relative_path
            ),
            sha256=str(checkpoint.sha256),
            size_bytes=int(
                checkpoint.size_bytes
            ),
            model_step=int(
                checkpoint.model_step
            ),
            run_seed=int(run.seed),
            config_schema_version=int(
                run.config_schema_version
            ),
            normalized_config_json=dict(
                normalized_config
            ),
            data_path=str(run.data_path),
            data_sha256=str(
                run.data_sha256
            ),
            split_index=int(
                run.split_index
            ),
            train_rows=int(
                run.train_rows
            ),
            validation_rows=int(
                run.validation_rows
            ),
            run_git_commit=str(run.git_commit),
            run_git_branch=str(run.git_branch),
            window_metadata=getattr(run, "window_metadata", None),
        )


def _source_config(
    source: CheckpointEvaluationSource,
) -> RunConfig:
    try:
        config = RunConfig.model_validate(
            source.normalized_config_json
        )
    except ValidationError as error:
        raise EvaluationSourceMismatchError(
            "Source run contains an invalid "
            "normalized configuration."
        ) from error

    if (
        config.config_schema_version
        != source.config_schema_version
    ):
        raise EvaluationSourceMismatchError(
            "Source config schema version does not "
            "match the run database row."
        )

    if int(config.run.seed) != source.run_seed:
        raise EvaluationSourceMismatchError(
            "Source config seed does not match "
            "the run database row."
        )

    if str(config.data.path) != source.data_path:
        raise EvaluationSourceMismatchError(
            "Source config data path does not match "
            "the run database row."
        )

    return config


def _aware_timestamp(
    value: Any,
    *,
    name: str,
) -> datetime:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        raise EvaluationSourceMismatchError(
            f"{name} must be timezone-aware."
        )

    return timestamp.to_pydatetime()


def _load_source_market_data(
    source: CheckpointEvaluationSource,
    *,
    data_directory: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    frame = load_market_data(
        source.data_path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    loaded_sha256 = frame.attrs.get(
        "sha256"
    )

    if loaded_sha256 != source.data_sha256:
        raise EvaluationSourceMismatchError(
            "Loaded market-data SHA-256 does not "
            "match the source run."
        )

    if source.window_metadata is not None:
        if len(frame) != source.window_metadata["data_rows"]:
            raise EvaluationSourceMismatchError("Temporal source dataset row count changed")
        return frame

    expected_rows = (
        source.train_rows
        + source.validation_rows
    )

    if len(frame) != expected_rows:
        raise EvaluationSourceMismatchError(
            "Loaded market-data row count does not "
            "match the source run."
        )

    if source.split_index != source.train_rows:
        raise EvaluationSourceMismatchError(
            "Source run split_index does not match "
            "train_rows."
        )

    if (
        len(frame) - source.split_index
        != source.validation_rows
    ):
        raise EvaluationSourceMismatchError(
            "Source validation range does not match "
            "validation_rows."
        )

    return frame


def evaluate_run_validation_checkpoint(
    session_factory,
    *,
    checkpoint_id: int,
    trigger: EvaluationTrigger | str,
    policy_mode: EvaluationPolicyMode | str,
    threshold_action: int | None = None,
    probability_threshold: float | None = None,
    seed: int | None = None,
    project_root: str | Path = PROJECT_ROOT,
    data_directory: str | Path = DATA_DIRECTORY,
    manifest_path: str | Path = MANIFEST_PATH,
    artifacts_directory: str | Path = (
        DEFAULT_ARTIFACTS_DIRECTORY
    ),
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval_steps: int = 128,
    persist_trajectory: bool = False,
    render_plots: bool = False,
    evaluation_range: tuple[int, int] | None = None,
    data_scope: EvaluationDataScope | str | None = None,
) -> PersistedEvaluationState:
    """
    Evaluate one persisted checkpoint on its archived validation range.

    The service derives the data path, chronological split,
    EnvironmentSection and PPO device from the source Run row.
    """
    root = (
        Path(project_root)
        .expanduser()
        .resolve()
    )

    # This is intentionally the first external operation.
    git_state = require_clean_git(
        root
    )

    resolved_checkpoint_id = (
        _positive_integer(
            checkpoint_id,
            name="checkpoint_id",
        )
    )

    source = _load_source(
        session_factory,
        checkpoint_id=(
            resolved_checkpoint_id
        ),
    )
    config = _source_config(
        source
    )
    market_data = _load_source_market_data(
        source,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    if evaluation_range is not None:
        evaluation_start_index, evaluation_end_index = evaluation_range
        if data_scope is None or data_scope == EvaluationDataScope.RUN_VALIDATION:
            raise EvaluationSourceMismatchError("Explicit ranges require custom or out-of-sample scope")
    elif source.window_metadata is not None:
        bounds = source.window_metadata.get("validation")
        if bounds is None:
            raise EvaluationSourceMismatchError("This training run has no validation range")
        evaluation_start_index, evaluation_end_index = bounds["start_index"], bounds["end_index"]
    else:
        evaluation_start_index, evaluation_end_index = source.split_index, len(market_data)
    if not 0 < evaluation_start_index < evaluation_end_index <= len(market_data):
        raise EvaluationSourceMismatchError("Invalid evaluation range")
    context_definition = (
        get_context_definition(
            config.environment.context
        )
    )
    lookback_rows = (
        context_definition
        .required_history_rows(
            int(config.environment.window)
        )
    )

    if (
        evaluation_start_index
        < lookback_rows
    ):
        raise EvaluationSourceMismatchError(
            "Validation range does not leave "
            "enough rows for environment lookback."
        )

    evaluation_start_at = (
        _aware_timestamp(
            market_data["DT"].iloc[
                evaluation_start_index
            ],
            name="evaluation_start_at",
        )
    )
    evaluation_end_at = (
        _aware_timestamp(
            market_data["DT"].iloc[
                evaluation_end_index - 1
            ],
            name="evaluation_end_at",
        )
    )

    resolved_seed = (
        source.run_seed
        if seed is None
        else _nonnegative_integer(
            seed,
            name="seed",
        )
    )

    pending = create_pending_evaluation(
        session_factory,
        checkpoint_id=source.id,
        trigger=trigger,
        policy_mode=policy_mode,
        threshold_action=threshold_action,
        probability_threshold=(
            probability_threshold
        ),
        seed=resolved_seed,
        data_scope=data_scope or EvaluationDataScope.RUN_VALIDATION,
        data_path=source.data_path,
        data_sha256=source.data_sha256,
        data_rows=len(market_data),
        evaluation_start_index=(
            evaluation_start_index
        ),
        evaluation_end_index=(
            evaluation_end_index
        ),
        evaluation_start_at=(
            evaluation_start_at
        ),
        evaluation_end_at=(
            evaluation_end_at
        ),
        lookback_rows=lookback_rows,
        git_commit=git_state.commit,
        git_branch=git_state.branch,
    )

    evaluation_id = (
        pending.evaluation_id
    )

    try:
        mark_evaluation_running(
            session_factory,
            evaluation_id=evaluation_id,
        )

        model = load_persisted_ppo_checkpoint(
            source,
            environment=None,
            device=config.evaluation.device,
            project_root=root,
            artifacts_directory=artifacts_directory,
        )

        result = run_ppo_evaluation(
            model,
            market_data,
            config.environment,
            evaluation_start_index=evaluation_start_index,
            evaluation_end_index=evaluation_end_index,
            lookback_rows=lookback_rows,
            policy_mode=policy_mode,
            seed=resolved_seed,
            threshold_action=threshold_action,
            probability_threshold=probability_threshold,
            progress_callback=progress_callback,
            progress_interval_steps=progress_interval_steps,
        )

        persisted = complete_evaluation(
            session_factory,
            evaluation_id=evaluation_id,
            result=result,
        )

    except BaseException as error:
        try:
            fail_evaluation(
                session_factory,
                evaluation_id=evaluation_id,
                error=error,
            )
        except BaseException as failure_error:
            error.add_note(
                "Additionally, persisting the failed evaluation state failed: "
                f"{type(failure_error).__name__}: {failure_error}"
            )
        raise

    # Derived validation artifacts are explicitly non-critical: their
    # failure must never invalidate a completed evaluation or training run.
    try:
        directory = evaluation_artifact_directory(
            root, artifacts_directory, source.run_id, evaluation_id
        )
        if persist_trajectory:
            directory = persist_evaluation_source_artifacts(
                result,
                project_root=root,
                artifacts_directory=artifacts_directory,
                run_id=source.run_id,
                evaluation_id=evaluation_id,
                checkpoint_id=source.id,
                metadata={
                    "git_commit": source.run_git_commit,
                    "git_branch": source.run_git_branch,
                    "data_sha256": source.data_sha256,
                    "checkpoint_sha256": source.sha256,
                },
            )
        if render_plots:
            render_evaluation_result_plots(
                result,
                directory=directory,
            )
    except Exception as artifact_error:
        warnings.warn(
            "Validation artifact generation failed and can be retried "
            f"offline: {type(artifact_error).__name__}: {artifact_error}",
            RuntimeWarning,
            stacklevel=2,
        )

    return persisted


def replay_run_validation_checkpoint(
    *,
    checkpoint_id: int,
    project_root: str | Path = PROJECT_ROOT,
    data_directory: str | Path = DATA_DIRECTORY,
    manifest_path: str | Path = MANIFEST_PATH,
    artifacts_directory: str | Path = DEFAULT_ARTIFACTS_DIRECTORY,
    session_factory,
    policy_mode: EvaluationPolicyMode | str | None = None,
    threshold_action: int | None = None,
    probability_threshold: float | None = None,
    seed: int | None = None,
    evaluation_range: tuple[int, int] | None = None,
) -> EvaluationRunResult:
    """Replay archived run-validation without inserting an Evaluation row.

    The repository must be clean, but it may be a later commit. The offline
    report layer verifies every replayed metric against the persisted
    Evaluation row before accepting regenerated artifacts.
    """
    root = Path(project_root).expanduser().resolve()
    require_clean_git(root)
    source = _load_source(
        session_factory,
        checkpoint_id=_positive_integer(checkpoint_id, name="checkpoint_id"),
    )
    config = _source_config(source)
    market_data = _load_source_market_data(
        source, data_directory=data_directory, manifest_path=manifest_path
    )
    if evaluation_range is not None:
        start, end = evaluation_range
    elif source.window_metadata is not None:
        bounds = source.window_metadata.get("validation")
        if bounds is None:
            raise EvaluationSourceMismatchError("No validation range; supply persisted evaluation bounds")
        start, end = bounds["start_index"], bounds["end_index"]
    else:
        start, end = source.split_index, len(market_data)
    definition = get_context_definition(config.environment.context)
    lookback = definition.required_history_rows(int(config.environment.window))
    resolved_seed = source.run_seed if seed is None else _nonnegative_integer(seed, name="seed")
    resolved_policy_mode = config.evaluation.policy_mode if policy_mode is None else policy_mode
    resolved_threshold_action = (
        config.evaluation.threshold_action if policy_mode is None else threshold_action
    )
    resolved_probability_threshold = (
        config.evaluation.probability_threshold if policy_mode is None else probability_threshold
    )
    model = load_persisted_ppo_checkpoint(
        source, environment=None, device=config.evaluation.device, project_root=root,
        artifacts_directory=artifacts_directory,
    )
    return run_ppo_evaluation(
        model, market_data, config.environment,
        evaluation_start_index=start, evaluation_end_index=end,
        lookback_rows=lookback, policy_mode=resolved_policy_mode, seed=resolved_seed,
        threshold_action=resolved_threshold_action,
        probability_threshold=resolved_probability_threshold,
    )
