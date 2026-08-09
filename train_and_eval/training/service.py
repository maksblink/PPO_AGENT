from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import select

from train_and_eval.artifact_storage.storage import (
    DEFAULT_ARTIFACTS_DIRECTORY,
    ArtifactStorage,
)
from train_and_eval.database.models import (
    Checkpoint,
    CheckpointSaveReason,
    Evaluation,
    EvaluationDataScope,
    EvaluationStatus,
    Run,
)
from train_and_eval.environment.trading_environment import (
    TradingEnvironment,
)
from train_and_eval.evaluation.persistence import (
    PersistedEvaluationState,
)
from train_and_eval.evaluation.service import (
    evaluate_run_validation_checkpoint,
)
from train_and_eval.market_data.load_market_data import (
    DATA_DIRECTORY,
    MANIFEST_PATH,
    load_market_data,
)
from train_and_eval.market_data.split_market_data import (
    ChronologicalMarketDataSplit,
    split_market_data_chronologically,
)
from train_and_eval.ppo.adapter import (
    create_ppo_model,
)
from train_and_eval.ppo.checkpoints import (
    load_persisted_ppo_checkpoint,
    persist_ppo_checkpoint,
)
from train_and_eval.reproducibility import (
    require_clean_git,
)
from train_and_eval.run_config import (
    FreshContinuationSection,
    LoadedRunConfig,
    ResumeContinuationSection,
    RunConfig,
    load_run_config,
    validate_resume_compatibility,
)
from train_and_eval.training.early_stopping import (
    BalancedScoreEarlyStopping,
    EarlyStoppingDecision,
)
from train_and_eval.training.execution import (
    ExactPPOTrainingResult,
    PPOTrainingUpdate,
    learn_ppo_exact_timesteps,
)
from train_and_eval.training.progress import (
    TrainingProgressReporter,
    ValidationMetricSnapshot,
)
from train_and_eval.training.persistence import (
    PersistedRunState,
    complete_run,
    create_pending_run,
    fail_run,
    mark_run_running,
    update_run_progress,
)
from train_and_eval.training.scheduling import (
    build_training_schedule,
)
from train_and_eval.checkpoints.persistence import (
    PersistedCheckpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrainingServiceError(RuntimeError):
    """Base error for complete PPO training services."""


class TrainingSourceNotFoundError(
    TrainingServiceError
):
    """Raised when a resume source run or checkpoint does not exist."""


class TrainingSourceMismatchError(
    TrainingServiceError
):
    """Raised when archived resume metadata is inconsistent."""


class TrainingStoppedEarlyError(
    TrainingServiceError
):
    """Raised when PPO stops before all requested local steps complete."""


@dataclass(frozen=True, slots=True)
class ResumeTrainingSource:
    """Immutable source checkpoint and configuration for resumed training."""

    checkpoint_id: int
    run_id: int
    relative_path: str
    sha256: str
    size_bytes: int
    model_step: int
    source_config: RunConfig


@dataclass(frozen=True, slots=True)
class TrainingServiceResult:
    """Completed run, checkpoints, evaluations, and aggregate training."""

    run: PersistedRunState
    checkpoint: PersistedCheckpoint
    checkpoints: tuple[PersistedCheckpoint, ...]
    evaluations: tuple[PersistedEvaluationState, ...]
    best_checkpoint: PersistedCheckpoint
    best_evaluation: PersistedEvaluationState
    training: ExactPPOTrainingResult
    early_stop_reason: str | None
    source_checkpoint_id: int | None


def _source_config(
    source_run: Run,
) -> RunConfig:
    normalized_config = (
        source_run.normalized_config_json
    )

    if not isinstance(
        normalized_config,
        dict,
    ):
        raise TrainingSourceMismatchError(
            "Source run normalized_config_json "
            "is not a JSON object."
        )

    try:
        config = RunConfig.model_validate(
            normalized_config
        )
    except ValidationError as error:
        raise TrainingSourceMismatchError(
            "Source run contains an invalid "
            "normalized configuration."
        ) from error

    if (
        int(source_run.config_schema_version)
        != int(config.config_schema_version)
    ):
        raise TrainingSourceMismatchError(
            "Source config schema version does not "
            "match the source run database row."
        )

    if int(source_run.seed) != int(
        config.run.seed
    ):
        raise TrainingSourceMismatchError(
            "Source config seed does not match "
            "the source run database row."
        )

    if str(source_run.data_path) != str(
        config.data.path
    ):
        raise TrainingSourceMismatchError(
            "Source config data path does not match "
            "the source run database row."
        )

    return config


def _checkpoint_source(
    checkpoint: Checkpoint,
    *,
    source_run: Run,
    source_config: RunConfig,
) -> ResumeTrainingSource:
    if int(checkpoint.run_id) != int(
        source_run.id
    ):
        raise TrainingSourceMismatchError(
            "Selected checkpoint does not belong "
            "to the requested source run."
        )

    return ResumeTrainingSource(
        checkpoint_id=int(checkpoint.id),
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
        source_config=source_config,
    )


def _resolve_resume_source(
    session_factory,
    *,
    current_config: RunConfig,
) -> ResumeTrainingSource:
    continuation = (
        current_config.continuation
    )

    if not isinstance(
        continuation,
        ResumeContinuationSection,
    ):
        raise TrainingSourceMismatchError(
            "Resume source resolution requires "
            "a resume continuation config."
        )

    with session_factory() as session:
        source_run = session.scalar(
            select(Run).where(
                Run.name
                == continuation.source_run
            )
        )

        if source_run is None:
            raise TrainingSourceNotFoundError(
                "Source run "
                f"{continuation.source_run!r} "
                "does not exist."
            )

        source_config = _source_config(
            source_run
        )
        validate_resume_compatibility(
            current_config,
            source_config,
        )

        selector = (
            continuation.checkpoint
            .strip()
            .lower()
        )

        if selector == "final":
            checkpoint = session.scalar(
                select(Checkpoint).where(
                    Checkpoint.run_id
                    == int(source_run.id),
                    Checkpoint.save_reason
                    == CheckpointSaveReason.FINAL,
                )
            )

        elif selector == "best":
            checkpoint = session.scalar(
                select(Checkpoint)
                .join(
                    Evaluation,
                    Evaluation.checkpoint_id
                    == Checkpoint.id,
                )
                .where(
                    Checkpoint.run_id
                    == int(source_run.id),
                    Evaluation.status
                    == EvaluationStatus.COMPLETED,
                    Evaluation.data_scope
                    == EvaluationDataScope.RUN_VALIDATION,
                    Evaluation.balanced_score
                    .is_not(None),
                )
                .order_by(
                    Evaluation.balanced_score.desc(),
                    Evaluation.finished_at.desc().nullslast(),
                    Evaluation.id.desc(),
                )
                .limit(1)
            )

        elif selector.isdecimal():
            checkpoint = session.get(
                Checkpoint,
                int(selector),
            )

        else:
            raise TrainingSourceMismatchError(
                "Unsupported resume checkpoint selector "
                f"{continuation.checkpoint!r}. "
                "Use 'best', 'final', or a numeric "
                "checkpoint ID."
            )

        if checkpoint is None:
            raise TrainingSourceNotFoundError(
                "No checkpoint matched selector "
                f"{continuation.checkpoint!r} "
                "for source run "
                f"{continuation.source_run!r}."
            )

        return _checkpoint_source(
            checkpoint,
            source_run=source_run,
            source_config=source_config,
        )


def _load_and_split_data(
    loaded_config: LoadedRunConfig,
    *,
    data_directory: str | Path,
    manifest_path: str | Path,
) -> tuple[
    pd.DataFrame,
    ChronologicalMarketDataSplit,
]:
    config = loaded_config.config
    market_data = load_market_data(
        config.data.path,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )
    split = split_market_data_chronologically(
        market_data,
        train_ratio=float(
            config.data.train_ratio
        ),
        window=int(
            config.environment.window
        ),
        context=str(
            config.environment.context
        ),
    )

    return market_data, split


def _local_model_steps(
    model: Any | None,
    *,
    model_steps_before: int | None,
    maximum: int,
) -> int:
    if (
        model is None
        or model_steps_before is None
    ):
        return 0

    try:
        current_steps = int(
            model.num_timesteps
        )
    except (TypeError, ValueError, AttributeError):
        return 0

    completed = max(
        0,
        current_steps
        - int(model_steps_before),
    )

    return min(
        completed,
        int(maximum),
    )


def _progress_call(
    progress_reporter: TrainingProgressReporter | None,
    method_name: str,
    /,
    *args,
    **kwargs,
) -> None:
    if progress_reporter is None:
        return

    try:
        method = getattr(progress_reporter, method_name)
        method(*args, **kwargs)
    except BaseException:
        # Terminal reporting is observational and must never alter
        # checkpointing, evaluation, early stopping, or training.
        return


def _validation_progress_metrics(
    session_factory,
    *,
    evaluation_id: int,
    checkpoint_id: int,
    run_step: int,
    trigger: str,
) -> ValidationMetricSnapshot:
    with session_factory() as session:
        evaluation = session.get(
            Evaluation,
            int(evaluation_id),
        )

        if evaluation is None:
            raise TrainingServiceError(
                f"Evaluation {evaluation_id} does not exist after completion."
            )

        if int(evaluation.checkpoint_id) != int(checkpoint_id):
            raise TrainingServiceError(
                "Completed evaluation checkpoint does not match the training event."
            )

        return ValidationMetricSnapshot(
            evaluation_id=int(evaluation.id),
            checkpoint_id=int(evaluation.checkpoint_id),
            run_step=int(run_step),
            trigger=str(trigger),
            balanced_score=(
                None if evaluation.balanced_score is None
                else float(evaluation.balanced_score)
            ),
            agent_return=(
                None if evaluation.agent_return is None
                else float(evaluation.agent_return)
            ),
            always_long_return=(
                None if evaluation.always_long_return is None
                else float(evaluation.always_long_return)
            ),
            always_short_return=(
                None if evaluation.always_short_return is None
                else float(evaluation.always_short_return)
            ),
            agent_max_drawdown=(
                None if evaluation.agent_max_drawdown is None
                else float(evaluation.agent_max_drawdown)
            ),
            profit_factor=(
                None if evaluation.profit_factor is None
                else float(evaluation.profit_factor)
            ),
            win_rate=(
                None if evaluation.win_rate is None
                else float(evaluation.win_rate)
            ),
            market_exposure=(
                None if evaluation.market_exposure is None
                else float(evaluation.market_exposure)
            ),
            round_trips=(
                None if evaluation.round_trips is None
                else int(evaluation.round_trips)
            ),
        )


def train_ppo_run(
    session_factory,
    *,
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    data_directory: str | Path = DATA_DIRECTORY,
    manifest_path: str | Path = MANIFEST_PATH,
    artifacts_directory: str | Path = (
        DEFAULT_ARTIFACTS_DIRECTORY
    ),
    verbose: int = 0,
    log_interval: int | None = 1,
    progress_bar: bool = False,
    progress_reporter: TrainingProgressReporter | None = None,
) -> TrainingServiceResult:
    """
    Execute one complete fresh or resumed PPO training run.

    Training is split at exact checkpoint and evaluation boundaries. Every
    evaluation receives one immutable checkpoint. The final local step is
    persisted once as a FINAL checkpoint and evaluated once with trigger
    FINAL. Consecutive non-improving scheduled evaluations may stop the
    run early; the stopping step then receives its own FINAL checkpoint
    and FINAL evaluation.
    """
    root = (
        Path(project_root)
        .expanduser()
        .resolve()
    )

    # This must remain the first external operation.
    git_state = require_clean_git(
        root
    )

    loaded_config = load_run_config(
        config_path,
        verify_data=True,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )
    config = loaded_config.config
    _, split = _load_and_split_data(
        loaded_config,
        data_directory=data_directory,
        manifest_path=manifest_path,
    )

    continuation = config.continuation
    resume_source: (
        ResumeTrainingSource | None
    ) = None

    if isinstance(
        continuation,
        ResumeContinuationSection,
    ):
        resume_source = _resolve_resume_source(
            session_factory,
            current_config=config,
        )
    elif not isinstance(
        continuation,
        FreshContinuationSection,
    ):
        raise TrainingSourceMismatchError(
            "Unsupported continuation configuration."
        )

    pending = create_pending_run(
        session_factory,
        loaded_config=loaded_config,
        split=split,
        git_state=git_state,
        source_checkpoint_id=(
            None
            if resume_source is None
            else resume_source.checkpoint_id
        ),
    )
    run_id = pending.run_id
    requested_steps = int(
        pending.training_steps_requested
    )
    schedule = build_training_schedule(
        total_steps=requested_steps,
        checkpoint_every_steps=int(
            config.evaluation
            .checkpoint_every_steps
        ),
        eval_every_steps=int(
            config.evaluation
            .eval_every_steps
        ),
    )

    model: Any | None = None
    model_steps_before: int | None = None
    final_checkpoint: (
        PersistedCheckpoint | None
    ) = None
    latest_checkpoint_step: int | None = None
    persisted_checkpoints: list[
        PersistedCheckpoint
    ] = []
    persisted_evaluations: list[
        PersistedEvaluationState
    ] = []
    early_stopping = BalancedScoreEarlyStopping(
        patience=int(
            config.evaluation
            .early_stop_patience_evals
        )
    )
    best_checkpoint: PersistedCheckpoint | None = None
    best_evaluation: PersistedEvaluationState | None = None
    last_decision: EarlyStoppingDecision | None = None
    early_stop_reason: str | None = None
    completed_steps = 0
    rollout_sizes: list[int] = []

    try:
        mark_run_running(
            session_factory,
            run_id=run_id,
        )

        environment = TradingEnvironment(
            split.train_data,
            config.environment,
            start_index=(
                split.training_start_index
            ),
        )

        if resume_source is None:
            model = create_ppo_model(
                environment,
                config.ppo,
                seed=int(config.run.seed),
                verbose=verbose,
            )
        else:
            model = load_persisted_ppo_checkpoint(
                resume_source,
                environment=environment,
                device=config.ppo.device,
                training_config=config.ppo,
                seed=int(config.run.seed),
                project_root=root,
                artifacts_directory=(
                    artifacts_directory
                ),
            )

        model_steps_before = int(
            model.num_timesteps
        )
        storage = ArtifactStorage(
            project_root=root,
            artifacts_directory=(
                artifacts_directory
            ),
        )

        _progress_call(
            progress_reporter,
            "start",
            run_id=run_id,
            run_name=config.run.name,
            requested_steps=requested_steps,
            model_steps_before=model_steps_before,
        )

        training_progress_every_steps = int(
            config.logging.training_progress_every_steps
        )
        next_training_progress_step = (
            training_progress_every_steps
        )
        validation_progress_every_steps = int(
            config.logging.validation_progress_every_steps
        )

        def report_validation_update(
            validation_steps_completed: int,
            validation_steps_expected: int,
        ) -> None:
            _progress_call(
                progress_reporter,
                "validation_update",
                completed_steps=validation_steps_completed,
                expected_steps=validation_steps_expected,
            )

        for event in schedule:
            segment_steps = (
                int(event.run_step)
                - completed_steps
            )
            expected_model_steps_before = (
                int(model_steps_before)
                + completed_steps
            )

            segment_completed_before = completed_steps
            rollout_count_before = len(rollout_sizes)

            def report_training_update(
                update: PPOTrainingUpdate,
            ) -> None:
                nonlocal next_training_progress_step

                run_steps_completed = (
                    segment_completed_before
                    + update.local_steps_completed
                )
                segment_finished = (
                    run_steps_completed >= int(event.run_step)
                )

                if (
                    run_steps_completed < next_training_progress_step
                    and not segment_finished
                ):
                    return

                while (
                    next_training_progress_step
                    <= run_steps_completed
                ):
                    next_training_progress_step += (
                        training_progress_every_steps
                    )

                _progress_call(
                    progress_reporter,
                    "training_update",
                    completed_steps=run_steps_completed,
                    model_steps=update.model_steps,
                    rollouts_completed=(
                        rollout_count_before
                        + update.rollout_iteration
                    ),
                    metrics=update.metrics,
                )

            segment_result = (
                learn_ppo_exact_timesteps(
                    model,
                    total_timesteps=(
                        segment_steps
                    ),
                    log_interval=(
                        None
                        if progress_reporter is not None
                        else log_interval
                    ),
                    progress_bar=(
                        False
                        if progress_reporter is not None
                        else progress_bar
                    ),
                    update_callback=(
                        report_training_update
                        if progress_reporter is not None
                        else None
                    ),
                )
            )

            if segment_result.stopped_early:
                raise TrainingStoppedEarlyError(
                    "PPO stopped before the next "
                    "scheduled training boundary."
                )

            if (
                segment_result.model_steps_before
                != expected_model_steps_before
                or segment_result.model_steps_after
                != expected_model_steps_before
                + segment_steps
            ):
                raise TrainingStoppedEarlyError(
                    "PPO segment reported inconsistent "
                    "absolute model steps."
                )

            if (
                segment_result.local_steps_completed
                != segment_steps
            ):
                raise TrainingStoppedEarlyError(
                    "PPO completed an unexpected number "
                    "of steps in a scheduled segment."
                )

            completed_steps += int(
                segment_result
                .local_steps_completed
            )
            rollout_sizes.extend(
                segment_result.rollout_sizes
            )

            if completed_steps != int(
                event.run_step
            ):
                raise TrainingStoppedEarlyError(
                    "PPO did not reach the exact "
                    "scheduled run step."
                )

            update_run_progress(
                session_factory,
                run_id=run_id,
                training_steps_completed=(
                    completed_steps
                ),
            )

            save_reason = (
                CheckpointSaveReason.FINAL
                if event.final
                else CheckpointSaveReason.PERIODIC
            )
            checkpoint = persist_ppo_checkpoint(
                session_factory,
                storage,
                model,
                run_id=run_id,
                run_step=completed_steps,
                save_reason=save_reason,
            )
            persisted_checkpoints.append(
                checkpoint
            )
            latest_checkpoint_step = (
                completed_steps
            )

            if event.final:
                final_checkpoint = checkpoint

            if event.evaluation_due:
                evaluation_trigger = (
                    "final"
                    if event.final
                    else "scheduled"
                )

                _progress_call(
                    progress_reporter,
                    "validation_started",
                    run_step=completed_steps,
                    checkpoint_id=checkpoint.checkpoint_id,
                    trigger=evaluation_trigger,
                    expected_steps=int(split.validation_rows),
                )

                evaluation = (
                    evaluate_run_validation_checkpoint(
                        session_factory,
                        checkpoint_id=(
                            checkpoint.checkpoint_id
                        ),
                        trigger=evaluation_trigger,
                        policy_mode=(
                            config.evaluation
                            .policy_mode
                        ),
                        threshold_action=(
                            config.evaluation
                            .threshold_action
                        ),
                        probability_threshold=(
                            config.evaluation
                            .probability_threshold
                        ),
                        seed=int(
                            config.run.seed
                        ),
                        progress_callback=(
                            report_validation_update
                            if progress_reporter is not None
                            else None
                        ),
                        progress_interval_steps=(
                            validation_progress_every_steps
                        ),
                        project_root=root,
                        data_directory=(
                            data_directory
                        ),
                        manifest_path=(
                            manifest_path
                        ),
                        artifacts_directory=(
                            artifacts_directory
                        ),
                    )
                )
                persisted_evaluations.append(
                    evaluation
                )

                if progress_reporter is not None:
                    try:
                        validation_metrics = _validation_progress_metrics(
                            session_factory,
                            evaluation_id=evaluation.evaluation_id,
                            checkpoint_id=checkpoint.checkpoint_id,
                            run_step=completed_steps,
                            trigger=evaluation_trigger,
                        )
                    except BaseException:
                        validation_metrics = None

                    if validation_metrics is not None:
                        _progress_call(
                            progress_reporter,
                            "validation_completed",
                            validation_metrics,
                        )

                last_decision = (
                    early_stopping.observe(
                        checkpoint_id=(
                            checkpoint.checkpoint_id
                        ),
                        evaluation_id=(
                            evaluation.evaluation_id
                        ),
                        balanced_score=(
                            evaluation.balanced_score
                        ),
                    )
                )

                if last_decision.improved:
                    best_checkpoint = checkpoint
                    best_evaluation = evaluation

                if (
                    not event.final
                    and last_decision.should_stop
                ):
                    early_stop_reason = (
                        "balanced_score did not improve "
                        f"for {last_decision.no_improvement_evals} "
                        "consecutive scheduled evaluations; "
                        "best score was "
                        f"{last_decision.best_score:.17g}."
                    )

                    final_checkpoint = (
                        persist_ppo_checkpoint(
                            session_factory,
                            storage,
                            model,
                            run_id=run_id,
                            run_step=completed_steps,
                            save_reason=(
                                CheckpointSaveReason
                                .FINAL
                            ),
                        )
                    )
                    persisted_checkpoints.append(
                        final_checkpoint
                    )

                    _progress_call(
                        progress_reporter,
                        "validation_started",
                        run_step=completed_steps,
                        checkpoint_id=(
                            final_checkpoint.checkpoint_id
                        ),
                        trigger="final",
                        expected_steps=int(split.validation_rows),
                    )

                    final_evaluation = (
                        evaluate_run_validation_checkpoint(
                            session_factory,
                            checkpoint_id=(
                                final_checkpoint
                                .checkpoint_id
                            ),
                            trigger="final",
                            policy_mode=(
                                config.evaluation
                                .policy_mode
                            ),
                            threshold_action=(
                                config.evaluation
                                .threshold_action
                            ),
                            probability_threshold=(
                                config.evaluation
                                .probability_threshold
                            ),
                            seed=int(
                                config.run.seed
                            ),
                            progress_callback=(
                                report_validation_update
                                if progress_reporter is not None
                                else None
                            ),
                            progress_interval_steps=(
                                validation_progress_every_steps
                            ),
                            project_root=root,
                            data_directory=(
                                data_directory
                            ),
                            manifest_path=(
                                manifest_path
                            ),
                            artifacts_directory=(
                                artifacts_directory
                            ),
                        )
                    )
                    persisted_evaluations.append(
                        final_evaluation
                    )

                    if progress_reporter is not None:
                        try:
                            final_validation_metrics = (
                                _validation_progress_metrics(
                                    session_factory,
                                    evaluation_id=(
                                        final_evaluation.evaluation_id
                                    ),
                                    checkpoint_id=(
                                        final_checkpoint.checkpoint_id
                                    ),
                                    run_step=completed_steps,
                                    trigger="final",
                                )
                            )
                        except BaseException:
                            final_validation_metrics = None

                        if final_validation_metrics is not None:
                            _progress_call(
                                progress_reporter,
                                "validation_completed",
                                final_validation_metrics,
                            )

                    final_decision = (
                        early_stopping.observe(
                            checkpoint_id=(
                                final_checkpoint
                                .checkpoint_id
                            ),
                            evaluation_id=(
                                final_evaluation
                                .evaluation_id
                            ),
                            balanced_score=(
                                final_evaluation
                                .balanced_score
                            ),
                        )
                    )

                    if final_decision.improved:
                        best_checkpoint = (
                            final_checkpoint
                        )
                        best_evaluation = (
                            final_evaluation
                        )

                    break

        if final_checkpoint is None:
            raise TrainingServiceError(
                "Training schedule did not create a "
                "final checkpoint."
            )

        model_steps_after = int(
            model.num_timesteps
        )
        stopped_early = (
            early_stop_reason is not None
        )
        aggregate_training = (
            ExactPPOTrainingResult(
                model_steps_before=int(
                    model_steps_before
                ),
                model_steps_after=(
                    model_steps_after
                ),
                local_steps_requested=(
                    requested_steps
                ),
                local_steps_completed=(
                    completed_steps
                ),
                rollout_sizes=tuple(
                    rollout_sizes
                ),
                stopped_early=stopped_early,
            )
        )

        if (
            aggregate_training
            .model_steps_after
            != aggregate_training
            .model_steps_before
            + completed_steps
        ):
            raise TrainingStoppedEarlyError(
                "PPO completed an inconsistent total "
                "number of training steps."
            )

        if stopped_early:
            if not 0 < completed_steps < requested_steps:
                raise TrainingStoppedEarlyError(
                    "Early stopping did not terminate at "
                    "a valid partial run step."
                )
        elif completed_steps != requested_steps:
            raise TrainingStoppedEarlyError(
                "PPO completed an unexpected total "
                "number of training steps."
            )

        if (
            best_checkpoint is None
            or best_evaluation is None
        ):
            raise TrainingServiceError(
                "Training completed without a usable "
                "validation evaluation."
            )

        completed = complete_run(
            session_factory,
            run_id=run_id,
            stopped_early=stopped_early,
            early_stop_reason=early_stop_reason,
        )

        _progress_call(
            progress_reporter,
            "finish",
            completed_steps=completed_steps,
            stopped_early=stopped_early,
        )

        return TrainingServiceResult(
            run=completed,
            checkpoint=final_checkpoint,
            checkpoints=tuple(
                persisted_checkpoints
            ),
            evaluations=tuple(
                persisted_evaluations
            ),
            best_checkpoint=best_checkpoint,
            best_evaluation=best_evaluation,
            training=aggregate_training,
            early_stop_reason=(
                early_stop_reason
            ),
            source_checkpoint_id=(
                None
                if resume_source is None
                else resume_source.checkpoint_id
            ),
        )

    except BaseException as error:
        local_steps = _local_model_steps(
            model,
            model_steps_before=(
                model_steps_before
            ),
            maximum=requested_steps,
        )

        if (
            model is not None
            and final_checkpoint is None
            and latest_checkpoint_step
            != local_steps
        ):
            try:
                storage = ArtifactStorage(
                    project_root=root,
                    artifacts_directory=(
                        artifacts_directory
                    ),
                )
                persist_ppo_checkpoint(
                    session_factory,
                    storage,
                    model,
                    run_id=run_id,
                    run_step=local_steps,
                    save_reason=(
                        CheckpointSaveReason
                        .INTERRUPTED
                    ),
                )
            except BaseException as checkpoint_error:
                error.add_note(
                    "Additionally, persisting the "
                    "interrupted PPO checkpoint failed: "
                    f"{type(checkpoint_error).__name__}: "
                    f"{checkpoint_error}"
                )

        try:
            fail_run(
                session_factory,
                run_id=run_id,
                error=error,
                training_steps_completed=(
                    local_steps
                ),
            )
        except BaseException as failure_error:
            error.add_note(
                "Additionally, persisting the failed "
                "run state failed: "
                f"{type(failure_error).__name__}: "
                f"{failure_error}"
            )

        _progress_call(
            progress_reporter,
            "fail",
            error,
            completed_steps=local_steps,
        )

        raise
