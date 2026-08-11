from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from train_and_eval.database.models import Run, TrainingMetric


class TrainingMetricPersistenceError(RuntimeError):
    """Raised when one training metric snapshot cannot be persisted."""


@dataclass(frozen=True, slots=True)
class PersistedTrainingMetric:
    metric_id: int
    run_id: int
    run_step: int
    model_step: int
    rollout_number: int


def _optional_float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(metrics: Mapping[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result != value:
        return None
    return result


def persist_training_metric(
    session_factory,
    *,
    run_id: int,
    run_step: int,
    model_step: int,
    rollout_number: int,
    metrics: Mapping[str, Any],
) -> PersistedTrainingMetric:
    """Persist one lightweight PPO diagnostic snapshot.

    The function is idempotent for ``(run_id, run_step)`` so forced final
    snapshots can safely coincide with a regular cadence boundary.
    """
    if run_id < 1:
        raise TrainingMetricPersistenceError("run_id must be positive")
    if run_step < 1:
        raise TrainingMetricPersistenceError("run_step must be positive")
    if model_step < run_step:
        raise TrainingMetricPersistenceError(
            "model_step cannot be smaller than run_step"
        )
    if rollout_number < 1:
        raise TrainingMetricPersistenceError(
            "rollout_number must be positive"
        )

    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise TrainingMetricPersistenceError(
                f"Run {run_id} does not exist."
            )

        existing = session.scalar(
            select(TrainingMetric).where(
                TrainingMetric.run_id == run_id,
                TrainingMetric.run_step == run_step,
            )
        )
        if existing is not None:
            return PersistedTrainingMetric(
                metric_id=int(existing.id),
                run_id=int(existing.run_id),
                run_step=int(existing.run_step),
                model_step=int(existing.model_step),
                rollout_number=int(existing.rollout_number),
            )

        record = TrainingMetric(
            run_id=run_id,
            run_step=run_step,
            model_step=model_step,
            rollout_number=rollout_number,
            ep_reward=_optional_float(metrics, "ep_rew_mean"),
            ep_len=_optional_float(metrics, "ep_len_mean"),
            rollout_reward_mean=_optional_float(metrics, "rollout_reward_mean"),
            rollout_reward_sum=_optional_float(metrics, "rollout_reward_sum"),
            approx_kl=_optional_float(metrics, "approx_kl"),
            clip_fraction=_optional_float(metrics, "clip_fraction"),
            clip_range=_optional_float(metrics, "clip_range"),
            entropy_loss=_optional_float(metrics, "entropy_loss"),
            explained_variance=_optional_float(metrics, "explained_variance"),
            learning_rate=_optional_float(metrics, "learning_rate"),
            loss=_optional_float(metrics, "loss"),
            n_updates=_optional_int(metrics, "n_updates"),
            policy_gradient_loss=_optional_float(
                metrics, "policy_gradient_loss"
            ),
            value_loss=_optional_float(metrics, "value_loss"),
        value_target_mean=_optional_float(
            metrics, "value_target_mean"
        ),
        value_target_std=_optional_float(
            metrics, "value_target_std"
        ),
        value_prediction_mean=_optional_float(
            metrics, "value_prediction_mean"
        ),
        value_prediction_std=_optional_float(
            metrics, "value_prediction_std"
        ),
        value_error_mean=_optional_float(
            metrics, "value_error_mean"
        ),
        value_error_std=_optional_float(
            metrics, "value_error_std"
        ),
        value_target_prediction_corr=_optional_float(
            metrics, "value_target_prediction_corr"
        ),
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        return PersistedTrainingMetric(
            metric_id=int(record.id),
            run_id=int(record.run_id),
            run_step=int(record.run_step),
            model_step=int(record.model_step),
            rollout_number=int(record.rollout_number),
        )
