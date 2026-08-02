from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from gymnasium import Env
from sqlalchemy.orm import Session
from stable_baselines3 import PPO

from train_and_eval.artifact_storage.storage import (
    DEFAULT_ARTIFACTS_DIRECTORY,
    ArtifactStorage,
)
from train_and_eval.checkpoints.artifacts import (
    resolve_checkpoint_artifact,
)
from train_and_eval.checkpoints.persistence import (
    PersistedCheckpoint,
    persist_checkpoint_file,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
)
from train_and_eval.ppo.adapter import (
    PolicyDevice,
    load_ppo_model_file,
    ppo_resume_custom_objects,
    save_ppo_model_file,
)
from train_and_eval.run_config import PPOSection


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SessionFactory = Callable[[], Session]


class PPOCheckpointRecord(Protocol):
    """Database fields required to load one PPO checkpoint."""

    id: int
    relative_path: str
    sha256: str
    size_bytes: int
    model_step: int


class PPOCheckpointIntegrationError(RuntimeError):
    """Base error for persisted PPO checkpoint operations."""


class PPOCheckpointStepMismatchError(
    PPOCheckpointIntegrationError
):
    """Raised when ZIP metadata and database model_step disagree."""


def _nonnegative_integer(
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise PPOCheckpointIntegrationError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PPOCheckpointIntegrationError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise PPOCheckpointIntegrationError(
            f"{name} must be an integer."
        )

    if result < 0:
        raise PPOCheckpointIntegrationError(
            f"{name} must be nonnegative."
        )

    return result


def persist_ppo_checkpoint(
    session_factory: SessionFactory,
    storage: ArtifactStorage,
    model: PPO,
    *,
    run_id: int,
    run_step: int,
    save_reason: CheckpointSaveReason | str,
) -> PersistedCheckpoint:
    """
    Save a PPO model and persist its immutable checkpoint record.

    model_step comes directly from Stable-Baselines3 num_timesteps.
    run_step remains the number of steps completed in the current run.
    """
    model_step = _nonnegative_integer(
        model.num_timesteps,
        name="model.num_timesteps",
    )

    with TemporaryDirectory(
        prefix="ppo-agent-checkpoint-"
    ) as temporary_directory:
        temporary_checkpoint = (
            Path(temporary_directory)
            / "model.zip"
        )

        save_ppo_model_file(
            model,
            temporary_checkpoint,
        )

        return persist_checkpoint_file(
            session_factory,
            storage,
            source_path=temporary_checkpoint,
            run_id=run_id,
            run_step=run_step,
            model_step=model_step,
            save_reason=save_reason,
        )


def load_persisted_ppo_checkpoint(
    checkpoint: PPOCheckpointRecord,
    *,
    environment: Env | None,
    device: PolicyDevice,
    training_config: PPOSection | None = None,
    seed: int | None = None,
    project_root: str | Path = PROJECT_ROOT,
    artifacts_directory: str | Path = (
        DEFAULT_ARTIFACTS_DIRECTORY
    ),
) -> PPO:
    """
    Verify an immutable artifact and load its PPO model.

    The Stable-Baselines3 num_timesteps stored inside the ZIP must
    match checkpoints.model_step in PostgreSQL.
    """
    expected_model_step = _nonnegative_integer(
        checkpoint.model_step,
        name="checkpoint.model_step",
    )

    artifact = resolve_checkpoint_artifact(
        checkpoint,
        project_root=project_root,
        artifacts_directory=artifacts_directory,
    )

    if (training_config is None) != (seed is None):
        raise PPOCheckpointIntegrationError(
            "training_config and seed must be supplied together."
        )

    custom_objects = (
        None
        if training_config is None
        else ppo_resume_custom_objects(
            training_config,
            seed=int(seed),
        )
    )

    model = load_ppo_model_file(
        artifact.absolute_path,
        environment=environment,
        device=device,
        custom_objects=custom_objects,
    )

    loaded_model_step = _nonnegative_integer(
        model.num_timesteps,
        name="loaded_model.num_timesteps",
    )

    if loaded_model_step != expected_model_step:
        raise PPOCheckpointStepMismatchError(
            "PPO checkpoint model_step mismatch: "
            f"database expects {expected_model_step}, "
            f"ZIP contains {loaded_model_step}."
        )

    return model
