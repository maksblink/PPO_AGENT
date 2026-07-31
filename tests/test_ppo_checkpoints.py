from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from train_and_eval.artifact_storage.storage import (
    ArtifactStorage,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
)
from train_and_eval.ppo.adapter import (
    create_ppo_model,
)
from train_and_eval.ppo.checkpoints import (
    PPOCheckpointStepMismatchError,
    load_persisted_ppo_checkpoint,
    persist_ppo_checkpoint,
)
from train_and_eval.run_config import (
    PPOSection,
)


class TinyEnvironment(gym.Env):
    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0

        return (
            np.zeros(4, dtype=np.float32),
            {},
        )

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self.steps += 1

        observation = np.full(
            4,
            min(self.steps / 10.0, 1.0),
            dtype=np.float32,
        )

        return (
            observation,
            float(int(action) == 1),
            self.steps >= 4,
            False,
            {},
        )


class FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commit_calls = 0
        self.rollback_calls = 0

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        return False

    def get(
        self,
        model: type,
        identity: int,
    ) -> object:
        return SimpleNamespace(id=identity)

    def add(self, value: object) -> None:
        self.added.append(value)

    def flush(self) -> None:
        checkpoint = self.added[-1]
        checkpoint.id = 91

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


class FakeSessionFactory:
    def __init__(
        self,
        session: FakeSession,
    ) -> None:
        self.session = session

    def __call__(self) -> FakeSession:
        return self.session


def _ppo_config() -> PPOSection:
    return PPOSection.model_validate(
        {
            "policy": "mlp",
            "device": "cpu",
            "hidden_sizes": [16, 16],
            "activation": "tanh",
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "learning_rate": 0.0003,
            "gamma": 0.90,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
        }
    )


def _trained_model():
    environment = TinyEnvironment()

    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=123,
    )

    model.learn(total_timesteps=8)

    return model


def test_persists_native_ppo_checkpoint(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession()

    persisted = persist_ppo_checkpoint(
        FakeSessionFactory(session),
        ArtifactStorage(
            project_root=project_root
        ),
        _trained_model(),
        run_id=4,
        run_step=8,
        save_reason=(
            CheckpointSaveReason.FINAL
        ),
    )

    assert persisted.checkpoint_id == 91
    assert persisted.run_id == 4
    assert persisted.run_step == 8
    assert persisted.model_step == 8

    assert persisted.relative_path == (
        "artifacts/runs/00000004/"
        "checkpoints/"
        "step_000000000008_final.zip"
    )

    assert persisted.absolute_path.is_file()
    assert zipfile.is_zipfile(
        persisted.absolute_path
    )

    checkpoint = session.added[0]

    assert checkpoint.model_step == 8
    assert checkpoint.relative_path == (
        persisted.relative_path
    )
    assert checkpoint.sha256 == (
        persisted.sha256
    )
    assert session.commit_calls == 1
    assert session.rollback_calls == 0


def test_loads_verified_persisted_ppo_checkpoint(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession()

    persisted = persist_ppo_checkpoint(
        FakeSessionFactory(session),
        ArtifactStorage(
            project_root=project_root
        ),
        _trained_model(),
        run_id=7,
        run_step=8,
        save_reason="periodic",
    )

    checkpoint = SimpleNamespace(
        id=persisted.checkpoint_id,
        relative_path=persisted.relative_path,
        sha256=persisted.sha256,
        size_bytes=persisted.size_bytes,
        model_step=persisted.model_step,
    )

    environment = TinyEnvironment()

    loaded = load_persisted_ppo_checkpoint(
        checkpoint,
        environment=environment,
        device="cpu",
        project_root=project_root,
    )

    assert loaded.num_timesteps == 8

    observation, _ = environment.reset(
        seed=123
    )

    action, _ = loaded.predict(
        observation,
        deterministic=True,
    )

    assert int(
        np.asarray(action).item()
    ) in {0, 1}


def test_rejects_database_model_step_mismatch(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    session = FakeSession()

    persisted = persist_ppo_checkpoint(
        FakeSessionFactory(session),
        ArtifactStorage(
            project_root=project_root
        ),
        _trained_model(),
        run_id=9,
        run_step=8,
        save_reason="final",
    )

    checkpoint = SimpleNamespace(
        id=persisted.checkpoint_id,
        relative_path=persisted.relative_path,
        sha256=persisted.sha256,
        size_bytes=persisted.size_bytes,
        model_step=999,
    )

    with pytest.raises(
        PPOCheckpointStepMismatchError,
        match=(
            "database expects 999, "
            "ZIP contains 8"
        ),
    ):
        load_persisted_ppo_checkpoint(
            checkpoint,
            environment=TinyEnvironment(),
            device="cpu",
            project_root=project_root,
        )
