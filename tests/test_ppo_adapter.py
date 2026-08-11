from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from torch import nn

from train_and_eval.ppo.adapter import (
    PPOCheckpointFileError,
    create_ppo_model,
    load_ppo_model_file,
    ppo_constructor_kwargs,
    save_ppo_model_file,
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
        self.action_space = gym.spaces.Discrete(
            2
        )
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(seed=seed)
        self.steps = 0

        return (
            np.zeros(
                4,
                dtype=np.float32,
            ),
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
            min(
                self.steps / 10.0,
                1.0,
            ),
            dtype=np.float32,
        )
        reward = float(
            int(action) == 1
        )
        terminated = self.steps >= 4

        return (
            observation,
            reward,
            terminated,
            False,
            {},
        )


def _ppo_config(
    *,
    activation: str = "relu",
    value_head_init_scale: float = 1.0,
) -> PPOSection:
    return PPOSection.model_validate(
        {
            "policy": "mlp",
            "device": "cpu",
            "hidden_sizes": [
                32,
                16,
            ],
            "activation": activation,
            "value_head_init_scale": value_head_init_scale,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 2,
            "learning_rate": 0.0003,
            "gamma": 0.90,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0002,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
        }
    )


def test_constructor_kwargs_are_explicit() -> None:
    config = _ppo_config(
        activation="relu"
    )

    kwargs = ppo_constructor_kwargs(
        config,
        seed=123,
        verbose=1,
    )

    assert kwargs["device"] == "cpu"
    assert kwargs["n_steps"] == 8
    assert kwargs["batch_size"] == 4
    assert kwargs["n_epochs"] == 2
    assert kwargs["learning_rate"] == pytest.approx(
        0.0003
    )
    assert kwargs["gamma"] == pytest.approx(
        0.90
    )
    assert kwargs["gae_lambda"] == pytest.approx(
        0.95
    )
    assert kwargs["clip_range"] == pytest.approx(
        0.20
    )
    assert kwargs["clip_range_vf"] is None
    assert kwargs["normalize_advantage"] is True
    assert kwargs["ent_coef"] == pytest.approx(
        0.0002
    )
    assert kwargs["vf_coef"] == pytest.approx(
        0.5
    )
    assert kwargs["max_grad_norm"] == pytest.approx(
        0.5
    )
    assert kwargs["target_kl"] is None
    assert kwargs["seed"] == 123
    assert kwargs["verbose"] == 1

    policy_kwargs = kwargs[
        "policy_kwargs"
    ]

    assert policy_kwargs[
        "net_arch"
    ] == [32, 16]
    assert policy_kwargs[
        "activation_fn"
    ] is nn.ReLU


def test_creates_ppo_from_explicit_config() -> None:
    environment = TinyEnvironment()

    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=7,
    )

    assert isinstance(model, PPO)
    assert model.device.type == "cpu"
    assert model.n_steps == 8
    assert model.batch_size == 4
    assert model.n_epochs == 2
    assert model.gamma == pytest.approx(
        0.90
    )
    assert model.gae_lambda == pytest.approx(
        0.95
    )


def test_scales_value_head_initialization() -> None:
    baseline = create_ppo_model(
        TinyEnvironment(),
        _ppo_config(
            value_head_init_scale=1.0,
        ),
        seed=17,
    )

    scaled = create_ppo_model(
        TinyEnvironment(),
        _ppo_config(
            value_head_init_scale=0.003,
        ),
        seed=17,
    )

    baseline_weight = (
        baseline.policy.value_net.weight
        .detach()
        .cpu()
        .numpy()
    )
    scaled_weight = (
        scaled.policy.value_net.weight
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(
        scaled_weight,
        baseline_weight * 0.003,
        rtol=1e-6,
        atol=1e-8,
    )


def test_saves_and_loads_native_ppo_checkpoint(
    tmp_path: Path,
) -> None:
    environment = TinyEnvironment()

    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=11,
    )
    model.learn(
        total_timesteps=8
    )

    checkpoint_path = (
        tmp_path / "model.zip"
    )

    saved_path = save_ppo_model_file(
        model,
        checkpoint_path,
    )

    assert saved_path == (
        checkpoint_path.resolve()
    )
    assert zipfile.is_zipfile(
        saved_path
    )

    loaded = load_ppo_model_file(
        saved_path,
        environment=environment,
        device="cpu",
    )

    assert loaded.num_timesteps == 8

    observation, _ = environment.reset(
        seed=11
    )

    action, _ = loaded.predict(
        observation,
        deterministic=True,
    )

    assert int(
        np.asarray(action).item()
    ) in {0, 1}


def test_does_not_overwrite_existing_checkpoint(
    tmp_path: Path,
) -> None:
    destination = (
        tmp_path / "model.zip"
    )
    destination.write_bytes(
        b"existing file"
    )

    model = create_ppo_model(
        TinyEnvironment(),
        _ppo_config(),
        seed=5,
    )

    with pytest.raises(
        PPOCheckpointFileError,
        match="already exists",
    ):
        save_ppo_model_file(
            model,
            destination,
        )

    assert destination.read_bytes() == (
        b"existing file"
    )


def test_rejects_invalid_checkpoint_zip(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "invalid.zip"
    )
    checkpoint_path.write_bytes(
        b"not a Stable-Baselines3 archive"
    )

    with pytest.raises(
        PPOCheckpointFileError,
        match="not a valid ZIP",
    ):
        load_ppo_model_file(
            checkpoint_path,
            environment=TinyEnvironment(),
            device="cpu",
        )


def test_resume_custom_objects_override_mutable_training_settings() -> None:
    from train_and_eval.ppo.adapter import (
        ppo_resume_custom_objects,
    )

    config = _ppo_config()
    values = ppo_resume_custom_objects(
        config,
        seed=321,
    )

    assert values == {
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 2,
        "learning_rate": pytest.approx(0.0003),
        "gamma": pytest.approx(0.90),
        "gae_lambda": pytest.approx(0.95),
        "clip_range": pytest.approx(0.20),
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": pytest.approx(0.0002),
        "vf_coef": pytest.approx(0.5),
        "max_grad_norm": pytest.approx(0.5),
        "target_kl": None,
        "seed": 321,
    }
