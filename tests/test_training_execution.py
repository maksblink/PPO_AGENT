from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from train_and_eval.ppo.adapter import (
    create_ppo_model,
)
from train_and_eval.run_config import (
    PPOSection,
)
from train_and_eval.training.execution import (
    PPOTrainingIdentityError,
    learn_ppo_exact_timesteps,
)


class CountingEnvironment(gym.Env):
    metadata: dict[str, Any] = {}

    def __init__(
        self,
        *,
        episode_steps: int,
    ) -> None:
        super().__init__()

        self.observation_space = (
            gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        )
        self.action_space = (
            gym.spaces.Discrete(2)
        )

        self.episode_steps = (
            episode_steps
        )
        self.step_in_episode = 0
        self.reset_count = 0
        self.executed_episode_steps: list[
            int
        ] = []

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
        self.step_in_episode = 0
        self.reset_count += 1

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
        self.step_in_episode += 1
        self.executed_episode_steps.append(
            self.step_in_episode
        )

        terminated = (
            self.step_in_episode
            >= self.episode_steps
        )
        observation = np.full(
            4,
            self.step_in_episode
            / self.episode_steps,
            dtype=np.float32,
        )

        return (
            observation,
            float(int(action) == 1),
            terminated,
            False,
            {},
        )


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


def test_trains_exact_non_multiple_of_rollout_size() -> None:
    environment = CountingEnvironment(
        episode_steps=5
    )
    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=123,
    )

    result = learn_ppo_exact_timesteps(
        model,
        total_timesteps=10,
        log_interval=None,
    )

    assert result.model_steps_before == 0
    assert result.model_steps_after == 10
    assert result.local_steps_requested == 10
    assert result.local_steps_completed == 10
    assert result.rollout_sizes == (8, 2)
    assert result.stopped_early is False
    assert model.num_timesteps == 10


def test_each_data_epoch_restarts_from_first_scored_step() -> None:
    environment = CountingEnvironment(
        episode_steps=5
    )
    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=7,
    )

    result = learn_ppo_exact_timesteps(
        model,
        total_timesteps=15,
        log_interval=None,
    )

    assert result.rollout_sizes == (8, 7)
    assert (
        environment.executed_episode_steps
        == [1, 2, 3, 4, 5] * 3
    )

    # Initial reset plus one automatic reset after each completed epoch.
    assert environment.reset_count == 4


def test_continued_training_reports_local_and_model_steps() -> None:
    environment = CountingEnvironment(
        episode_steps=5
    )
    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=11,
    )

    first = learn_ppo_exact_timesteps(
        model,
        total_timesteps=8,
        log_interval=None,
    )
    second = learn_ppo_exact_timesteps(
        model,
        total_timesteps=3,
        log_interval=None,
    )

    assert first.model_steps_after == 8
    assert second.model_steps_before == 8
    assert second.model_steps_after == 11
    assert second.local_steps_completed == 3
    assert second.rollout_sizes == (3,)
    assert model.num_timesteps == 11


def test_rejects_nonpositive_requested_steps() -> None:
    model = create_ppo_model(
        CountingEnvironment(
            episode_steps=5
        ),
        _ppo_config(),
        seed=5,
    )

    with pytest.raises(
        PPOTrainingIdentityError,
        match="positive integer",
    ):
        learn_ppo_exact_timesteps(
            model,
            total_timesteps=0,
        )


def test_trains_final_single_step_rollout() -> None:
    environment = CountingEnvironment(
        episode_steps=20
    )
    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=17,
    )

    result = learn_ppo_exact_timesteps(
        model,
        total_timesteps=9,
        log_interval=None,
    )

    assert result.rollout_sizes == (8, 1)
    assert result.local_steps_completed == 9
    assert model.num_timesteps == 9


def test_reports_latest_training_metrics_after_each_rollout() -> None:
    environment = CountingEnvironment(
        episode_steps=20
    )
    model = create_ppo_model(
        environment,
        _ppo_config(),
        seed=19,
    )
    updates = []

    result = learn_ppo_exact_timesteps(
        model,
        total_timesteps=10,
        log_interval=None,
        update_callback=updates.append,
    )

    assert result.rollout_sizes == (8, 2)
    assert [
        update.local_steps_completed
        for update in updates
    ] == [8, 10]
    assert [
        update.rollout_iteration
        for update in updates
    ] == [1, 2]
    assert [
        update.rollout_size
        for update in updates
    ] == [8, 2]
    assert all(
        "approx_kl" in update.metrics
        for update in updates
    )
    assert all(
        "entropy_loss" in update.metrics
        for update in updates
    )
    assert all(
        "explained_variance" in update.metrics
        for update in updates
    )
    assert updates[-1].metrics["n_updates"] == 2
