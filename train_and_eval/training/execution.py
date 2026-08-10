from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import MaybeCallback


SUPPORTED_SB3_VERSION = "2.9.0"


class PPOTrainingExecutionError(RuntimeError):
    """Base error for exact PPO training execution."""


class PPOTrainingCompatibilityError(
    PPOTrainingExecutionError
):
    """Raised when pinned SB3 internals no longer match expectations."""


class PPOTrainingIdentityError(
    PPOTrainingExecutionError
):
    """Raised when exact training arguments are invalid."""




@dataclass(frozen=True, slots=True)
class PPOTrainingUpdate:
    model_steps: int
    local_steps_completed: int
    rollout_iteration: int
    rollout_size: int
    metrics: Mapping[str, int | float]


@dataclass(frozen=True, slots=True)
class ExactPPOTrainingResult:
    model_steps_before: int
    model_steps_after: int
    local_steps_requested: int
    local_steps_completed: int
    rollout_sizes: tuple[int, ...]
    stopped_early: bool


def _positive_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise PPOTrainingIdentityError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PPOTrainingIdentityError(
            f"{name} must be an integer."
        ) from error

    if result != value or result < 1:
        raise PPOTrainingIdentityError(
            f"{name} must be a positive integer."
        )

    return result


def _new_rollout_buffer(
    model: PPO,
    *,
    buffer_size: int,
) -> RolloutBuffer:
    rollout_buffer_class = (
        model.rollout_buffer_class
    )

    if rollout_buffer_class is None:
        raise PPOTrainingCompatibilityError(
            "PPO rollout_buffer_class is not initialized."
        )

    return rollout_buffer_class(
        buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )



_TRAIN_LOG_KEYS = {
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/clip_range": "clip_range",
    "train/entropy_loss": "entropy_loss",
    "train/explained_variance": "explained_variance",
    "train/learning_rate": "learning_rate",
    "train/loss": "loss",
    "train/n_updates": "n_updates",
    "train/policy_gradient_loss": "policy_gradient_loss",
    "train/value_loss": "value_loss",
}


def _numeric_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_training_metrics(model: PPO) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    values = getattr(model.logger, "name_to_value", {})

    for source_name, output_name in _TRAIN_LOG_KEYS.items():
        value = _numeric_value(values.get(source_name))
        if value is not None:
            metrics[output_name] = value

    rollout_buffer = getattr(model, "rollout_buffer", None)
    rollout_rewards = getattr(rollout_buffer, "rewards", None)
    if rollout_rewards is not None:
        try:
            metrics["rollout_reward_mean"] = float(rollout_rewards.mean())
            metrics["rollout_reward_sum"] = float(rollout_rewards.sum())
        except (TypeError, ValueError, AttributeError):
            pass

    episode_buffer = getattr(model, "ep_info_buffer", None)
    if episode_buffer:
        rewards = [
            _numeric_value(item.get("r"))
            for item in episode_buffer
            if isinstance(item, dict) and "r" in item
        ]
        lengths = [
            _numeric_value(item.get("l"))
            for item in episode_buffer
            if isinstance(item, dict) and "l" in item
        ]
        rewards = [value for value in rewards if value is not None]
        lengths = [value for value in lengths if value is not None]

        if rewards:
            metrics["ep_rew_mean"] = float(sum(rewards) / len(rewards))
        if lengths:
            metrics["ep_len_mean"] = float(sum(lengths) / len(lengths))

    return metrics


def learn_ppo_exact_timesteps(
    model: PPO,
    *,
    total_timesteps: int,
    callback: MaybeCallback = None,
    log_interval: int | None = 1,
    progress_bar: bool = False,
    update_callback: Callable[[PPOTrainingUpdate], None] | None = None,
) -> ExactPPOTrainingResult:
    """
    Train PPO for exactly the requested number of environment steps.

    Stable-Baselines3 normally gathers fixed ``n_steps`` rollouts, so a
    normal ``learn()`` call may pass the requested total. This function
    preserves full rollouts and uses one smaller final RolloutBuffer when
    the requested total is not divisible by ``model.n_steps``.

    Episode boundaries remain controlled by the environment. Therefore,
    when one TradingEnvironment episode is one complete data pass, every
    automatic reset starts the next data epoch at its configured warm-up
    index.
    """
    requested_steps = _positive_integer(
        total_timesteps,
        name="total_timesteps",
    )

    if (
        stable_baselines3.__version__
        != SUPPORTED_SB3_VERSION
    ):
        raise PPOTrainingCompatibilityError(
            "Exact PPO training depends on "
            "Stable-Baselines3 internals from version "
            f"{SUPPORTED_SB3_VERSION}; found "
            f"{stable_baselines3.__version__}."
        )

    environment = model.get_env()

    if environment is None:
        raise PPOTrainingIdentityError(
            "PPO model must have a training environment."
        )

    if int(environment.num_envs) != 1:
        raise PPOTrainingIdentityError(
            "Exact training currently requires exactly "
            "one environment."
        )

    configured_rollout_steps = _positive_integer(
        model.n_steps,
        name="model.n_steps",
    )
    original_buffer = model.rollout_buffer
    model_steps_before = int(
        model.num_timesteps
    )

    absolute_target, resolved_callback = (
        model._setup_learn(
            requested_steps,
            callback,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            progress_bar=progress_bar,
        )
    )

    expected_target = (
        model_steps_before
        + requested_steps
    )

    if int(absolute_target) != expected_target:
        raise PPOTrainingCompatibilityError(
            "Stable-Baselines3 returned an unexpected "
            "absolute training target."
        )

    rollout_sizes: list[int] = []
    stopped_early = False
    iteration = 0
    training_started = False

    try:
        resolved_callback.on_training_start(
            locals(),
            globals(),
        )
        training_started = True

        while int(model.num_timesteps) < expected_target:
            remaining_steps = (
                expected_target
                - int(model.num_timesteps)
            )
            rollout_steps = min(
                configured_rollout_steps,
                remaining_steps,
            )

            if (
                rollout_steps
                == configured_rollout_steps
            ):
                active_buffer = original_buffer
            else:
                active_buffer = _new_rollout_buffer(
                    model,
                    buffer_size=rollout_steps,
                )

            model.rollout_buffer = (
                active_buffer
            )

            continue_training = (
                model.collect_rollouts(
                    environment,
                    resolved_callback,
                    active_buffer,
                    n_rollout_steps=(
                        rollout_steps
                    ),
                )
            )

            if not continue_training:
                stopped_early = True
                break

            rollout_sizes.append(
                rollout_steps
            )
            iteration += 1

            model._update_current_progress_remaining(
                int(model.num_timesteps),
                expected_target,
            )

            if (
                log_interval is not None
                and log_interval > 0
                and iteration % log_interval == 0
            ):
                model.dump_logs(iteration)

            model.train()

            if update_callback is not None:
                update_callback(
                    PPOTrainingUpdate(
                        model_steps=int(model.num_timesteps),
                        local_steps_completed=(
                            int(model.num_timesteps) - model_steps_before
                        ),
                        rollout_iteration=iteration,
                        rollout_size=rollout_steps,
                        metrics=_latest_training_metrics(model),
                    )
                )

    except BaseException as error:
        if training_started:
            try:
                resolved_callback.on_training_end()
            except BaseException as callback_error:
                error.add_note(
                    "Additionally, the PPO training-end "
                    "callback failed: "
                    f"{type(callback_error).__name__}: "
                    f"{callback_error}"
                )

        raise

    else:
        resolved_callback.on_training_end()

    finally:
        model.rollout_buffer = original_buffer

    model_steps_after = int(
        model.num_timesteps
    )
    local_steps_completed = (
        model_steps_after
        - model_steps_before
    )

    if (
        not stopped_early
        and local_steps_completed
        != requested_steps
    ):
        raise PPOTrainingCompatibilityError(
            "Exact PPO training completed with an "
            "unexpected number of environment steps."
        )

    return ExactPPOTrainingResult(
        model_steps_before=(
            model_steps_before
        ),
        model_steps_after=(
            model_steps_after
        ),
        local_steps_requested=(
            requested_steps
        ),
        local_steps_completed=(
            local_steps_completed
        ),
        rollout_sizes=tuple(
            rollout_sizes
        ),
        stopped_early=stopped_early,
    )
