from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Literal

from gymnasium import Env
from stable_baselines3 import PPO
from torch import nn

from train_and_eval.run_config import PPOSection


PolicyDevice = Literal[
    "auto",
    "cpu",
    "cuda",
]


_POLICY_NAMES = {
    "mlp": "MlpPolicy",
}

_ACTIVATION_CLASSES = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


class PPOAdapterError(RuntimeError):
    """Base error for PPO adapter failures."""


class PPOCheckpointFileError(
    PPOAdapterError
):
    """Raised when a native PPO checkpoint file is invalid."""


def _integer_value(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise PPOAdapterError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PPOAdapterError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise PPOAdapterError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise PPOAdapterError(
            f"{name} must be at least {minimum}."
        )

    return result


def ppo_constructor_kwargs(
    config: PPOSection,
    *,
    seed: int,
    verbose: int = 0,
) -> dict[str, Any]:
    """
    Convert the explicit project configuration to SB3 PPO arguments.

    Every algorithm parameter that affects training is supplied
    explicitly rather than inherited silently from SB3 defaults.
    """
    resolved_seed = _integer_value(
        seed,
        name="seed",
        minimum=0,
    )
    resolved_verbose = _integer_value(
        verbose,
        name="verbose",
        minimum=0,
    )

    policy_kwargs: dict[str, Any] = {
        "net_arch": list(
            config.hidden_sizes
        ),
        "activation_fn": (
            _ACTIVATION_CLASSES[
                config.activation
            ]
        ),
    }

    return {
        "device": config.device,
        "policy_kwargs": policy_kwargs,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "learning_rate": (
            config.learning_rate
        ),
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "clip_range_vf": (
            config.clip_range_vf
        ),
        "normalize_advantage": (
            config.normalize_advantage
        ),
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": (
            config.max_grad_norm
        ),
        "target_kl": config.target_kl,
        "seed": resolved_seed,
        "verbose": resolved_verbose,
    }


def ppo_resume_custom_objects(
    config: PPOSection,
    *,
    seed: int,
) -> dict[str, Any]:
    """Override mutable PPO training settings when loading for resume."""
    resolved_seed = _integer_value(
        seed,
        name="seed",
        minimum=0,
    )

    return {
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "learning_rate": (
            config.learning_rate
        ),
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "clip_range_vf": (
            config.clip_range_vf
        ),
        "normalize_advantage": (
            config.normalize_advantage
        ),
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": (
            config.max_grad_norm
        ),
        "target_kl": config.target_kl,
        "seed": resolved_seed,
    }


def create_ppo_model(
    environment: Env,
    config: PPOSection,
    *,
    seed: int,
    verbose: int = 0,
) -> PPO:
    """Create a new PPO model from the complete run configuration."""
    policy_name = _POLICY_NAMES[
        config.policy
    ]

    return PPO(
        policy_name,
        environment,
        **ppo_constructor_kwargs(
            config,
            seed=seed,
            verbose=verbose,
        ),
    )


def save_ppo_model_file(
    model: PPO,
    destination: str | Path,
) -> Path:
    """
    Save one native Stable-Baselines3 PPO ZIP file.

    The destination must end in .zip and must not already exist.
    """
    destination_path = (
        Path(destination)
        .expanduser()
        .resolve()
    )

    if destination_path.suffix.lower() != ".zip":
        raise PPOCheckpointFileError(
            "PPO checkpoint destination must "
            "end with .zip."
        )

    if destination_path.exists():
        raise PPOCheckpointFileError(
            "PPO checkpoint destination "
            f"already exists: {destination_path}"
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        model.save(
            str(destination_path)
        )
    except Exception:
        destination_path.unlink(
            missing_ok=True
        )
        raise

    if not destination_path.is_file():
        raise PPOCheckpointFileError(
            "Stable-Baselines3 did not create "
            f"the expected file: {destination_path}"
        )

    if not zipfile.is_zipfile(
        destination_path
    ):
        destination_path.unlink(
            missing_ok=True
        )

        raise PPOCheckpointFileError(
            "Stable-Baselines3 produced an "
            "invalid ZIP checkpoint."
        )

    return destination_path


def load_ppo_model_file(
    checkpoint_path: str | Path,
    *,
    environment: Env | None,
    device: PolicyDevice,
    custom_objects: dict[str, Any] | None = None,
) -> PPO:
    """Load a native PPO ZIP file after basic file validation."""
    resolved_path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.is_file():
        raise PPOCheckpointFileError(
            "PPO checkpoint file does not exist: "
            f"{resolved_path}"
        )

    if not zipfile.is_zipfile(
        resolved_path
    ):
        raise PPOCheckpointFileError(
            "PPO checkpoint is not a valid ZIP file: "
            f"{resolved_path}"
        )

    try:
        model = PPO.load(
            str(resolved_path),
            env=environment,
            device=device,
            custom_objects=custom_objects,
            print_system_info=False,
        )
    except Exception as error:
        raise PPOCheckpointFileError(
            "Stable-Baselines3 could not load "
            f"the PPO checkpoint: {resolved_path}"
        ) from error

    return model
