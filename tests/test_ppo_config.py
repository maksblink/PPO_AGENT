from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from train_and_eval.run_config import (
    PPOSection,
    RESUME_IMMUTABLE_FIELDS,
    SCHEMA_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PPO_FIELDS = {
    "policy",
    "device",
    "hidden_sizes",
    "activation",
    "n_steps",
    "batch_size",
    "n_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
}


def _valid_ppo_config() -> dict[str, object]:
    return {
        "policy": "mlp",
        "device": "cpu",
        "hidden_sizes": [64, 64],
        "activation": "tanh",
        "n_steps": 2048,
        "batch_size": 1024,
        "n_epochs": 10,
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


def test_accepts_complete_explicit_ppo_config() -> None:
    section = PPOSection.model_validate(
        _valid_ppo_config()
    )

    assert section.policy == "mlp"
    assert section.activation == "tanh"
    assert section.n_epochs == 10
    assert section.gae_lambda == pytest.approx(
        0.95
    )
    assert section.clip_range_vf is None
    assert section.target_kl is None


def test_rejects_unknown_policy() -> None:
    config = _valid_ppo_config()
    config["policy"] = "cnn"

    with pytest.raises(ValidationError):
        PPOSection.model_validate(config)


def test_rejects_unknown_activation() -> None:
    config = _valid_ppo_config()
    config["activation"] = "sigmoid"

    with pytest.raises(ValidationError):
        PPOSection.model_validate(config)


def test_rejects_nondivisible_batch_size() -> None:
    config = _valid_ppo_config()
    config["batch_size"] = 1000

    with pytest.raises(
        ValidationError,
        match="must divide n_steps exactly",
    ):
        PPOSection.model_validate(config)


def test_rejects_nonpositive_n_epochs() -> None:
    config = _valid_ppo_config()
    config["n_epochs"] = 0

    with pytest.raises(ValidationError):
        PPOSection.model_validate(config)


def test_rejects_invalid_gae_lambda() -> None:
    config = _valid_ppo_config()
    config["gae_lambda"] = 0.0

    with pytest.raises(ValidationError):
        PPOSection.model_validate(config)


def test_rejects_nonpositive_clip_range_vf() -> None:
    config = _valid_ppo_config()
    config["clip_range_vf"] = 0.0

    with pytest.raises(
        ValidationError,
        match="greater than zero or null",
    ):
        PPOSection.model_validate(config)


def test_rejects_nonpositive_target_kl() -> None:
    config = _valid_ppo_config()
    config["target_kl"] = 0.0

    with pytest.raises(
        ValidationError,
        match="greater than zero or null",
    ):
        PPOSection.model_validate(config)


def test_experiment_yamls_define_all_ppo_fields() -> None:
    config_paths = [
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "example_5m_data_epochs.yml",
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "example_5m_timesteps.yml",
    ]

    for path in config_paths:
        document = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

        assert document[
            "config_schema_version"
        ] == 1

        ppo_config = document["ppo"]

        assert set(ppo_config) == EXPECTED_PPO_FIELDS

        PPOSection.model_validate(
            ppo_config
        )


def test_schema_and_resume_architecture_are_explicit() -> None:
    assert SCHEMA_VERSION == 1

    assert {
        "ppo.policy",
        "ppo.hidden_sizes",
        "ppo.activation",
    } <= set(RESUME_IMMUTABLE_FIELDS)
