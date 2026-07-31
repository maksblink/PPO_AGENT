from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from train_and_eval.database.models import (
    EvaluationPolicyMode,
)
from train_and_eval.ppo.adapter import (
    create_ppo_model,
)
from train_and_eval.ppo.policy import (
    PPOPolicySelectionError,
    PPOPolicySelector,
)
from train_and_eval.run_config import (
    PPOSection,
)


class TinyEnvironment(gym.Env):
    metadata: dict[str, Any] = {}

    def __init__(
        self,
        *,
        action_count: int = 2,
    ) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(
            action_count
        )

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
        return (
            np.zeros(
                4,
                dtype=np.float32,
            ),
            0.0,
            True,
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


def _model(
    *,
    action_count: int = 2,
):
    model = create_ppo_model(
        TinyEnvironment(
            action_count=action_count
        ),
        _ppo_config(),
        seed=123,
    )

    # Produce stable logits independent of the observation.
    with torch.no_grad():
        model.policy.action_net.weight.zero_()

        if action_count == 2:
            model.policy.action_net.bias.copy_(
                torch.tensor(
                    [0.0, 2.0]
                )
            )
        else:
            model.policy.action_net.bias.copy_(
                torch.tensor(
                    [0.0, 1.0, 2.0]
                )
            )

    return model


def _observation() -> np.ndarray:
    return np.zeros(
        4,
        dtype=np.float32,
    )


def test_deterministic_argmax_selects_highest_probability() -> None:
    selector = PPOPolicySelector(
        model=_model(),
        policy_mode=(
            EvaluationPolicyMode
            .DETERMINISTIC_ARGMAX
        ),
        seed=7,
    )

    decision = selector.select(
        _observation()
    )

    assert decision.action == 1
    assert decision.threshold_met is None
    assert len(decision.probabilities) == 2
    assert sum(
        decision.probabilities
    ) == pytest.approx(1.0)

    assert (
        decision.probabilities[1]
        > decision.probabilities[0]
    )
    assert (
        decision.selected_action_probability
        == pytest.approx(
            decision.probabilities[1]
        )
    )


def test_probability_threshold_selects_directional_action() -> None:
    selector = PPOPolicySelector(
        model=_model(),
        policy_mode="probability_threshold",
        seed=7,
        threshold_action=1,
        probability_threshold=0.80,
    )

    decision = selector.select(
        _observation()
    )

    assert decision.probabilities[
        1
    ] == pytest.approx(
        0.880797,
        rel=1e-5,
    )
    assert decision.threshold_met is True
    assert decision.action == 1


def test_probability_threshold_falls_back_to_flat() -> None:
    selector = PPOPolicySelector(
        model=_model(),
        policy_mode="probability_threshold",
        seed=7,
        threshold_action=1,
        probability_threshold=0.90,
    )

    decision = selector.select(
        _observation()
    )

    assert decision.threshold_met is False
    assert decision.action == 0
    assert (
        decision.selected_action_probability
        == pytest.approx(
            decision.probabilities[0]
        )
    )


def test_stochastic_sampling_is_reproducible() -> None:
    model = _model()

    first_selector = PPOPolicySelector(
        model=model,
        policy_mode="stochastic_sample",
        seed=12345,
    )

    first_actions = [
        first_selector.select(
            _observation()
        ).action
        for _ in range(20)
    ]

    second_selector = PPOPolicySelector(
        model=model,
        policy_mode="stochastic_sample",
        seed=12345,
    )

    second_actions = [
        second_selector.select(
            _observation()
        ).action
        for _ in range(20)
    ]

    assert first_actions == second_actions
    assert set(first_actions) <= {0, 1}


def test_non_threshold_mode_rejects_threshold_fields() -> None:
    with pytest.raises(
        PPOPolicySelectionError,
        match="threshold_action must be null",
    ):
        PPOPolicySelector(
            model=_model(),
            policy_mode="deterministic_argmax",
            seed=1,
            threshold_action=1,
        )


def test_three_action_threshold_selects_more_probable_short() -> None:
    selector = PPOPolicySelector(
        model=_model(
            action_count=3
        ),
        policy_mode="probability_threshold",
        seed=1,
        threshold_action=None,
        probability_threshold=0.60,
    )

    decision = selector.select(
        _observation()
    )

    assert (
        decision.probabilities[2]
        > decision.probabilities[1]
    )
    assert decision.threshold_met is True
    assert decision.action == 2


def test_three_action_threshold_falls_back_to_flat() -> None:
    selector = PPOPolicySelector(
        model=_model(
            action_count=3
        ),
        policy_mode="probability_threshold",
        seed=1,
        threshold_action=None,
        probability_threshold=0.70,
    )

    decision = selector.select(
        _observation()
    )

    assert (
        decision.probabilities[2]
        > decision.probabilities[1]
    )
    assert decision.threshold_met is False
    assert decision.action == 0


def test_three_action_threshold_tie_falls_back_to_flat() -> None:
    model = _model(
        action_count=3
    )

    with torch.no_grad():
        model.policy.action_net.bias.copy_(
            torch.tensor(
                [0.0, 2.0, 2.0]
            )
        )

    selector = PPOPolicySelector(
        model=model,
        policy_mode="probability_threshold",
        seed=1,
        threshold_action=None,
        probability_threshold=0.40,
    )

    decision = selector.select(
        _observation()
    )

    assert decision.probabilities[1] == pytest.approx(
        decision.probabilities[2]
    )
    assert decision.threshold_met is False
    assert decision.action == 0


def test_three_action_threshold_rejects_fixed_action() -> None:
    with pytest.raises(
        PPOPolicySelectionError,
        match="must be null",
    ):
        PPOPolicySelector(
            model=_model(
                action_count=3
            ),
            policy_mode="probability_threshold",
            seed=1,
            threshold_action=1,
            probability_threshold=0.50,
        )


def test_threshold_mode_requires_action_one() -> None:
    with pytest.raises(
        PPOPolicySelectionError,
        match="requires threshold_action=1",
    ):
        PPOPolicySelector(
            model=_model(),
            policy_mode="probability_threshold",
            seed=1,
            threshold_action=0,
            probability_threshold=0.50,
        )
